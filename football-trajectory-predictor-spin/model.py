"""Two-observation soccer predictor. SI units; r0 is treated as known exactly.

No noise is added during forward posterior propagation. Uniform rounding uses
an interval likelihood, not an extra Gaussian error or perturb-and-refit scheme.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize


PARAMETER_NAMES = ('vx0', 'vy0', 'vz0', 'omega_x0', 'omega_y0', 'omega_z0')
# Trionda A: sparse nominal approximation of Asai et al. (2026), Fig. 3.
# Team 471: smooth-sphere reference from their Appendix (not a match-ball curve).
DRAG_NODES = {
    'trionda': ((7., 10., 12., 15., 20., 25., 30., 35.), (.46, .34, .168, .23, .22, .22, .23, .235)),
    'team471': ((0., 17.5, 20., 22.5, 25., 27.5, 30.), (.5, .5, .45, .4, .2, .1, .06)),
}
_DRAG_CURVES = {key: PchipInterpolator(*nodes) for key, nodes in DRAG_NODES.items()}
_DRAG_SLOPES = {key: curve.derivative() for key, curve in _DRAG_CURVES.items()}


@dataclass(frozen=True)
class Physics:
    m: float = .43
    R: float = .11
    rho: float = 1.20
    g: float = 9.81
    spin_decay: float = .10
    drag_profile: str = 'trionda'

    def __post_init__(self):
        if not np.isfinite(self.spin_decay) or self.spin_decay < 0:
            raise ValueError('Spin decay must be nonnegative and finite.')
        if self.drag_profile not in DRAG_NODES:
            raise ValueError('Unknown drag profile.')

    @property
    def magnus(self):
        """Coefficient of omega cross u in acceleration, from Team 480."""
        return 8 * np.pi * self.rho * self.R**3 / (3 * self.m)

    @property
    def drag_factor(self):
        return self.rho * np.pi * self.R**2 / (2 * self.m)


def drag_coefficient(speed, profile='trionda', *, derivative=False):
    speeds, values = DRAG_NODES[profile]
    if speed <= speeds[0]:
        return 0. if derivative else values[0]
    if speed >= speeds[-1]:
        return 0. if derivative else values[-1]
    curve = _DRAG_SLOPES[profile] if derivative else _DRAG_CURVES[profile]
    return float(curve(speed))


def prior_spec(name, direction):
    """Gaussian shrinkage along the early flight direction, in inertial axes."""
    if name not in ('wide', 'strong'):
        raise ValueError('prior must be wide or strong')
    parallel_sd, transverse_sd = (10., 35.) if name == 'wide' else (5., 20.)
    parallel = np.outer(direction, direction)
    transverse = np.eye(3) - parallel
    whitener = np.diag([.1, .1, .2, 0., 0., 0.])
    whitener[3:, 3:] = parallel / parallel_sd + transverse / transverse_sd
    return dict(mean=np.array([30., 0., 6., 0., 0., 0.]),
                whitener=whitener, direction=direction,
                spin_sd_parallel=parallel_sd, spin_sd_transverse=transverse_sd,
                lo=np.array([1., -35., -15., -120., -120., -120.]),
                hi=np.array([60., 35., 30., 120., 120., 120.]))


def validate_input(r0, times, obs):
    r0, times, obs = np.asarray(r0, float), np.asarray(times, float), np.asarray(obs, float)
    if r0.shape != (3,) or times.shape != (2,) or obs.shape != (2, 3):
        raise ValueError('Expected r0 (3,), times (2,), observations (2,3)')
    if not all(np.isfinite(x).all() for x in (r0, times, obs)):
        raise ValueError('Inputs must be finite')
    if not 0 < times[0] < times[1]:
        raise ValueError('Require 0 < t1 < t2')
    if np.linalg.norm(obs[1] - r0) < 1e-9:
        raise ValueError('Reference flight direction is undefined')
    return r0, times, obs


def flight_direction(r0, obs, times, wind=(0, 0, 0)):
    relative = (obs[-1] - r0) / times[-1] - np.asarray(wind)
    norm = np.linalg.norm(relative)
    if norm < 1e-9:
        raise ValueError('Early relative flight direction is undefined')
    return relative / norm


def rhs(t, state, physics=Physics(), wind=(0, 0, 0), fixed_cd=None):
    v = state[3:6]
    omega = state[6:9]
    u = v - wind
    speed = float(np.linalg.norm(u))
    cd = drag_coefficient(speed, physics.drag_profile) if fixed_cd is None else fixed_cd
    wx, wy, wz = omega
    ux, uy, uz = u
    magnus = physics.magnus * np.array([wy*uz - wz*uy, wz*ux - wx*uz, wx*uy - wy*ux])
    acc = -physics.drag_factor * cd * speed * u + magnus
    acc[2] -= physics.g
    return np.r_[v, acc, -physics.spin_decay * omega]


def predict_positions(theta, r0, times, physics=Physics(), wind=(0, 0, 0),
                      rtol=2e-8, atol=1e-10, reject_ground=False, fixed_cd=None):
    """Return ball-centre coordinates at the requested times in seconds."""
    theta = np.asarray(theta)
    if reject_ground and (r0[2] < physics.R-1e-9 or (r0[2] <= physics.R+1e-9 and theta[2] <= 0)):
        raise ValueError('Trajectory is grounded before observations')
    def ground(t,s,*args): return s[2]-physics.R
    ground.direction=-1;ground.terminal=True
    sol = solve_ivp(rhs, (0, float(np.max(times))), np.r_[r0, theta],
                    args=(physics, wind, fixed_cd),
                    t_eval=np.asarray(times), rtol=rtol, atol=atol, method='DOP853',
                    events=ground if reject_ground else None)
    if reject_ground and len(sol.t_events[0]):
        raise ValueError('Trajectory hits ground before observations')
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:3].T


def propagate(theta, r0, physics=Physics(), wind=(0, 0, 0), target_x=11.,
              max_time=5., rtol=1e-10, atol=1e-12, fixed_cd=None):
    """Find the first forward crossing and report earlier contact with the ground."""
    theta, r0 = np.asarray(theta), np.asarray(r0)
    fail = dict(tg=float('nan'), yg=float('nan'), zg=float('nan'))
    if target_x <= r0[0]:
        raise ValueError('Target plane must be ahead of initial x')
    if r0[2] < physics.R - 1e-9 or (r0[2] <= physics.R + 1e-9 and theta[2] <= 0):
        return dict(status='ground', **fail)
    def goal(t, s, *args): return s[0] - target_x
    def ground(t, s, *args): return s[2] - physics.R
    goal.direction = 1; goal.terminal = True
    ground.direction = -1; ground.terminal = True
    sol = solve_ivp(rhs, (0, max_time), np.r_[r0, theta],
                    args=(physics, wind, fixed_cd),
                    events=(goal, ground), rtol=rtol, atol=atol, max_step=.05, method='DOP853')
    if not sol.success: raise RuntimeError(sol.message)
    if len(sol.t_events[0]):
        s = sol.y_events[0][0]
        return dict(status='crossed', tg=float(sol.t_events[0][0]), yg=float(s[1]), zg=float(s[2]),
                    speed_goal=float(np.linalg.norm(s[3:6])))
    return dict(status='ground' if len(sol.t_events[1]) else 'timeout', **fail)


def fit(r0, times, obs, noise='gaussian', prior='wide', model='full', fixed_cd=None,
        sigma=.01, halfwidth=.005, regularize=True, physics=Physics(), wind=(0,0,0)):
    """Fit release velocity and active aerodynamic parameters to two observations."""
    r0, times, obs = validate_input(r0, times, obs)
    if noise not in ('gaussian', 'rounding'): raise ValueError('Unknown noise')
    if not np.isfinite([sigma, halfwidth]).all() or sigma <= 0 or halfwidth <= 0: raise ValueError('Observation scales must be positive')
    if model not in ('full','drag','gravity'): raise ValueError('Unknown model')
    direction = flight_direction(r0, obs, times, wind)
    spec = prior_spec(prior, direction)
    active = list(range(6 if model == 'full' else 3))
    fixed_cd = 0. if model == 'gravity' else fixed_cd
    v = np.sum(times[:, None] * (obs - r0 + .5 * physics.g * times[:, None]**2
                                * np.array([0, 0, 1])), axis=0) / np.sum(times**2)
    start = spec['mean'].copy()
    start[:3] = v
    scale = sigma if noise == 'gaussian' else halfwidth / np.sqrt(3)

    def expand(q):
        theta = np.zeros(6)
        theta[active] = q
        return theta

    def error(theta):
        return predict_positions(theta, r0, times, physics, wind, fixed_cd=fixed_cd) - obs

    def prior_residual(theta):
        return (spec['whitener'] @ (theta - spec['mean']))[active]

    def residual(q):
        theta = expand(q)
        dr = error(theta).ravel() / scale
        return np.r_[dr, prior_residual(theta)] if regularize else dr

    lo, hi = spec['lo'][active], spec['hi'][active]
    scales = np.linalg.norm(np.linalg.inv(spec['whitener']), axis=1)[active]
    opt = least_squares(residual, np.clip(start[active], lo + 1e-8, hi - 1e-8),
                        bounds=(lo, hi), x_scale=scales, ftol=1e-11, xtol=1e-11, gtol=1e-10)
    # Rounding MAP: exact feasible box and truncated Gaussian prior.
    if noise == 'rounding':
        def constraint(q): return np.r_[halfwidth - error(expand(q)).ravel(),
                                        halfwidth + error(expand(q)).ravel()]
        def objective(q): return .5*np.sum(prior_residual(expand(q))**2) if regularize else 0.
        res = minimize(objective,opt.x,method='SLSQP',bounds=list(zip(lo,hi)),
                       constraints={'type':'ineq','fun':constraint},options={'maxiter':300,'ftol':1e-11})
        if np.min(constraint(res.x)) < -1e-7:
            raise ValueError('No compatible interval fit: rounding likelihood is zero')
        point = expand(res.x)
    else: point = expand(opt.x)
    # Standardized observational Jacobian (no prior rows) for identifiability.
    J = opt.jac[:6]
    prior_transform = np.linalg.inv(spec['whitener'][np.ix_(active, active)])
    singular = np.linalg.svd(J @ prior_transform, compute_uv=False)
    cov = np.zeros((6,6)); cov[np.ix_(active,active)] = np.linalg.pinv(opt.jac.T@opt.jac)
    return dict(theta=point, prior=spec, fixed_cd=fixed_cd, jac=opt.jac,
                posterior_cov=cov, active=active, residual_m=error(point),
                singular_values=singular,condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf'),
                success=bool(opt.success),surrogate_center=expand(opt.x))


def log_posterior(theta, r0, times, obs, p, noise, sigma=.01, halfwidth=.005,
                  physics=Physics(), wind=(0, 0, 0)):
    theta = np.asarray(theta)
    if np.any(theta < p['lo']) or np.any(theta > p['hi']): return -np.inf
    try:
        err=predict_positions(theta, r0, times, physics, wind, reject_ground=True)-obs
    except ValueError:
        return -np.inf
    lp=-.5*np.sum((p['whitener'] @ (theta - p['mean']))**2)
    if noise == 'rounding': return lp if np.max(np.abs(err)) <= halfwidth else -np.inf
    return lp-.5*np.sum((err/sigma)**2)


def chain_diagnostics(chains):
    # Classic split Rhat and initial-positive-sequence ESS; diagnostics, not a proof.
    m,n,d=chains.shape; n2=n//2
    x=np.concatenate([chains[:,:n2],chains[:,-n2:]],axis=0); n=n2; m=len(x)
    W=x.var(axis=1,ddof=1).mean(axis=0); between=n*x.mean(axis=1).var(axis=0,ddof=1)
    varhat=(n-1)/n*W+between/n
    rhat=np.sqrt(np.divide(varhat, W, out=np.full(d, np.inf), where=W > 0))
    ess=[]
    for j in range(d):
        if W[j] <= 0:
            ess.append(0.)
            continue
        xc=x[:,:,j]-x[:,:,j].mean(axis=1)[:,None]
        fft=np.fft.rfft(xc,n=2*n,axis=1)
        ac=np.fft.irfft(fft*np.conjugate(fft),axis=1)[:,:n].mean(axis=0)/np.arange(n,0,-1)
        rho=1-(W[j]-ac)/max(varhat[j],1e-30)
        pairs=[]
        for k in range(1,n-1,2):
            val=rho[k]+rho[k+1]
            if val < 0: break
            pairs.append(min(val,pairs[-1]) if pairs else val)
        ess.append(min(m*n,m*n/max(1.,1+2*sum(pairs))))
    return dict(split_rhat=rhat.tolist(),ess=ess)


def _rounding_proposal(fitted, r0, times, obs, transform, halfwidth, physics, wind):
    """Find an interior point and scale the proposal across narrow interval faces."""
    origin = fitted['theta']
    spec = fitted['prior']

    def faces(z):
        theta = origin + transform @ z
        residual = (predict_positions(theta, r0, times, physics, wind) - obs).ravel() / halfwidth
        return np.r_[1 - residual, 1 + residual]

    def constraints(q):
        theta = origin + transform @ q[:6]
        return np.r_[faces(q[:6]) - q[-1], theta - spec['lo'], spec['hi'] - theta]

    result = minimize(lambda q: -q[-1] + .0001 * (q[:6] @ q[:6]), np.zeros(7),
                      method='SLSQP', bounds=[(-4., 4.)] * 6 + [(None, 1.)],
                      constraints={'type': 'ineq', 'fun': constraints},
                      options={'ftol': 1e-10, 'maxiter': 200})
    z = result.x[:6]
    slack = faces(z)
    if slack.min() <= 1e-8:
        raise ValueError('Cannot locate an interior point for the rounding likelihood.')
    step = np.eye(6) * 1e-4
    gradients = np.column_stack([(faces(z + delta) - faces(z - delta)) / 2e-4
                                 for delta in step])
    scaled_gradients = gradients / slack[:, None]
    precision = np.eye(6) + scaled_gradients.T @ scaled_gradients
    chol = np.linalg.cholesky(precision)
    return origin + transform @ z, transform @ np.linalg.solve(chol.T, np.eye(6))


def sample_posterior(r0,times,obs,noise='gaussian',prior='wide',draws=4000,seed=20261107,
                     burn=1500, thin=2, sigma=.01, halfwidth=.005, physics=Physics(),
                     wind=(0, 0, 0)):
    """Retain a total of draws across four chains in PARAMETER_NAMES order."""
    if draws < 20 or draws % 4 or burn < 0 or thin < 1:
        raise ValueError('Use draws >= 20 divisible by four, nonnegative burn and positive thin.')
    r0,times,obs=validate_input(r0,times,obs)
    f=fit(r0,times,obs,noise=noise,prior=prior,sigma=sigma, halfwidth=halfwidth, physics=physics, wind=wind)
    center = f['surrogate_center']
    L = np.linalg.cholesky(f['posterior_cov'] + np.eye(6) * 1e-12)
    if noise == 'rounding':
        center, L = _rounding_proposal(f, r0, times, obs, L, halfwidth, physics, wind)
    rng=np.random.default_rng(seed); count=draws//4; chains=[]; rates=[]
    def lp(z):
        return log_posterior(center + L @ z, r0, times, obs, f['prior'], noise,
                             sigma, halfwidth, physics, wind)
    for c in range(4):
        z = np.zeros(6) if noise == 'rounding' else np.linalg.solve(L, f['theta'] - center)
        val = lp(z)
        # Use a separate feasible starting point for each chain.
        for _ in range(5000):
            cand=rng.normal(size=6)
            if np.isfinite(lp(cand)): z=cand;val=lp(z);break
        if not np.isfinite(val): raise RuntimeError('Cannot initialize feasible posterior chain')
        step=.65 if noise=='rounding' else .9; saved=[]; accepted=0; window=0
        for it in range(burn+count*thin):
            proposal=z+step*rng.normal(size=6); new=lp(proposal)
            if np.log(rng.random()) < new-val:
                z=proposal; val=new; accepted+=1; window+=1
            if it < burn and (it+1)%100==0:
                step*=np.exp(np.clip(window/100-.25,-.25,.25));window=0
            if it>=burn and (it-burn)%thin==0: saved.append(center+L@z)
        chains.append(saved); rates.append(accepted/(burn+count*thin))
    chains=np.asarray(chains); diag=chain_diagnostics(chains)
    diag.update(draws=draws, draws_per_chain=count, acceptance=rates,seed=seed,burn_per_chain=burn,thin=thin,chains=4,
                sampler='whitened random-walk Metropolis; adaptation only during burn-in',
                reliable=bool(max(diag['split_rhat'])<1.05 and min(diag['ess'])>=200))
    return dict(samples=chains.reshape(-1,6),chains=chains,diagnostics=diag,fit=f)
