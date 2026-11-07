"""Two-observation soccer predictor. SI units; r0 is treated as known exactly.

No noise is added during forward posterior propagation. Uniform rounding uses
an interval likelihood, not an extra Gaussian error or perturb-and-refit scheme.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize


@dataclass(frozen=True)
class Physics:
    m: float = .43
    R: float = .11
    rho: float = 1.20
    g: float = 9.81


def prior_spec(name='wide'):
    """Return the prior means, scales and bounds in parameter-vector order."""
    if name not in ('wide', 'strong'):
        raise ValueError('prior must be wide or strong')
    return dict(mean=np.array([30., 0., 6., .30 if name == 'wide' else .25, 0., 0.]),
                sd=np.array([10., 10., 5., .20 if name == 'wide' else .06,
                             .25 if name == 'wide' else .15, .25 if name == 'wide' else .15]),
                lo=np.array([1., -35., -15., .05, -.6, -.6]),
                hi=np.array([60., 35., 30., .65, .6, .6]))


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


def basis_from_observations(r0, obs):
    """Return two orthonormal axes perpendicular to the early flight direction."""
    d = np.asarray(obs, float)[-1] - np.asarray(r0, float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        raise ValueError('Reference direction is undefined')
    d /= norm
    axis = np.eye(3)[np.argmin(np.abs(d))]
    e1 = np.cross(d, axis); e1 /= np.linalg.norm(e1)
    return np.column_stack([e1, np.cross(d, e1)])


def rhs(t, state, cd, b, physics=Physics(), wind=(0, 0, 0)):
    u = state[3:] - np.asarray(wind)
    a = physics.rho * np.pi * physics.R**2 / (2 * physics.m) * np.linalg.norm(u) * (-cd * u + np.cross(b, u))
    a[2] -= physics.g
    return np.r_[state[3:], a]


def predict_positions(theta, r0, times, B, physics=Physics(), wind=(0, 0, 0), rtol=2e-8, atol=1e-10,
                      reject_ground=False):
    """Return ball-centre coordinates at the requested times in seconds."""
    theta = np.asarray(theta)
    if reject_ground and (r0[2] < physics.R-1e-9 or (r0[2] <= physics.R+1e-9 and theta[2] <= 0)):
        raise ValueError('Trajectory is grounded before observations')
    def ground(t,s,*args): return s[2]-physics.R
    ground.direction=-1;ground.terminal=True
    sol = solve_ivp(rhs, (0, float(np.max(times))), np.r_[r0, theta[:3]],
                    args=(theta[3], B @ theta[4:], physics, wind),
                    t_eval=np.asarray(times), rtol=rtol, atol=atol, method='DOP853',
                    events=ground if reject_ground else None)
    if reject_ground and len(sol.t_events[0]):
        raise ValueError('Trajectory hits ground before observations')
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:3].T


def propagate(theta, r0, B, physics=Physics(), wind=(0, 0, 0), target_x=11.,
              max_time=5., rtol=1e-10, atol=1e-12):
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
    sol = solve_ivp(rhs, (0, max_time), np.r_[r0, theta[:3]],
                    args=(theta[3], B @ theta[4:], physics, wind),
                    events=(goal, ground), rtol=rtol, atol=atol, max_step=.05, method='DOP853')
    if not sol.success: raise RuntimeError(sol.message)
    if len(sol.t_events[0]):
        s = sol.y_events[0][0]
        return dict(status='crossed', tg=float(sol.t_events[0][0]), yg=float(s[1]), zg=float(s[2]),
                    speed_goal=float(np.linalg.norm(s[3:])))
    return dict(status='ground' if len(sol.t_events[1]) else 'timeout', **fail)


def fit(r0, times, obs, noise='gaussian', prior='wide', model='full', fixed_cd=None,
        sigma=.01, halfwidth=.005, regularize=True, physics=Physics(), wind=(0,0,0)):
    """Fit release velocity and active aerodynamic parameters to two observations."""
    r0, times, obs = validate_input(r0, times, obs)
    if noise not in ('gaussian', 'rounding'): raise ValueError('Unknown noise')
    if sigma<=0 or halfwidth<=0: raise ValueError('Observation scales must be positive')
    if model not in ('full','drag','gravity'): raise ValueError('Unknown model')
    B = basis_from_observations(r0, obs); spec = prior_spec(prior)
    active = [0,1,2] + ([3] if model != 'gravity' and fixed_cd is None else []) + ([4,5] if model == 'full' else [])
    base = np.zeros(6); base[3] = fixed_cd or 0.
    v = np.sum(times[:,None] * (obs-r0 + .5*physics.g*times[:,None]**2*np.array([0,0,1])), axis=0) / np.sum(times**2)
    start = spec['mean'].copy(); start[:3] = v
    scale = sigma if noise == 'gaussian' else halfwidth/np.sqrt(3)
    def expand(q):
        th = base.copy(); th[active] = q; return th
    def residual(q, with_prior=True):
        th = expand(q)
        dr = (predict_positions(th,r0,times,B,physics,wind)-obs).ravel()/scale
        return np.r_[dr, ((th-spec['mean'])/spec['sd'])[active]] if regularize and with_prior else dr
    lo,hi=spec['lo'][active],spec['hi'][active]
    opt = least_squares(residual,np.clip(start[active],lo+1e-8,hi-1e-8),bounds=(lo,hi),
                        x_scale=spec['sd'][active],ftol=1e-11,xtol=1e-11,gtol=1e-10)
    # Rounding MAP: exact feasible box and truncated Gaussian prior.
    if noise == 'rounding':
        def constraint(q): return halfwidth-np.abs(predict_positions(expand(q),r0,times,B,physics,wind)-obs).ravel()
        def objective(q): return .5*np.sum(((expand(q)-spec['mean'])/spec['sd'])[active]**2) if regularize else 0.
        res = minimize(objective,opt.x,method='SLSQP',bounds=list(zip(lo,hi)),
                       constraints={'type':'ineq','fun':constraint},options={'maxiter':300,'ftol':1e-11})
        if np.min(constraint(res.x)) < -1e-7:
            raise ValueError('No compatible interval fit: rounding likelihood is zero')
        point = expand(res.x)
    else: point = expand(opt.x)
    # Standardized observational Jacobian (no prior rows) for identifiability.
    J = opt.jac[:6]
    singular = np.linalg.svd(J * spec['sd'][active],compute_uv=False)
    cov = np.zeros((6,6)); cov[np.ix_(active,active)] = np.linalg.pinv(opt.jac.T@opt.jac)
    return dict(theta=point,B=B,jac=opt.jac,posterior_cov=cov,active=active,
                residual_m=predict_positions(point,r0,times,B,physics,wind)-obs,
                singular_values=singular,condition_number=float(singular[0]/singular[-1]),
                success=bool(opt.success),surrogate_center=expand(opt.x))


def log_posterior(theta,r0,times,obs,B,noise,prior,sigma=.01,halfwidth=.005):
    p=prior_spec(prior)
    if np.any(theta < p['lo']) or np.any(theta > p['hi']): return -np.inf
    try:
        err=predict_positions(theta,r0,times,B,reject_ground=True)-obs
    except ValueError:
        return -np.inf
    lp=-.5*np.sum(((theta-p['mean'])/p['sd'])**2)
    if noise == 'rounding': return lp if np.max(np.abs(err)) <= halfwidth else -np.inf
    return lp-.5*np.sum((err/sigma)**2)


def chain_diagnostics(chains):
    # Classic split Rhat and initial-positive-sequence ESS; diagnostics, not a proof.
    m,n,d=chains.shape; n2=n//2
    x=np.concatenate([chains[:,:n2],chains[:,-n2:]],axis=0); n=n2; m=len(x)
    W=x.var(axis=1,ddof=1).mean(axis=0); between=n*x.mean(axis=1).var(axis=0,ddof=1)
    varhat=(n-1)/n*W+between/n
    rhat=np.sqrt(varhat/np.maximum(W,1e-30))
    ess=[]
    for j in range(d):
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


def sample_posterior(r0,times,obs,noise='gaussian',prior='wide',draws=1200,seed=20261107,
                     burn=1000,thin=2,sigma=.01,halfwidth=.005):
    """Draw parameter samples in order (vx, vy, vz, Cd, b1, b2) using four chains."""
    r0,times,obs=validate_input(r0,times,obs)
    f=fit(r0,times,obs,noise=noise,prior=prior,sigma=sigma,halfwidth=halfwidth)
    center=f['surrogate_center']; L=np.linalg.cholesky(f['posterior_cov']+np.eye(6)*1e-12)
    rng=np.random.default_rng(seed); count=int(np.ceil(draws/4)); chains=[]; rates=[]
    def lp(z): return log_posterior(center+L@z,r0,times,obs,f['B'],noise,prior,sigma,halfwidth)
    for c in range(4):
        z=np.linalg.solve(L,f['theta']-center); val=lp(z)
        # Interior initial state for hard interval support (MAP may be on boundary).
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
    diag.update(acceptance=rates,seed=seed,burn_per_chain=burn,thin=thin,chains=4,
                sampler='whitened random-walk Metropolis; adaptation only during burn-in',
                reliable=bool(max(diag['split_rhat'])<1.05 and min(diag['ess'])>=200))
    return dict(samples=chains.reshape(-1,6),chains=chains,diagnostics=diag,fit=f)
