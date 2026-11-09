"""Nine-state extended Kalman filter with Gaussian position errors."""
import numpy as np
from scipy.integrate import solve_ivp

from model import Physics, drag_coefficient, flight_direction, prior_spec, propagate, rhs, validate_input


STATE_NAMES = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'omega_x', 'omega_y', 'omega_z')


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _rhs_and_jac(state, physics, wind=(0, 0, 0)):
    u = state[3:6] - wind
    speed = np.linalg.norm(u)
    cd = drag_coefficient(speed, physics.drag_profile)
    dcd = drag_coefficient(speed, physics.drag_profile, derivative=True)
    eye = np.eye(3)
    drag_jac = cd * speed * eye
    if speed > 0:
        drag_jac += (cd / speed + dcd) * np.outer(u, u)
    jac = np.zeros((9, 9))
    jac[:3, 3:6] = eye
    jac[3:6, 3:6] = -physics.drag_factor * drag_jac + physics.magnus * _skew(state[6:9])
    jac[3:6, 6:9] = -physics.magnus * _skew(u)
    jac[6:9, 6:9] = -physics.spin_decay * eye
    return rhs(0, state, physics, wind), jac


def _advance(mean, cov, dt, physics, wind):
    def augmented_rhs(t, state):
        f, jac = _rhs_and_jac(state[:9], physics, wind)
        p = state[9:].reshape(9, 9)
        return np.r_[f, (jac @ p + p @ jac.T).ravel()]

    result = solve_ivp(augmented_rhs, (0., dt), np.r_[mean, cov.ravel()],
                       method='DOP853', rtol=1e-8, atol=1e-10)
    if not result.success:
        raise RuntimeError(result.message)
    p = result.y[9:, -1].reshape(9, 9)
    return result.y[:9, -1], (p + p.T) / 2


def _cross(state, t0, physics, target_x, wind=(0, 0, 0)):
    if state[0] >= target_x:
        return 'past_target', np.full(3, np.nan)
    event = propagate(state[3:], state[:3], physics, wind, target_x=target_x,
                      rtol=2e-8, atol=1e-10)
    return event['status'], np.array([t0 + event['tg'], event['yg'], event['zg']])


def ekf_predict(r0, times, observations, prior='wide', noise_sd=.01,
                draws=4000, seed=2026, target_x=11., physics=Physics(),
                wind=(0, 0, 0)):
    """Assimilate two positions and propagate draws from the last state Gaussian."""
    r0, times, observations = validate_input(r0, times, observations)
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError('Require positive finite noise_sd.')
    if not np.isfinite(target_x) or target_x <= max(r0[0], *observations[:, 0]):
        raise ValueError('EKF requires a target plane ahead of the release and both observations.')
    if draws < 2:
        raise ValueError('Require at least two draws.')
    spec = prior_spec(prior, flight_direction(r0, observations, times, wind))
    prior_transform = np.linalg.inv(spec['whitener'])
    mean = np.r_[r0, spec['mean']]
    cov = np.zeros((9, 9))
    cov[3:, 3:] = prior_transform @ prior_transform.T
    measurement_cov = np.eye(3) * noise_sd**2
    h = np.c_[np.eye(3), np.zeros((3, 6))]
    previous = 0.
    for time, obs in zip(times, observations):
        mean, cov = _advance(mean, cov, time - previous, physics, wind)
        innovation = obs - h @ mean
        innovation_cov = h @ cov @ h.T + measurement_cov
        gain = np.linalg.solve(innovation_cov, h @ cov).T
        mean += gain @ innovation
        remaining = np.eye(9) - gain @ h
        cov = remaining @ cov @ remaining.T + gain @ measurement_cov @ gain.T
        previous = time

    cov = (cov + cov.T) / 2
    eigval, eigvec = np.linalg.eigh(cov)
    transform = eigvec * np.sqrt(np.maximum(eigval, 0))
    rng = np.random.default_rng(seed)
    states = mean + rng.normal(size=(draws, 9)) @ transform.T
    statuses, predictions = zip(*(_cross(state, times[-1], physics, target_x, wind)
                                  for state in states))
    return dict(predictions=np.asarray(predictions), statuses=np.asarray(statuses),
                mean=mean, covariance=cov, states=states, prior=spec,
                diagnostics=dict(draws=draws, seed=seed, observation_model='Gaussian independent axes',
                                 state_time_s=float(times[-1]), state_dimension=9,
                                 state_approximation='EKF Gaussian without hard parameter bounds'))

