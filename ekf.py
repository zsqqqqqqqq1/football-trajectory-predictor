"""Two-update EKF mixture benchmark, with independent initial velocity prior.

This comparator uses Gaussian coordinate errors only. Aerodynamic parameters are
fixed within each filter and reweighted by both innovation likelihoods. It uses
the same parameter prior as model.py; the conditional state distribution is an
EKF Gaussian approximation (including the otherwise truncated velocity prior).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
from scipy.stats import truncnorm

from model import Physics, basis_from_observations, prior_spec, validate_input


def _skew(b):
    x, y, z = b
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _rhs_and_jac(state, cd, b, physics):
    v = state[3:]
    speed = np.linalg.norm(v)
    linear = -cd * np.eye(3) + _skew(b)
    force = linear @ v
    k = physics.rho * np.pi * physics.R**2 / (2 * physics.m)
    acc = k * speed * force + np.array([0., 0., -physics.g])
    jac = np.zeros((6, 6))
    jac[:3, 3:] = np.eye(3)
    jac[3:, 3:] = k * (speed * linear + np.outer(force, v) / max(speed, 1e-12))
    return np.r_[v, acc], jac


def _advance(mean, cov, dt, cd, b, physics):
    def rhs(t, y):
        f, jac = _rhs_and_jac(y[:6], cd, b, physics)
        p = y[6:].reshape(6, 6)
        return np.r_[f, (jac @ p + p @ jac.T).ravel()]
    result = solve_ivp(rhs, [0., dt], np.r_[mean, cov.ravel()],
                       method='DOP853', rtol=1e-8, atol=1e-10)
    if not result.success:
        raise RuntimeError(result.message)
    p = result.y[6:, -1].reshape(6, 6)
    return result.y[:6, -1], (p + p.T) / 2


def _cross(state, t0, cd, b, physics, target_x):
    if state[2] < physics.R - 1e-9 or (state[2] <= physics.R + 1e-9 and state[5] <= 0):
        return 'ground', np.full(3, np.nan)
    def goal(t, y):
        return y[0] - target_x
    goal.direction, goal.terminal = 1, True
    def ground(t, y):
        return y[2] - physics.R
    ground.direction, ground.terminal = -1, True
    result = solve_ivp(lambda t, s: _rhs_and_jac(s, cd, b, physics)[0],
                       [t0, t0 + 5], state, events=[goal, ground],
                       rtol=2e-8, atol=1e-10, max_step=.05)
    if len(result.t_events[0]):
        return 'crossed', np.r_[result.t_events[0][0], result.y_events[0][0, 1:3]]
    return ('ground' if len(result.t_events[1]) else 'timeout'), np.full(3, np.nan)


def _prior_arrays(prior):
    spec = prior_spec(prior)
    if isinstance(spec, dict):
        return tuple(np.asarray(spec[k], float) for k in ('mean', 'sd', 'lo', 'hi'))
    return tuple(np.asarray(x, float) for x in spec)


def ekf_predict(r0, times, observations, prior='wide', noise_sd=.01,
                n_filters=128, draws=500, seed=2026, target_x=11.,
                physics=None):
    """Return prediction samples and filter weights; no early finite differences.

    All returned prediction rows have (absolute time, y, z). Non-crossing rows
    are NaN and have corresponding status labels. The equal-weight aerodynamic
    prior ensemble is updated by the product of the two innovation likelihoods.
    """
    physics = physics or Physics()
    r0, times, observations = validate_input(r0, times, observations)
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError('Require positive finite noise_sd.')
    if not np.isfinite(target_x) or target_x <= r0[0]:
        raise ValueError('Target plane must be finite and ahead of initial x.')
    if n_filters < 2 or draws < 2:
        raise ValueError('Require at least two filters and draws.')
    rng = np.random.default_rng(seed)
    mean, sd, lo, hi = _prior_arrays(prior)
    bmat = basis_from_observations(r0, observations)
    aero = truncnorm.rvs((lo[3:] - mean[3:]) / sd[3:],
                        (hi[3:] - mean[3:]) / sd[3:],
                        loc=mean[3:], scale=sd[3:], size=(n_filters, 3),
                        random_state=rng)
    means, covs, logweights = [], [], []
    noise = np.eye(3) * noise_sd**2
    h = np.c_[np.eye(3), np.zeros((3, 3))]
    for cd, b1, b2 in aero:
        b = bmat @ [b1, b2]
        mu = np.r_[r0, mean[:3]]
        cov = np.diag(np.r_[np.zeros(3), sd[:3]**2])
        logweight, previous = 0., 0.
        for t, obs in zip(times, observations):
            mu, cov = _advance(mu, cov, t - previous, cd, b, physics)
            innovation = obs - h @ mu
            s = h @ cov @ h.T + noise
            _, logdet = np.linalg.slogdet(s)
            logweight += -.5 * (innovation @ np.linalg.solve(s, innovation)
                               + logdet + 3 * np.log(2 * np.pi))
            gain = np.linalg.solve(s, h @ cov).T
            mu += gain @ innovation
            kh = np.eye(6) - gain @ h
            cov = kh @ cov @ kh.T + gain @ noise @ gain.T
            previous = t
        means.append(mu)
        covs.append((cov + cov.T) / 2)
        logweights.append(logweight)
    weights = np.exp(np.asarray(logweights) - logsumexp(logweights))
    means, covs = np.asarray(means), np.asarray(covs)
    component = rng.choice(n_filters, draws, p=weights)
    predictions, statuses = [], []
    for i in component:
        eigval, eigvec = np.linalg.eigh(covs[i])
        state = means[i] + eigvec @ (np.sqrt(np.maximum(eigval, 0)) * rng.normal(size=6))
        cd, b1, b2 = aero[i]
        status, row = _cross(state, times[-1], cd, bmat @ [b1, b2], physics, target_x)
        statuses.append(status)
        predictions.append(row)
    return {'predictions': np.asarray(predictions), 'statuses': np.asarray(statuses),
            'weights': weights, 'aero_samples': aero, 'B': bmat,
            'means': means, 'covs': covs,
            'diagnostics': {'aero_effective_sample_size': float(1 / (weights @ weights)),
                            'n_filters': n_filters, 'draws': draws,
                            'observation_model': 'Gaussian independent axes',
                            'state_approximation': 'EKF Gaussian, velocity bounds not truncated',
                            'velocity_prior_mean': mean[:3].tolist(),
                            'velocity_prior_sd': sd[:3].tolist()}}
