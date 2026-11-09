"""Reproducible synthetic hold-out benchmark; no training on later positions.

Batch uncertainty here uses a truncated local Laplace approximation.
The EKF uses a single nine-dimensional Gaussian state. Regions are empirical 95% covariance ellipses, conditional on crossing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from model import Physics, drag_coefficient, fit, propagate
from ekf import ekf_predict


def region_metrics(predictions, truth):
    """Empirical covariance ellipse containing 95% of crossed predictions."""
    p = np.asarray(predictions, float)
    crossed = np.all(np.isfinite(p), axis=1)
    cloud = p[crossed, 1:3]
    if len(cloud) < 8:
        return dict(error_m=None, covered=False, area_m2=None,
                    cross_probability=float(crossed.mean()), prediction_y=None,
                    prediction_z=None, time_error_s=None)
    center = cloud.mean(axis=0)
    cov = np.cov(cloud, rowvar=False) + np.eye(2) * 1e-12
    inverse = np.linalg.inv(cov)
    diff = cloud - center
    radii = np.einsum('ni,ij,nj->n', diff, inverse, diff)
    q95 = np.quantile(radii, .95)
    delta = np.asarray(truth)[1:] - center
    return dict(error_m=float(np.linalg.norm(delta)),
                covered=bool(delta @ inverse @ delta <= q95),
                area_m2=float(np.pi * q95 * np.sqrt(np.linalg.det(cov))),
                cross_probability=float(crossed.mean()),
                prediction_y=float(center[0]), prediction_z=float(center[1]),
                time_error_s=float(p[crossed, 0].mean() - truth[0]))


def batch_laplace(r0, times, obs, draws, seed, prior='wide', target_x=11.):
    solution = fit(r0, times, obs, noise='gaussian', prior=prior, model='full')
    if not solution['success']:
        raise RuntimeError('MAP optimizer did not converge.')
    theta = np.asarray(solution['theta'])
    jac = np.asarray(solution['jac'])
    covariance = np.linalg.pinv(jac.T @ jac, rcond=1e-12)
    eigval, eigvec = np.linalg.eigh(covariance)
    transform = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 0)))
    lo, hi = solution['prior']['lo'], solution['prior']['hi']
    rng = np.random.default_rng(seed)
    accepted, attempted = [], 0
    while len(accepted) < draws and attempted < draws * 500:
        chunk = theta + rng.normal(size=(max(100, draws), 6)) @ transform.T
        valid = np.all((chunk >= lo) & (chunk <= hi), axis=1)
        accepted.extend(chunk[valid])
        attempted += len(chunk)
    if len(accepted) < draws:
        raise RuntimeError('Insufficient bounded Laplace samples.')
    predictions = []
    for sample in accepted[:draws]:
        result = propagate(sample, r0, target_x=target_x)
        predictions.append([result.get(k, np.nan) for k in ('tg', 'yg', 'zg')])
    return np.asarray(predictions), solution


def make_truth(theta, times, scenario, r0, physics):
    """Independent generator; scenarios are intentionally absent from inversion."""
    wind = np.array([0., 4., 0.]) if scenario == 'crosswind' else np.zeros(3)
    decay = 1.2 if scenario == 'fast_spin_decay' else physics.spin_decay

    def flight(t, state):
        u = state[3:6] - wind
        speed = np.linalg.norm(u)
        cd = drag_coefficient(speed, physics.drag_profile)
        if scenario == 'higher_drag':
            cd *= 1.3
        spin = theta[3:] * np.exp(-decay * t)
        acceleration = (np.array([0., 0., -physics.g]) - physics.drag_factor * cd * speed * u
                        + physics.magnus * np.cross(spin, u))
        return np.r_[state[3:6], acceleration]

    def goal(t, state):
        return state[0] - 11

    def ground(t, state):
        return state[2] - physics.R

    goal.terminal, goal.direction = True, 1
    ground.terminal, ground.direction = True, -1
    result = solve_ivp(flight, (0., 3.), np.r_[r0, theta[:3]], dense_output=True,
                       events=[goal, ground], rtol=2e-10, atol=1e-12, max_step=.01)
    if not result.success:
        raise RuntimeError(result.message)
    if not len(result.t_events[0]) or result.t_events[0][0] <= times[-1]:
        return None
    crossing = np.r_[result.t_events[0][0], result.y_events[0][0, 1:3]]
    return result.sol(times)[:3].T, crossing


def wilson(successes, total):
    if total == 0:
        return [None, None]
    z, p = 1.95996398454, successes / total
    d = 1 + z*z / total
    center = (p + z*z / (2*total)) / d
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [float(center-half), float(center+half)]


def run_benchmark(output, shots=12, draws=4000, seed=20261107):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    physics, r0 = Physics(), np.array([0., 0., .11])
    scenarios = ['same_model', 'higher_drag', 'crosswind', 'fast_spin_decay']
    rows, generated, rejected = [], [], []
    for shot in range(shots):
        theta = np.r_[rng.uniform(27, 34), rng.uniform(-6, 6), rng.uniform(5, 9),
                      rng.uniform(-25., 25., 3)]
        times = np.array([.05, .10])
        noise = rng.normal(0, .01, (2, 3))
        for scenario in scenarios:
            truth = make_truth(theta, times, scenario, r0, physics)
            if truth is None:
                rejected.append({'shot': shot, 'scenario': scenario, 'reason': 'ground_before_goal'})
                continue
            clean, reference = truth
            obs = clean + noise
            generated.append(dict(shot=shot, scenario=scenario, theta=theta.tolist(),
                                  r0=r0.tolist(), times=times.tolist(), observations=obs.tolist(),
                                  clean_observations=clean.tolist(), reference=reference.tolist()))
            shared_seed = seed + shot * 100 + scenarios.index(scenario)
            for method in ['batch_laplace', 'ekf_9d']:
                try:
                    if method == 'batch_laplace':
                        predictions, fitted = batch_laplace(r0, times, obs, draws, shared_seed)
                    else:
                        result = ekf_predict(r0, times, obs, draws=draws, seed=shared_seed)
                        predictions = result['predictions']
                    metrics = region_metrics(predictions, reference)
                    metrics['exception'] = ''
                except Exception as exc:
                    metrics = dict(error_m=None, covered=False, area_m2=None,
                                   cross_probability=None, prediction_y=None, prediction_z=None,
                                   time_error_s=None, exception=f'{type(exc).__name__}: {exc}')
                rows.append(dict(shot=shot, scenario=scenario, method=method,
                                 t1=times[0], t2=times[1], truth_t=reference[0],
                                 truth_y=reference[1], truth_z=reference[2],
                                 **metrics))
            print(f'synthetic shot={shot+1}/{shots} scenario={scenario}', flush=True)
    summaries = []
    for scenario in scenarios:
        for method in ['batch_laplace', 'ekf_9d']:
            group = [r for r in rows if r['scenario'] == scenario and r['method'] == method]
            valid = [r for r in group if r['error_m'] is not None]
            errors = [r['error_m'] for r in valid]
            count = sum(r['covered'] for r in group)
            summaries.append(dict(scenario=scenario, method=method, n=len(group),
                                  inference_failures=len(group)-len(valid),
                                  mean_error_m=float(np.mean(errors)) if errors else None,
                                  rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                                  coverage_95=count/len(group) if group else None,
                                  coverage_wilson_95=wilson(count, len(group)),
                                  mean_cross_probability=float(np.mean([r['cross_probability'] for r in valid])) if valid else None,
                                  mean_region_area_m2=float(np.mean([r['area_m2'] for r in valid])) if valid else None))
    with (output / 'synthetic_per_shot.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(seed=seed, shots_per_scenario=shots, posterior_draws=draws,
                  observation_sd_m=.01,
                  region='Empirical 95% covariance ellipse, conditional on crossing',
                  batch_approximation='MAP and inverse augmented-Jacobian Hessian; bounded Laplace draws',
                  ekf_approximation='First-order state Gaussian in nine dimensions',
                  limitations=['Small fixed-design test, not proof of 95% calibration',
                               'Matched test theta sampled from stated plausible ranges, not full prior',
                               'All methods share each truth, timestamps, and noisy coordinates',
                               'Four scenarios share launch/noise seeds; scenario estimates are correlated',
                               'R0 and timestamps are exact; measurement noise applied once to r1/r2'],
                  mismatch=dict(higher_drag='Cd(speed)=1.3*nominal Cd(speed)',
                                crosswind='wind=(0,4,0) m/s',
                                fast_spin_decay='spin decay = 1.2 /s'),
                  excluded=rejected, summaries=summaries)
    (output / 'synthetic_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output / 'synthetic_cases.json').write_text(json.dumps(generated, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--shots', type=int, default=12)
    parser.add_argument('--draws', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    if args.shots < 1 or args.draws < 20 or args.seed < 0:
        parser.error('Require positive shots, at least 20 draws and nonnegative seed.')
    run_benchmark(args.output, args.shots, args.draws, args.seed)
