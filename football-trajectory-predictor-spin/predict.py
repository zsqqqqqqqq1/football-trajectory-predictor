"""Predict a football's crossing of a target plane from two 3D observations."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ekf import STATE_NAMES, ekf_predict
from model import DRAG_NODES, PARAMETER_NAMES, Physics, propagate, sample_posterior, validate_input
from run_analysis import clean, save_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSON file containing r0, times and observations')
    parser.add_argument('--output', type=Path, default=Path('results/prediction'),
                        help='directory for the prediction files')
    parser.add_argument('--method', choices=['mcmc', 'ekf'], default='mcmc')
    parser.add_argument('--cd-profile', choices=DRAG_NODES, default='trionda')
    parser.add_argument('--spin-decay', type=float, default=Physics.spin_decay,
                        help='fixed exponential spin decay rate in 1/s')
    parser.add_argument('--noise', choices=['gaussian', 'rounding'], default='gaussian')
    parser.add_argument('--prior', choices=['wide', 'strong'], default='wide')
    parser.add_argument('--sigma', type=float, default=.01, help='Gaussian standard deviation per axis (m)')
    parser.add_argument('--halfwidth', type=float, default=.005, help='rounding interval half-width (m)')
    parser.add_argument('--draws', type=int, default=4000, help='total posterior draws (split across four chains for MCMC)')
    parser.add_argument('--burn', type=int, default=1500, help='warm-up steps per chain')
    parser.add_argument('--thin', type=int, default=2, help='MCMC iterations between retained draws')
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    if not np.isfinite([args.sigma, args.halfwidth]).all() or min(args.sigma, args.halfwidth) <= 0:
        parser.error('Observation scales must be finite and positive.')
    if args.draws < 20 or (args.method == 'mcmc' and args.draws % 4) or args.burn < 0 or args.seed < 0:
        parser.error('Use at least 20 draws, divisible by four for MCMC, and nonnegative burn and seed.')
    if args.thin < 1:
        parser.error('--thin must be positive.')
    if args.method == 'ekf' and args.noise != 'gaussian':
        parser.error('The EKF requires Gaussian observation noise.')
    try:
        data = json.loads(args.input.read_text(encoding='utf-8-sig'))
        r0, times, observations = validate_input(data['r0'], data['times'], data['observations'])
        target_x = float(data.get('target_x', 11))
        if not np.isfinite(target_x) or target_x <= r0[0]:
            raise ValueError('target_x must be finite and ahead of the launch point.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Invalid input: {exc}')

    try:
        physics = Physics(spin_decay=args.spin_decay, drag_profile=args.cd_profile)
        metadata = {}
        if args.method == 'ekf':
            result = ekf_predict(r0, times, observations, prior=args.prior, noise_sd=args.sigma,
                                 draws=args.draws, seed=args.seed, target_x=target_x, physics=physics)
            predictions, statuses = result['predictions'], result['statuses']
            payload = dict(states=result['states'], state_names=STATE_NAMES)
            metadata.update(state_mean=result['mean'], state_covariance=result['covariance'],
                            prior_spec=result['prior'])
        else:
            result = sample_posterior(r0, times, observations, noise=args.noise, prior=args.prior,
                                      sigma=args.sigma, halfwidth=args.halfwidth, draws=args.draws,
                                      burn=args.burn, thin=args.thin, seed=args.seed, physics=physics)
            events = [propagate(theta, r0, physics, target_x=target_x) for theta in result['samples']]
            statuses = [event['status'] for event in events]
            predictions = np.array([[event['tg'], event['yg'], event['zg']] for event in events])
            payload = dict(parameters=result['samples'], chains=result['chains'],
                           parameter_names=PARAMETER_NAMES)
            metadata.update(map_theta=result['fit']['theta'], parameter_names=PARAMETER_NAMES,
                            prior_spec=result['fit']['prior'],
                            observation_singular_values=result['fit']['singular_values'])
    except (ValueError, RuntimeError) as exc:
        parser.error(f'Could not fit the trajectory: {exc}')
    summary, grid = summarize(predictions, statuses)
    summary.update(input=data, method=args.method, noise=args.noise, sigma=args.sigma,
                   halfwidth=args.halfwidth, prior=args.prior, diagnostics=result['diagnostics'],
                   physics=asdict(physics), drag_nodes=DRAG_NODES[physics.drag_profile],
                   magnus_coefficient=physics.magnus, **metadata)

    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'summary.json', summary)
    np.savez_compressed(args.output / 'samples.npz', tyz=predictions, statuses=statuses, **payload)
    density_path = args.output / 'density.npz'
    if grid is not None:
        np.savez_compressed(density_path, **grid)
    else:
        density_path.unlink(missing_ok=True)
    print(json.dumps({'output': str(args.output.resolve()), 'p_cross': summary['p_cross'],
                      'diagnostics': clean(result['diagnostics'])}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
