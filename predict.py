"""Predict a football's crossing of a target plane from two 3D observations."""

import argparse
import json
from pathlib import Path

import numpy as np

from model import propagate, sample_posterior, validate_input
from run_analysis import save_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSON file containing r0, times and observations')
    parser.add_argument('--output', type=Path, default=Path('results/prediction'),
                        help='directory for the prediction files')
    parser.add_argument('--noise', choices=['gaussian', 'rounding'], default='gaussian')
    parser.add_argument('--prior', choices=['wide', 'strong'], default='wide')
    parser.add_argument('--sigma', type=float, default=.01, help='Gaussian standard deviation per axis (m)')
    parser.add_argument('--halfwidth', type=float, default=.005, help='rounding interval half-width (m)')
    parser.add_argument('--draws', type=int, default=4000, help='total retained samples across four chains')
    parser.add_argument('--burn', type=int, default=1500, help='warm-up steps per chain')
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    if not np.isfinite([args.sigma, args.halfwidth]).all() or min(args.sigma, args.halfwidth) <= 0:
        parser.error('Observation scales must be finite and positive.')
    if args.draws < 20 or args.draws % 4 or args.burn < 0 or args.seed < 0:
        parser.error('Use a multiple of 4 for --draws (at least 20), and nonnegative burn and seed values.')
    try:
        data = json.loads(args.input.read_text(encoding='utf-8-sig'))
        r0, times, observations = validate_input(data['r0'], data['times'], data['observations'])
        target_x = float(data.get('target_x', 11))
        if not np.isfinite(target_x) or target_x <= r0[0]:
            raise ValueError('target_x must be finite and ahead of the launch point.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Invalid input: {exc}')

    try:
        result = sample_posterior(r0, times, observations, noise=args.noise, prior=args.prior,
                                  sigma=args.sigma, halfwidth=args.halfwidth, draws=args.draws,
                                  burn=args.burn, seed=args.seed)
    except (ValueError, RuntimeError) as exc:
        parser.error(f'Could not fit the trajectory: {exc}')
    events = [propagate(theta, r0, result['fit']['B'], target_x=target_x) for theta in result['samples']]
    statuses = [event['status'] for event in events]
    predictions = np.array([[event['tg'], event['yg'], event['zg']] for event in events])
    summary, grid = summarize(predictions, statuses)
    summary.update(input=data, noise=args.noise, sigma=args.sigma, halfwidth=args.halfwidth,
                   prior=args.prior, diagnostics=result['diagnostics'], map_theta=result['fit']['theta'])

    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'summary.json', summary)
    np.savez_compressed(args.output / 'samples.npz', parameters=result['samples'],
                        tyz=predictions, statuses=statuses)
    if grid is not None:
        np.savez_compressed(args.output / 'density.npz', **grid)
    print(json.dumps({'output': str(args.output.resolve()), 'p_cross': summary['p_cross'],
                      'diagnostics': result['diagnostics']}, indent=2))


if __name__ == '__main__':
    main()
