"""Reproduce the four reference predictions for the contest example."""

import argparse
from concurrent.futures import ProcessPoolExecutor

from run_analysis import ROOT, deterministic, posterior_job, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1, help='number of parallel case workers')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive.')

    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    deterministic(output)
    settings = (
        ('gaussian', 'wide', 4000, 1500),
        ('gaussian', 'strong', 4000, 1500),
        ('rounding', 'wide', 4000, 1500),
        ('rounding', 'strong', 12000, 2500),
    )
    jobs = [(noise, prior, draws, burn, 20261107 + index, str(output))
            for index, (noise, prior, draws, burn) in enumerate(settings)]
    if args.workers == 1:
        summaries = [posterior_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            summaries = list(pool.map(posterior_job, jobs))
    save_json(output / 'posterior_summary.json', summaries)


if __name__ == '__main__':
    main()
