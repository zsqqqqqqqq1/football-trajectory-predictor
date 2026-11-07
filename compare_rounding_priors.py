"""Compare wide and strong priors at the contest observation times."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

from model import propagate, sample_posterior
from run_analysis import summarize
from validate import wilson


ROOT = Path(__file__).resolve().parent
CASES = ROOT / 'data/validation_cases.json'
OUTPUT = ROOT / 'results/rounding_prior_comparison_fixed_times'
OBSERVATION_TIMES = (.05, .10)
PRIORS = ('wide', 'strong')


def read_cases():
    data = CASES.read_bytes()
    cases = json.loads(data)['cases']
    if len(cases) != 24 or len({case['shot'] for case in cases}) != len(cases):
        raise ValueError('The validation fixture must contain 24 distinct launches.')
    if any(tuple(case['times']) != OBSERVATION_TIMES for case in cases):
        raise ValueError('Validation observations must be at 0.05 and 0.10 seconds.')
    return cases, hashlib.sha256(data).hexdigest()


def bend_group(case):
    # Measure lateral displacement from the launch direction using synthetic truth only.
    _, y, _ = case['reference']
    x0, y0, _ = case['r0']
    vx, vy, *_ = case['theta']
    bend = abs(y - (y0 + (11. - x0) * vy / vx))
    if bend < .15:
        return 'near_straight'
    return 'moderate_bend' if bend < .40 else 'large_bend'


def evaluate(case, prior, draws):
    obs = np.round(case['clean_observations'], 2)
    shot = case['shot']
    row = dict(shot=shot, prior=prior, group=bend_group(case), draws=draws, covered=False,
               error_m=None, area_m2=None, p_cross=None, max_rhat=None, min_ess=None,
               reliable=False, error='')
    try:
        result = sample_posterior(case['r0'], case['times'], obs, noise='rounding',
                                  prior=prior, draws=draws, burn=1500, seed=13041 + shot)
        events = [propagate(theta, case['r0'], result['fit']['B']) for theta in result['samples']]
        pred = np.array([[e['tg'], e['yg'], e['zg']] for e in events])
        summary, grid = summarize(pred, [e['status'] for e in events])
        diagnostics = result['diagnostics']
        row.update(p_cross=summary['p_cross'], max_rhat=max(diagnostics['split_rhat']),
                   min_ess=min(diagnostics['ess']), reliable=diagnostics['reliable'])
        if grid is not None:
            truth = np.array(case['reference'][1:])
            crossed = pred[np.isfinite(pred).all(axis=1), 1:]
            row.update(error_m=float(np.linalg.norm(crossed.mean(axis=0) - truth)),
                       area_m2=summary['region_95_area'],
                       covered=bool(gaussian_kde(crossed.T)(truth[:, None])[0] >= grid['level']))
    except Exception as exc:
        row.update(reliable=False, error=f'{type(exc).__name__}: {exc}')
    return row


def save_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def summarize_pairs(rows, cases):
    groups = ['all', 'near_straight', 'moderate_bend', 'large_bend']
    summary = []
    for group in groups:
        shots = [c['shot'] for c in cases if group == 'all' or bend_group(c) == group]
        pairs = [(rows.get((shot, 'wide')), rows.get((shot, 'strong'))) for shot in shots]
        valid = [(wide, strong) for wide, strong in pairs
                 if wide and strong and wide['draws'] == strong['draws']
                 and not wide['error'] and not strong['error']]
        reliable = [(wide, strong) for wide, strong in valid if wide['reliable'] and strong['reliable']]
        outcomes = dict(group=group, total_shots=len(shots), paired_completed=len(valid),
                        paired_reliable=len(reliable))
        for index, prior in enumerate(PRIORS):
            selected = [pair[index] for pair in valid]
            covered = sum(r['covered'] for r in selected)
            errors = [r['error_m'] for r in selected if r['error_m'] is not None]
            areas = [r['area_m2'] for r in selected if r['area_m2'] is not None]
            outcomes[prior] = dict(covered=covered, wilson_95=wilson(covered, len(selected)),
                                   mean_error_m=float(np.mean(errors)) if errors else None,
                                   mean_area_m2=float(np.mean(areas)) if areas else None,
                                   with_region=len(areas))
        outcomes['wide_only_covers'] = sum(w['covered'] and not s['covered'] for w, s in valid)
        outcomes['strong_only_covers'] = sum(s['covered'] and not w['covered'] for w, s in valid)
        summary.append(outcomes)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=4000, help='Retained samples per shot and prior')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retry-failed', action='store_true',
                        help='Repeat both priors for incomplete or unreliable pairs only')
    args = parser.parse_args()
    if args.draws < 4 or args.draws % 4 or args.workers < 1:
        parser.error('--draws must be a positive multiple of 4, and --workers must be positive')
    cases, fingerprint = read_cases()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT / 'checkpoint.json'
    saved = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}
    if saved and saved['cases_sha256'] != fingerprint:
        parser.error('Frozen input changed. Move the existing comparison directory before rerunning.')
    rows = {(row['shot'], row['prior']): row for row in saved.get('rows', [])}
    jobs = {}
    for case in cases:
        shot = case['shot']
        previous = [rows.get((shot, prior)) for prior in PRIORS]
        failed = (any(not r or r['error'] or not r['reliable'] for r in previous)
                  or previous[0]['draws'] != previous[1]['draws'])
        if args.retry_failed and not failed:
            continue
        # If sample counts increase, update both sides of the pair.
        draws = max([args.draws] + [r['draws'] for r in previous if r])
        if any(r and r['draws'] != draws for r in previous) or args.retry_failed:
            previous = [None, None]
        for prior, row in zip(PRIORS, previous):
            if row is None or row['error']:
                jobs[shot, prior] = (case, prior, draws)

    def record(row):
        key = row['shot'], row['prior']
        rows[key] = row
        del jobs[key]
        save_json(checkpoint, dict(cases_sha256=fingerprint, rows=list(rows.values())))
        state = row['error'] or ('ok' if row['reliable'] else 'diagnostics warning')
        print(f'{total - len(jobs)}/{total} shot={key[0]} {key[1]} draws={row["draws"]} {state}', flush=True)

    total = len(jobs)
    print(f'Observation times {OBSERVATION_TIMES} s, {len(cases)} paired shots, '
          f'{total} fits remaining. Output: {OUTPUT}', flush=True)
    if jobs and args.workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate, *job) for job in jobs.values()]
                for future in as_completed(futures):
                    record(future.result())
        except BrokenProcessPool as exc:
            print(f'Worker pool stopped: {exc}. Continuing serially.', flush=True)
    for job in list(jobs.values()):
        record(evaluate(*job))

    ordered = [rows[key] for key in sorted(rows)]
    with (OUTPUT / 'per_shot.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=ordered[0].keys())
        writer.writeheader()
        writer.writerows(ordered)
    summary = summarize_pairs(rows, cases)
    save_json(OUTPUT / 'summary.json', dict(cases_sha256=fingerprint,
              observation_times_s=OBSERVATION_TIMES, groups=summary))
    for group in summary:
        print(json.dumps(group), flush=True)


if __name__ == '__main__':
    main()
