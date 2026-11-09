"""Reproduce the DFL J03WOH airborne holdout at the measured goal line.

Streams the positions XML once, keeps BALL frames near shot events, finds
active-run windows that cross a goal line (x = +/-52.5 m, centre-based pitch
frame, z up), and scores the clean in-air windows with the two-point predictor
(gravity / batch full model / EKF). The target plane is the REAL goal line
along the known pitch axis: local x' = s*(x - x0) with s the flight sign,
y' = s*(y - y0), z unchanged. Inputs are only start + 0.04/0.08 s; the crossing
reference is interpolated from the two straddling hold-out frames.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from model import drag_coefficient, fit, propagate
from ekf import ekf_predict
from validate import region_metrics

ROOT = Path(__file__).resolve().parent
GOAL_X = 52.5
FPS = 25.0


def shot_events(events_path):
    root = ET.parse(events_path).getroot()
    out = []
    for e in root.iter("Event"):
        kind = e[0].tag if len(e) else ""
        if kind not in ("ShotAtGoal", "Penalty"):
            continue
        out.append(dict(event_id=e.get("EventId"), kind=kind,
                        time=datetime.fromisoformat(e.get("EventTime")),
                        x=float(e.get("X-Position")), y=float(e.get("Y-Position")),
                        team=e[0].get("Team")))
    return out


def extract_ball_window(positions_path, events, pad_s=8.0):
    """Single pass; keep BALL frames within +/-pad of any event time."""
    windows = [(ev["time"] - timedelta(seconds=pad_s), ev["time"] + timedelta(seconds=pad_s)) for ev in events]
    rows = []
    frameset = {}
    for event, element in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            frameset = dict(element.attrib)
        elif event == "end" and element.tag == "Frame":
            if frameset.get("TeamId", "").upper() == "BALL":
                t = datetime.fromisoformat(element.attrib["T"])
                if any(lo <= t <= hi for lo, hi in windows):
                    rows.append({**element.attrib, "GameSection": frameset.get("GameSection", "")})
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            element.clear()
    return rows


def runs_of(frames):
    frames = sorted(frames)
    out, cur = [], []
    for f in frames:
        if cur and f - cur[-1] != 1:
            out.append(cur)
            cur = []
        cur.append(f)
    if cur:
        out.append(cur)
    return out


def candidate_runs(rows):
    active = [r for r in rows if r.get("BallStatus") == "1"]
    by_n = {int(r["N"]): r for r in active}
    candidates = []
    for run in runs_of(sorted(by_n)):
        if len(run) < 5:
            continue
        idx = [by_n[n] for n in run]
        x = np.array([float(r["X"]) for r in idx])
        y = np.array([float(r["Y"]) for r in idx])
        z = np.array([float(r["Z"]) for r in idx])
        for sign, line in ((+1.0, GOAL_X), (-1.0, -GOAL_X)):
            for i in range(len(run) - 4):
                start = run[i]
                d0 = sign * (line - x[i])          # positive on the field side of the goal
                if not (4.0 <= d0 <= 25.0) or z[i] < 0.3:
                    continue
                tgt = sign * line
                hits = np.flatnonzero((sign * x[i:-1] < tgt) & (sign * x[i + 1:] >= tgt))
                if not len(hits):
                    continue
                j = int(hits[0])
                if j < 3:                          # crossing after the observed frames
                    continue
                span_x, span_y, span_z = x[i:i + j + 2], y[i:i + j + 2], z[i:i + j + 2]
                frames = run[i:i + j + 2]
                vel = np.diff(np.c_[span_x, span_y, span_z], axis=0) / (np.diff(frames) / FPS)[:, None]
                dv = np.linalg.norm(np.diff(vel, axis=0), axis=1)
                alpha = (tgt - sign * span_x[j]) / (sign * (span_x[j + 1] - span_x[j]))
                ref = dict(t_s=(run[i + j] - start + alpha) / FPS,
                           y_m=float((1 - alpha) * sign * span_y[j] + alpha * sign * span_y[j + 1] - sign * y[i]),
                           z_m=float((1 - alpha) * span_z[j] + alpha * span_z[j + 1]))
                flags = []
                if np.min(span_z) < 0.15:
                    flags.append("ground_contact_before_line")
                if ref["z_m"] < 0.15:
                    flags.append("reference_below_0.15_m")
                if np.max(np.linalg.norm(vel, axis=1)) > 40 or np.median(np.linalg.norm(vel, axis=1)) < 8:
                    flags.append("implausible_speed")
                if np.max(dv) > 8:
                    flags.append(f"speed_jump_max_{np.max(dv):.1f}")
                candidates.append(dict(run_start=int(run[0]), run_end=int(run[-1]), start_frame=start,
                                       n_frames=len(run), sign=int(sign), distance_m=float(d0),
                                       crossing_frames=[run[i + j], run[i + j + 1]], reference=ref,
                                       start_z_m=float(z[i]), min_z_m=float(np.min(span_z)),
                                       median_speed_m_s=float(np.median(np.linalg.norm(vel, axis=1))),
                                       max_step_dv_m_s=float(np.max(dv)),
                                       first_timestamp=idx[0]["T"], last_timestamp=idx[-1]["T"],
                                       flags=flags,
                                       status="accepted_for_scoring" if not flags else "accepted_with_flags"))
    candidates.sort(key=lambda c: c["start_frame"])
    return candidates


def score_case(by_n, start, line_sign, sigma_cm):
    rows = [by_n.get(start + k) for k in range(3)]
    if any(r is None for r in rows):
        raise ValueError("missing observation frames")
    s = line_sign
    x = np.array([float(r["X"]) for r in rows])
    y = np.array([float(r["Y"]) for r in rows])
    z = np.array([float(r["Z"]) for r in rows])
    x0, y0, z0 = s * x[0], s * y[0], z[0]
    r0 = np.array([0.0, 0.0, z0])
    obs = np.column_stack((s * x[1:] - x0, s * y[1:] - y0, z[1:]))
    target = GOAL_X - s * x[0]               # local x' of the real goal line
    frames = [f for f in sorted(by_n) if f > start + 2]
    fx = s * np.array([float(by_n[f]["X"]) for f in frames]) - s * x[0]
    fy = s * np.array([float(by_n[f]["Y"]) for f in frames]) - y0
    fz = np.array([float(by_n[f]["Z"]) for f in frames])
    hits = np.flatnonzero((fx[:-1] < target) & (fx[1:] >= target))
    if not len(hits):
        return None
    j = int(hits[0])
    alpha = (target - fx[j]) / (fx[j + 1] - fx[j])
    ref = np.array([(frames[j] - start + alpha) / FPS,
                    (1 - alpha) * fy[j] + alpha * fy[j + 1],
                    (1 - alpha) * fz[j] + alpha * fz[j + 1]])
    case = dict(r0=r0, times=np.array([0.04, 0.08]), obs=obs, target_x=float(target),
                reference=ref, crossing_frames=[frames[j], frames[j + 1]])
    out = []
    for method, sigma in (("gravity", sigma_cm / 100), ("batch", sigma_cm / 100), ("batch", 0.01),
                          ("ekf", sigma_cm / 100)):
        label = method if method != "batch" else f"batch_{sigma * 100:g}cm"
        try:
            if method == "gravity":
                f = fit(r0, case["times"], obs, model="gravity", regularize=False)
                e = propagate(f["theta"], r0, fixed_cd=f["fixed_cd"], target_x=case["target_x"])
                point, cd = np.array([e["tg"], e["yg"], e["zg"]]), 0.0
                cov, area, status = None, None, e["status"]
            elif method == "batch":
                f = fit(r0, case["times"], obs, noise="gaussian", sigma=sigma)
                e = propagate(f["theta"], r0, fixed_cd=f["fixed_cd"], target_x=case["target_x"])
                point, cd = np.array([e["tg"], e["yg"], e["zg"]]), drag_coefficient(np.linalg.norm(f["theta"][:3]))
                cov, area, status = None, None, e["status"]
            else:
                res = ekf_predict(r0, case["times"], obs, noise_sd=sigma,
                                  draws=4000, seed=31072026 + start, target_x=case["target_x"])
                m = region_metrics(res["predictions"], ref)
                crossed = np.all(np.isfinite(res["predictions"]), axis=1)
                point = res["predictions"][crossed].mean(axis=0) if crossed.any() else np.full(3, np.nan)
                cd, cov, area, status = None, m["covered"], m["area_m2"], \
                    "crossed" if crossed.any() else "none_crossed"
            if status != "crossed":
                out.append(dict(start_frame=start, method=label, sigma_cm=sigma * 100,
                                status=status, cd=cd, exception=""))
                continue
            err = point - ref
            out.append(dict(start_frame=start, method=label, sigma_cm=sigma * 100,
                            status=status, cd=cd, dt_err_s=float(err[0]),
                            dy_err_m=float(err[1]), dz_err_m=float(err[2]),
                            err2d_m=float(np.linalg.norm(err[1:])),
                            coverage_95=cov, region_area_m2=area,
                            ref_t_s=float(ref[0]), ref_y_m=float(ref[1]), ref_z_m=float(ref[2]),
                            exception=""))
        except Exception as exc:
            out.append(dict(start_frame=start, method=label, sigma_cm=sigma * 100,
                            status="failed", exception=f"{type(exc).__name__}: {exc}"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True,
                   help="directory containing the J03WOH event and position XML files")
    p.add_argument("--output", type=Path, default=ROOT / "results/dfl/diagnostics")
    p.add_argument("--results", type=Path, default=ROOT / "results/dfl")
    p.add_argument("--sigma-cm", type=float, default=3.0)
    p.add_argument("--score-starts", type=str, default="",
                   help="comma-separated start frames; by default select one per flight plus sensitivity checks")
    a = p.parse_args()
    events_path = a.root / "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    positions_path = a.root / "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    events = shot_events(events_path)
    print(f"shot events: {len(events)}", flush=True)
    rows = extract_ball_window(positions_path, events)
    print(f"ball frames in windows: {len(rows)}", flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    a.results.mkdir(parents=True, exist_ok=True)
    with (a.output / "dfl_J03WOH_ball_window.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    candidates = candidate_runs(rows)
    with (a.output / "dfl_J03WOH_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["run_start", "run_end", "n_frames", "start_frame", "sign", "distance_m", "crossing_frames",
                  "start_z_m", "min_z_m", "median_speed_m_s", "max_step_dv_m_s", "reference_t_s",
                  "reference_y_m", "reference_z_m", "first_timestamp", "last_timestamp", "flags", "status"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in candidates:
            row = dict(c)
            row["reference_t_s"] = c["reference"]["t_s"]
            row["reference_y_m"] = c["reference"]["y_m"]
            row["reference_z_m"] = c["reference"]["z_m"]
            row["crossing_frames"] = "/".join(map(str, c["crossing_frames"]))
            row["flags"] = ";".join(c["flags"])
            w.writerow(row)
    by_n = {int(r["N"]): r for r in rows if r.get("BallStatus") == "1"}
    sign_by_start = {c["start_frame"]: c["sign"] for c in candidates}
    if a.score_starts:
        to_score = [int(x) for x in a.score_starts.split(",") if x.strip()]
        shift_of = {s: 0 for s in to_score}
    else:
        # one sample per flight: cleanest start with the largest distance <= 11.5 m
        selected = {}
        for c in candidates:
            if c["flags"] or c["distance_m"] > 11.5:
                continue
            key = (c["run_start"], c["sign"])
            if key not in selected or c["distance_m"] > selected[key]["distance_m"]:
                selected[key] = c
        to_score = sorted(c["start_frame"] for c in selected.values())
        shift_of = {s: 0 for s in to_score}
        # start-frame sensitivity for the penalty flight (designated start 128956)
        for frame, shift in ((128955, -1), (128957, 1)):
            if all(n in by_n for n in (frame, frame + 1, frame + 2)):
                to_score.append(frame)
                shift_of[frame] = shift
    all_rows = []
    for start in to_score:
        try:
            rows_out = score_case(by_n, start, float(sign_by_start.get(start, 1.0)), a.sigma_cm)
        except ValueError as exc:
            all_rows.append(dict(start_frame=start, method="", sigma_cm="", status="skipped",
                                 shift=shift_of.get(start, 0), exception=str(exc)))
            continue
        if rows_out is None:
            all_rows.append(dict(start_frame=start, method="", sigma_cm="", status="no_crossing",
                                 shift=shift_of.get(start, 0), exception=""))
            continue
        for r in rows_out:
            r["shift"] = shift_of.get(start, 0)
        all_rows.extend(rows_out)
    with (a.results / "dfl_j03woh_per_case.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = sorted({k for r in all_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    summary = dict(match="DFL-MAT-J03WOH", evidence_group="air_window_real_goal_line",
                   goal_line_x_m=GOAL_X, fps=FPS,
                   protocol="start + 0.04/0.08 s inputs; reference interpolated at the real goal line; known pitch axis direction",
                   shot_event_count=len(events),
                   candidates=[{**c, "event": next((dict(kind=e["kind"], team=e["team"],
                                                       event_id=e["event_id"]) for e in events
                                                    if abs((datetime.fromisoformat(c["first_timestamp"])
                                                            - e["time"]).total_seconds()) < 8), None)}
                               for c in candidates], scoring=all_rows)
    (a.results / "dfl_j03woh_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({c["run_start"]: c for c in candidates}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
