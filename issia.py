"""ISSIA-3D adapter: read, audit candidate flights, score two-point hold-out predictions.

Dataset: SoccerNet-v3D ISSIA-3D.csv (Gutierrez-Perez & Agudo, CVPRW 2025),
released from the ISSIA-CNR soccer dataset. 25 fps, coordinates in meters in the
sn-calibration pitch frame (origin at pitch centre, x along the long axis,
y across the pitch, z pointing DOWN). The model uses z up, so z_model = -z_raw.

Scoring protocol for 25 Hz airborne observations:
  - candidate start frame is t = 0; only start + next two frames (0.04/0.08 s)
    are given to the predictor;
  - the virtual target plane is 11 m from the start along the horizontal
    direction of the early observations;
  - the reference crossing is interpolated from the two later hold-out frames
    straddling that plane; y' is lateral, z is absolute height above ground;
  - one independent sample per flight; start-frame +/-1 shift is a sensitivity
    check, not an extra sample.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

from model import Physics, drag_coefficient, fit, propagate
from ekf import ekf_predict
from validate import region_metrics, wilson

ROOT = Path(__file__).resolve().parent
FPS = 25.0
TARGET = 11.0                      # virtual plane distance from start
DESIGNATED_STARTS = (608, 1056, 1144, 2972)   # previous screening: one flight each


def read_issia(path: Path):
    """Return (rows, positions) for all rows with a ball_3D annotation."""
    rows, positions = [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            raw = r.get("ball_3D", "")
            if not raw:
                continue
            try:
                p = np.array(ast.literal_eval(raw), float)
            except (ValueError, SyntaxError):
                continue
            if p.shape != (3,) or not np.isfinite(p).all():
                continue
            errs = [float(r[f"err_cam{j}"]) for j in range(1, 7) if r.get(f"err_cam{j}") not in (None, "")]
            rows.append(dict(frame=int(r["frame"]), pos=np.r_[p[:2], -p[2]],   # z up
                             n_cam=int(r["num_cameras"] or 0), cam_list=r.get("cam_list", ""),
                             err_max=max(errs) if errs else None))
            positions.append(np.r_[p[:2], -p[2]])
    return rows, np.asarray(positions)


def frame_runs(frames):
    """Split frame indices into maximal runs with diff == 1."""
    frames = np.asarray(frames)
    starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
    return [list(range(int(frames[s]), int(frames[e - 1]) + 1)) for s, e in zip(starts, starts[1:].tolist() + [len(frames)])]


def local_basis(p0, p2):
    """Horizontal unit direction from the first two observed frames."""
    d = p2[:2] - p0[:2]
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ValueError("horizontal direction undefined")
    d = d / n
    return d, np.array([-d[1], d[0]])


def audit_candidates(rows):
    """Screen the whole file; return designated-flight candidates, extra accepted
    starts, and a rejection table with reasons (ground contact, bounce, glitches)."""
    by_frame = {r["frame"]: r for r in rows}
    runs = frame_runs([r["frame"] for r in rows])
    candidates, rejected, all_starts = [], [], []
    for run in runs:
        idx = [by_frame[f] for f in run if f in by_frame]
        if len(idx) < 5:
            continue
        pts = np.array([r["pos"] for r in idx])
        run_starts, run_rejections = [], []
        for i in range(len(idx) - 4):
            p0, p2 = pts[i], pts[i + 2]
            try:
                d, lat = local_basis(p0, p2)
            except ValueError:
                continue
            if p0[2] < 0.2 or np.linalg.norm(p2[:2] - p0[:2]) < 0.8:   # ~10 m/s minimum
                continue
            xp = (pts[i:, :2] - p0[:2]) @ d
            yp = (pts[i:, :2] - p0[:2]) @ lat
            z = pts[i:, 2]
            hits = np.flatnonzero((xp[:-1] < TARGET) & (xp[1:] >= TARGET))
            if not len(hits):
                continue                       # data ends before the plane
            j = int(hits[0])
            if j < 3:                          # crossing inside the observed frames
                continue
            span = idx[i:i + j + 2]
            frames = np.array([r["frame"] for r in span])
            z_span = pts[i:i + j + 2, 2]
            vel = np.diff(pts[i:i + j + 2], axis=0) / (np.diff(frames) / FPS)[:, None]
            dv = np.linalg.norm(np.diff(vel, axis=0), axis=1)
            alpha = (TARGET - xp[j]) / (xp[j + 1] - xp[j])
            ref = dict(t_s=(frames[j] - idx[i]["frame"] + alpha * (frames[j + 1] - frames[j])) / FPS,
                       y_m=float((1 - alpha) * yp[j] + alpha * yp[j + 1]),
                       z_m=float((1 - alpha) * z[j] + alpha * z[j + 1]))
            reasons = []
            if np.any(z_span < 0.15):
                reasons.append("ground_contact_or_glitch_z_below_0.15_m")
            if ref["z_m"] < 0.15:
                reasons.append("reference_below_0.15_m")
            # genuine bounce: a local z minimum below 0.6 m followed by a rebound > 0.4 m
            zmins = 1 + np.flatnonzero((z_span[1:-1] < z_span[:-2]) & (z_span[1:-1] <= z_span[2:]))
            if any(z_span[k] < 0.6 and np.max(z_span[k:]) - z_span[k] > 0.4 for k in zmins):
                reasons.append("bounce_like_z_rebound")
            if np.any(np.diff(frames) != 1):
                reasons.append("missing_frames")
            if reasons:
                run_rejections.append(dict(run_id=f"frames_{run[0]}-{run[-1]}", start_frame=idx[i]["frame"],
                                           crossing_frames=[idx[i + j]["frame"], idx[i + j + 1]["frame"]],
                                           min_z_m=float(np.min(z_span)), reference=ref, reasons=reasons))
                continue
            flags = []
            if np.max(dv) > 3.0:
                flags.append(f"speed_jump_max_{np.max(dv):.1f}_m_s_per_step")
            if min(r["err_max"] or 1e9 for r in span[:3]) > 5.0:
                flags.append("early_reproj_err_high")
            if max(r["err_max"] or 0 for r in span[-2:]) > 5.0:
                flags.append("reference_reproj_err_high")
            if any(r["n_cam"] < 2 for r in span):
                flags.append("single_camera_frame")
            record = dict(
                run_id=f"frames_{run[0]}-{run[-1]}", start_frame=idx[i]["frame"],
                obs_frames=[idx[i + 1]["frame"], idx[i + 2]["frame"]],
                crossing_frames=[idx[i + j]["frame"], idx[i + j + 1]["frame"]],
                window_frames=f"{idx[i]['frame']}-{idx[i + j + 1]['frame']}",
                n_holdout_frames=int(len(idx) - i - 1),
                start_z_m=float(z[0]), min_z_m=float(np.min(z_span)),
                median_speed_m_s=float(np.median(np.linalg.norm(vel, axis=1))),
                max_step_dv_m_s=float(np.max(dv)),
                n_cam_min=int(min(r["n_cam"] for r in span)),
                n_cam_max=int(max(r["n_cam"] for r in span)),
                cam_list=idx[i]["cam_list"],
                err_max_early_px=float(max(r["err_max"] or 0 for r in span[:3])),
                err_max_reference_px=float(max(r["err_max"] or 0 for r in span[-2:])),
                direction_horizontal=d.tolist(), reference=ref,
                flags=flags, status="accepted_with_flags" if flags else "accepted_for_scoring",
                target_plane="virtual 11 m, perpendicular to initial horizontal direction")
            if idx[i]["frame"] in DESIGNATED_STARTS:
                candidates.append(record)
            run_starts.append(record)
        if run_starts:
            all_starts.append(run_starts)
        # rejection table: one row per run that has qualifying starts but none accepted
        if run_rejections and not run_starts:
            first = run_rejections[0]
            rejected.append(dict(run_id=f"frames_{run[0]}-{run[-1]}",
                                 first_qualifying_start=first["start_frame"],
                                 crossing_frames="/".join(map(str, first["crossing_frames"])),
                                 min_z_m=min(r["min_z_m"] for r in run_rejections),
                                 reference_z_m=min(r["reference"]["z_m"] for r in run_rejections),
                                 reasons=";".join(sorted({x for r in run_rejections for x in r["reasons"]}))))
        elif run_rejections:
            # accepted run with some late-window starts failing the contact rules
            rejected.append(dict(run_id=f"frames_{run[0]}-{run[-1]}",
                                 first_qualifying_start=run_rejections[0]["start_frame"],
                                 crossing_frames="/".join(map(str, run_rejections[0]["crossing_frames"])),
                                 min_z_m=min(r["min_z_m"] for r in run_rejections),
                                 reference_z_m=min(r["reference"]["z_m"] for r in run_rejections),
                                 reasons=";".join(sorted({x for r in run_rejections for x in r["reasons"]})),
                                 note="accepted_starts_exist_in_this_run"))
    candidates.sort(key=lambda c: c["start_frame"])
    return candidates, rejected, all_starts


def build_case(by_frame, start, target=TARGET):
    """Build predictor input and hold-out reference for one start frame."""
    rows = [by_frame.get(start + k) for k in range(3)]
    if any(r is None for r in rows):
        raise ValueError(f"missing observation frames around {start}")
    p = np.array([r["pos"] for r in rows])
    d, lat = local_basis(p[0], p[2])
    r0 = np.array([0.0, 0.0, p[0, 2]])
    obs = np.column_stack(((p[1:, :2] - p[0, :2]) @ d, (p[1:, :2] - p[0, :2]) @ lat, p[1:, 2]))
    # hold-out frames after the observed three
    f0 = rows[0]["frame"]
    frames = [f for f in sorted(by_frame) if f > f0 + 2]
    pts = np.array([by_frame[f]["pos"] for f in frames])
    xp = (pts[:, :2] - p[0, :2]) @ d
    hits = np.flatnonzero((xp[:-1] < target) & (xp[1:] >= target))
    if not len(hits):
        return None
    j = int(hits[0])
    alpha = (target - xp[j]) / (xp[j + 1] - xp[j])
    yp = (pts[:, :2] - p[0, :2]) @ lat
    ref = np.array([(frames[j] - f0 + alpha * (frames[j + 1] - frames[j])) / FPS,
                    (1 - alpha) * yp[j] + alpha * yp[j + 1],
                    (1 - alpha) * pts[j, 2] + alpha * pts[j + 1, 2]])
    return dict(r0=r0, times=np.array([0.04, 0.08]), obs=obs, target_x=target,
                reference=ref, crossing_frames=[frames[j], frames[j + 1]],
                basis=d.tolist())


def score_method(method, case, sigma, seed):
    """Return per-method errors; EKF additionally returns region metrics."""
    r0, times, obs = case["r0"], case["times"], case["obs"]
    target, ref = case["target_x"], case["reference"]
    if method == "gravity":
        f = fit(r0, times, obs, model="gravity", regularize=False)
        e = propagate(f["theta"], r0, fixed_cd=f["fixed_cd"], target_x=target)
        point = np.array([e["tg"], e["yg"], e["zg"]])
        return dict(status=e["status"], point=point, cd=0.0, coverage=None, area=None)
    if method == "batch":
        f = fit(r0, times, obs, noise="gaussian", sigma=sigma)
        e = propagate(f["theta"], r0, fixed_cd=f["fixed_cd"], target_x=target)
        return dict(status=e["status"], point=np.array([e["tg"], e["yg"], e["zg"]]),
                    cd=drag_coefficient(np.linalg.norm(f["theta"][:3])), coverage=None, area=None)
    if method == "ekf":
        res = ekf_predict(r0, times, obs, noise_sd=sigma, draws=4000, seed=seed,
                          target_x=target)
        metrics = region_metrics(res["predictions"], ref)
        crossed = np.all(np.isfinite(res["predictions"]), axis=1)
        point = res["predictions"][crossed].mean(axis=0) if crossed.any() else np.full(3, np.nan)
        return dict(status="crossed" if crossed.any() else "none_crossed", point=point,
                    cd=None, coverage=metrics["covered"], area=metrics["area_m2"])


def score_all(by_frame, output, sigma_cm=3.0):
    cases = []
    for start in DESIGNATED_STARTS:
        for shift in (-1, 0, 1):
            s = start + shift
            if s not in by_frame:
                continue
            try:
                case = build_case(by_frame, s)
            except ValueError:
                continue
            if case is None:
                continue
            cases.append(dict(start_frame=s, designated=start, shift=shift, case=case))
    rows = []
    for c in cases:
        case = c["case"]
        for method, sigma in (("gravity", sigma_cm / 100), ("batch", sigma_cm / 100),
                              ("batch", 0.01), ("ekf", sigma_cm / 100)):
            label = method if method != "batch" else f"batch_{sigma * 100:g}cm"
            try:
                r = score_method(method, case, sigma, seed=20261107 + c["start_frame"])
            except Exception as exc:
                rows.append(dict(start_frame=c["start_frame"], designated=c["designated"],
                                 shift=c["shift"], method=label, sigma_cm=sigma * 100,
                                 exception=f"{type(exc).__name__}: {exc}"))
                continue
            err = r["point"] - case["reference"]
            rows.append(dict(start_frame=c["start_frame"], designated=c["designated"],
                             shift=c["shift"], method=label, sigma_cm=sigma * 100,
                             status=r["status"], cd=r["cd"],
                             dt_err_s=float(err[0]), dy_err_m=float(err[1]), dz_err_m=float(err[2]),
                             err2d_m=float(np.linalg.norm(err[1:])),
                             coverage_95=r["coverage"], region_area_m2=r["area"],
                             ref_t_s=float(case["reference"][0]), ref_y_m=float(case["reference"][1]),
                             ref_z_m=float(case["reference"][2]), exception=""))
    summary = dict(sigma_cm=sigma_cm, target_m=TARGET, fps=FPS,
                   protocol="start + 0.04/0.08 s inputs; hold-out reference interpolated at the virtual plane",
                   note="shift -1/+1 are sensitivity re-runs of the same flight, not independent samples")
    per_case = {}
    for r in rows:
        if r["shift"] != 0 or r["exception"]:
            continue
        key = r["start_frame"]
        per_case.setdefault(key, dict(start_frame=key, reference=dict(
            t_s=r["ref_t_s"], y_m=r["ref_y_m"], z_m=r["ref_z_m"]), methods={}))
        per_case[key]["methods"][r["method"]] = {k: r[k] for k in
            ("status", "cd", "dt_err_s", "dy_err_m", "dz_err_m", "err2d_m", "coverage_95", "region_area_m2")}
    summary["per_case"] = per_case
    summary["sensitivity"] = [r for r in rows if r["shift"] != 0]
    with (output / "issia_per_case.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (output / "issia_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="ISSIA-3D.csv from the SoccerNet-v3D release")
    p.add_argument("--output", type=Path, default=ROOT / "results/issia/diagnostics")
    p.add_argument("--results", type=Path, default=ROOT / "results/issia")
    p.add_argument("--sigma-cm", type=float, default=3.0)
    a = p.parse_args()
    rows, positions = read_issia(a.csv)
    candidates, rejected, all_starts = audit_candidates(rows)
    by_frame = {r["frame"]: r for r in rows}
    a.output.mkdir(parents=True, exist_ok=True)
    a.results.mkdir(parents=True, exist_ok=True)
    with (a.output / "issia_candidate_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["run_id", "start_frame", "obs_frames", "crossing_frames", "window_frames",
                  "n_holdout_frames", "start_z_m", "min_z_m", "median_speed_m_s", "max_step_dv_m_s",
                  "n_cam_min", "n_cam_max", "cam_list", "err_max_early_px", "err_max_reference_px",
                  "reference_t_s", "reference_y_m", "reference_z_m", "flags", "status"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in candidates:
            row = dict(c)
            row["reference_t_s"] = c["reference"]["t_s"]
            row["reference_y_m"] = c["reference"]["y_m"]
            row["reference_z_m"] = c["reference"]["z_m"]
            row["flags"] = ";".join(c["flags"])
            row["obs_frames"] = "/".join(map(str, c["obs_frames"]))
            row["crossing_frames"] = "/".join(map(str, c["crossing_frames"]))
            w.writerow(row)
    with (a.output / "issia_rejected_runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["run_id", "first_qualifying_start", "crossing_frames", "min_z_m", "reference_z_m", "reasons", "note"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rejected)
    audit = dict(dataset="ISSIA-3D (SoccerNet-v3D release)", rows_with_ball=len(rows),
                 total_rows=2465, fps=FPS, coordinate="z_model = -z_raw; x,y sn-calibration pitch frame, origin at centre",
                 screening_rules=dict(consecutive_frames=True, start_airborne_z_m=0.2,
                                      min_horizontal_displacement_2frames_m=0.8,
                                      crossing_after_observed_frames=True,
                                      rejection_rules=["z < 0.15 m at any frame before the plane",
                                                       "interpolated reference z < 0.15 m",
                                                       "bounce-like z rebound (dip below 0.6 m then rise > 0.4 m)"],
                                      one_start_per_flight=True,
                                      designated_starts=list(DESIGNATED_STARTS)),
                 flags_definition=dict(speed_jump="max |dv| between consecutive 0.04 s steps > 3 m/s",
                                       reproj_err_high="max per-frame reprojection error > 5 px",
                                       single_camera_frame="any frame triangulated with < 2 cameras"),
                 extra_accepted_starts=all_starts,
                 candidates=candidates, rejected_runs=rejected)
    (a.output / "issia_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary, rows = score_all(by_frame, a.results, sigma_cm=a.sigma_cm)
    print(json.dumps(dict(candidates=[c["start_frame"] for c in candidates],
                          rejected_count=len(rejected),
                          scored_cases=list(summary["per_case"]),
                          per_case={k: v["methods"] for k, v in summary["per_case"].items()},
                          ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
