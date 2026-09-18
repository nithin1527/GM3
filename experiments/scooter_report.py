"""Inventory of the Hiboy scooter logs and the checks behind their extraction.

    python -m experiments.scooter_report            # per-run table + trajectory shapes
    python -m experiments.scooter_report --checks   # re-derive every loader assumption

Each run is classified by its turn sequence: contiguous stretches with
|yaw rate| above a threshold are integrated into signed heading changes and
bucketed as sweeps (30-60 deg), corners (60-150), U-turns (150-300) and loops
(300+). Speeds are ground speed while moving (wheel speed after the 0.5 scale
correction the loader applies; see ``gm3.dataset.scooter``).

``--checks`` runs seven validations against the recordings rather than
asserting them:

1. wheel-speed scale: GPS chord over odometer on straight spans;
2. yaw-rate sign: GPS course rotation against the integrated gyro;
3. steering geometry: small-angle kinematic yaw gain -> implied wheelbase for
   the measured gear ratio, per run and pooled, plus the residual zero;
4. effective wheelbase by speed band and by steering-angle band. Flat in
   speed means the tires are kinematic; flat in angle confirms the body speed
   is ``v_front * cos(delta)`` (with the raw wheel speed it climbs from 0.89
   to 1.07 m across the angle bands);
5. steer-to-yaw lag by cross-correlation (what inertia/damping must explain);
6. IMU axis assignment from accelerometer correlations;
7. dead-reckoned path against GPS after rigid alignment, which tests scale
   and sign together;
8. grade: how much of the coasting deceleration heading direction explains
   (an IMU-free planar-ground fit), the accelerometer-derived grade's spread,
   and the coasting drag intercept that sets ``rolling_resistance``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from experiments import _bootstrap  # noqa: F401  (registers the `gm3` package)

from gm3.dataset.scooter import (
    GRAVITY, STEER_RATIO, STEER_SIGN, ScooterRun, _moving_average, kinematic_steer_fit, load_all,
    rigid_align_rms, surface_angles,
)
from gm3.shared import make_scooter_config

OUT = Path(__file__).resolve().parent / "out"


def turn_sequence(run: ScooterRun, rate_threshold: float = 0.15, min_angle: float = 30.0,
                  smooth: int = 15) -> list[float]:
    """Signed heading change (deg, +left) of every turn above ``min_angle``."""
    r = _moving_average(run.states[:, 5], smooth)
    turning = (np.abs(r) > rate_threshold) & run.valid
    angles: list[float] = []
    begin = None
    for i in range(len(turning) + 1):
        inside = i < len(turning) and turning[i]
        if inside and begin is None:
            begin = i
        elif not inside and begin is not None:
            angle = math.degrees(run.states[begin:i, 5].sum() * run.dt)
            if abs(angle) >= min_angle:
                angles.append(angle)
            begin = None
    return angles


def classify(angles: list[float], turning_fraction: float) -> str:
    if not angles:
        return "straight line"
    sizes = [abs(a) for a in angles]
    parts = []
    for label, lo, hi in (("loop", 300, 1e9), ("U-turn", 150, 300), ("corner", 60, 150), ("sweep", 30, 60)):
        n = sum(lo <= s < hi for s in sizes)
        if n:
            parts.append(f"{n} {label}{'s' if n > 1 else ''}")
    signs = np.sign(angles)
    alternating = sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b)
    if len(angles) >= 3 and alternating >= len(angles) - 1 and max(sizes) < 150:
        parts.append("slalom")
    if turning_fraction < 0.2:
        parts.append("mostly straight")
    return ", ".join(parts)


def stops(run: ScooterRun, min_seconds: float = 1.0) -> int:
    invalid = ~run.valid
    edges = np.flatnonzero(np.diff(invalid.astype(int)))
    count = 0
    inside = invalid[0]
    start = 0
    for edge in list(edges) + [len(invalid) - 1]:
        if inside and start > 0 and (edge - start) * run.dt >= min_seconds and edge < len(invalid) - 1:
            count += 1
        inside = not inside
        start = edge + 1
    return count


def run_row(run: ScooterRun, zero_deg: float) -> dict[str, float | str]:
    moving = run.valid
    v = run.states[moving, 3]
    r = run.states[moving, 5]
    column = np.degrees(STEER_RATIO * run.controls[moving, 1])
    angles = turn_sequence(run)
    turning_fraction = float((np.abs(_moving_average(run.states[:, 5], 15)) > 0.15)[moving].mean())
    accel = np.gradient(_moving_average(run.states[:, 3], 15), run.t)[moving]
    net = math.degrees(run.states[moving, 2][-1] - run.states[moving, 2][0])
    return {
        "run": run.name,
        "duration_s": round(run.duration, 1),
        "moving_s": round(run.moving_seconds, 1),
        "distance_m": round(run.distance, 1),
        "gps_path_m": round(run.gps_path_length, 1),
        "v_mean": round(float(v.mean()), 2),
        "v_median": round(float(np.median(v)), 2),
        "v_p95": round(float(np.percentile(v, 95)), 2),
        "v_max": round(float(v.max()), 2),
        "yaw_p95_dps": round(float(np.degrees(np.percentile(np.abs(r), 95))), 0),
        "yaw_max_dps": round(float(np.degrees(np.abs(r).max())), 0),
        "ay_p95": round(float(np.percentile(np.abs(v * r), 95)), 2),
        "ay_max": round(float(np.abs(v * r).max()), 2),
        "steer_p95_deg": round(float(np.percentile(np.abs(column), 95)), 0),
        "steer_max_deg": round(float(np.abs(column).max()), 0),
        "steer_zero_motor_deg": round(zero_deg, 1),
        "coast_decel": round(float(np.percentile(accel, 10)), 2),
        "kick_accel": round(float(np.percentile(accel, 90)), 2),
        "stops": stops(run),
        "turning_frac": round(turning_fraction, 2),
        "net_heading_deg": round(net, 0),
        "turns": " ".join(("L" if a > 0 else "R") + f"{abs(a):.0f}" for a in angles),
        "shape": classify(angles, turning_fraction),
    }


def print_table(rows: list[dict]) -> None:
    columns = [("run", 15), ("duration_s", 8), ("moving_s", 8), ("distance_m", 8), ("gps_path_m", 8),
               ("v_mean", 6), ("v_p95", 6), ("v_max", 6), ("yaw_p95_dps", 8), ("ay_p95", 6), ("ay_max", 6),
               ("steer_p95_deg", 8), ("stops", 5), ("turning_frac", 6), ("shape", 0)]
    print("  " + " ".join(f"{name:>{width}}" if width else name for name, width in columns))
    for row in rows:
        print("  " + " ".join(f"{row[name]:>{width}}" if width else str(row[name]) for name, width in columns))
    print("\n  turn sequences (+L / R, deg):")
    for row in rows:
        print(f"    {row['run']:15s} net {row['net_heading_deg']:+5.0f}  {row['turns']}")


def check_speed_scale(runs: list[ScooterRun], span: float = 5.0) -> None:
    ratios = []
    for run in runs:
        odo = np.interp(run.gps_t, run.t, run.odometer)
        psi = np.interp(run.gps_t, run.t, run.states[:, 2])
        for i in range(len(run.gps_t)):
            j = int(np.searchsorted(run.gps_t, run.gps_t[i] + span))
            if j >= len(run.gps_t):
                break
            travelled = odo[j] - odo[i]
            if travelled > 4.0 and abs(psi[j] - psi[i]) < 0.3:
                ratios.append(float(np.linalg.norm(run.gps_xy[j] - run.gps_xy[i]) / travelled))
    ratios = np.asarray(ratios)
    median = float(np.median(ratios))
    verdict = "PASS" if 0.9 <= median <= 1.15 else "FAIL"
    print(f"  1. wheel-speed scale: GPS chord / scaled odometer over {span:.0f} s straight spans = "
          f"{median:.3f} (IQR {np.percentile(ratios, 25):.3f}-{np.percentile(ratios, 75):.3f}, n={len(ratios)})  {verdict}")
    print("     (chords cut curves and GPS noise inflates them; with the raw wheel speed this reads ~0.52)")


def check_yaw_sign(runs: list[ScooterRun]) -> None:
    k = 3
    xs, ys = [], []
    for run in runs:
        if len(run.gps_t) < 2 * k + 2:
            continue
        chord = run.gps_xy[k:] - run.gps_xy[:-k]
        course = np.arctan2(chord[:, 1], chord[:, 0])
        t_mid = 0.5 * (run.gps_t[k:] + run.gps_t[:-k])
        psi = np.interp(t_mid, run.t, run.states[:, 2])
        long_enough = np.linalg.norm(chord, axis=1)[1:] > 3.0
        xs.append(np.angle(np.exp(1j * np.diff(course)))[long_enough])
        ys.append(np.diff(psi)[long_enough])
    corr = float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])
    print(f"  2. yaw sign: corr(GPS course rotation, integrated r) = {corr:+.2f}  "
          f"{'PASS' if corr > 0.5 else 'FAIL'}   (r = -gyro_z: the IMU is upside down)")


def _pooled(runs: list[ScooterRun]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.concatenate([run.states[run.valid, 3] for run in runs])
    delta = np.concatenate([STEER_SIGN * STEER_RATIO * run.controls[run.valid, 1] for run in runs])
    r = np.concatenate([run.states[run.valid, 5] for run in runs])
    return v, delta, r


def wheelbase_tan(v: np.ndarray, delta: np.ndarray, r: np.ndarray) -> float:
    """Least-squares L in r = v tan(delta) / L."""
    x = v * np.tan(delta)
    return float((x @ x) / (x @ r))


def check_geometry(runs: list[ScooterRun]) -> None:
    config = make_scooter_config()
    pooled = kinematic_steer_fit(runs)
    v, delta, r = _pooled(runs)
    small = np.degrees(np.abs(delta)) < 10.0
    per_run = []
    for run in runs:
        rv, rd, rr = _pooled([run])
        keep = np.degrees(np.abs(rd)) < 10.0
        per_run.append(wheelbase_tan(rv[keep], rd[keep], rr[keep]))
    print(f"  3. steering geometry (ratio {STEER_RATIO:.4f} = 3.75 / 10.9):")
    print(f"     small-angle (< 10 deg column) wheelbase {wheelbase_tan(v[small], delta[small], r[small]):.3f} m pooled, "
          f"{min(per_run):.2f}-{max(per_run):.2f} m per run; preset {config.wheelbase:.2f} m")
    print(f"     linear fit over all angles: {pooled['wheelbase_implied']:.3f} m, R^2 {pooled['r2']:.3f} "
          f"over {pooled['samples']} samples")
    print(f"     residual straight-ahead offset after per-run zeroing: {pooled['steer_zero_deg']:+.2f} deg motor")
    zeros = ", ".join(f"{run.steer_zero_deg:+.0f}" for run in runs)
    print(f"     per-run encoder zero (motor deg): {zeros}")


def check_understeer(runs: list[ScooterRun]) -> None:
    v, delta, r = _pooled(runs)
    angle = np.degrees(np.abs(delta))
    speed_bands = ((0.3, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.5))
    angle_bands = ((3, 10), (10, 20), (20, 30), (30, 45))
    print("  4. effective wheelbase L_eff = v tan(delta) / r, by speed (rows) and column angle (cols):")
    print("     " + " " * 11 + "".join(f"{lo:>3d}-{hi:<2d} deg  " for lo, hi in angle_bands))
    for vlo, vhi in speed_bands:
        cells = []
        for lo, hi in angle_bands:
            cell = (v >= vlo) & (v < vhi) & (angle >= lo) & (angle < hi)
            cells.append(f"{wheelbase_tan(v[cell], delta[cell], r[cell]):8.3f}   " if cell.sum() > 30 else "       -   ")
        print(f"     v {vlo:.1f}-{vhi:.1f} " + "".join(cells))
    print("     (constant down a column = no speed dependence, so no tire slip to identify;\n"
          "      constant along a row = the front-wheel speed projection is right)")


def check_lag(runs: list[ScooterRun]) -> None:
    print("  5. steer -> yaw lag (cross-correlation of v*theta with r):")
    for run in runs:
        a = run.states[run.valid, 3] * run.controls[run.valid, 1]
        b = run.states[run.valid, 5]
        a, b = a - a.mean(), b - b.mean()
        best, best_lag = -1.0, 0
        for lag in range(-6, 7):
            x = a[max(0, -lag): len(a) - max(0, lag)]
            y = b[max(0, lag): len(b) - max(0, -lag)]
            c = float(np.corrcoef(x, y)[0, 1])
            if abs(c) > best:
                best, best_lag = abs(c), lag
        print(f"     {run.name}: best at {best_lag:+d} samples ({best_lag * run.dt * 1000:+.0f} ms), |corr| {abs(best):.3f}")


def check_imu_axes(runs: list[ScooterRun]) -> None:
    print("  6. IMU axes after the vehicle-frame mapping (corr with dv/dt | with v*r):")
    for axis, name in enumerate("xyz"):
        c1, c2 = [], []
        for run in runs:
            dv = np.gradient(_moving_average(run.states[:, 3], 15), run.t)[run.valid]
            ay = (run.states[:, 3] * run.states[:, 5])[run.valid]
            acc = run.accel[run.valid, axis]
            c1.append(np.corrcoef(dv, acc)[0, 1])
            c2.append(np.corrcoef(ay, acc)[0, 1])
        print(f"     accel_{name}: {np.mean(c1):+.2f} | {np.mean(c2):+.2f}")
    print("     (x should be longitudinal; the lateral channel is too noisy to fit on; z reads ~+9.8 at rest)")


def check_dead_reckoning(runs: list[ScooterRun]) -> None:
    print("  7. dead-reckoned path vs GPS after rigid alignment (GPS sigma ~3 m):")
    for run in runs:
        rms = rigid_align_rms(run.states[:, :2], run.t, run)
        sigma = float(np.median(run.gps_sigma))
        print(f"     {run.name}: RMS {rms:.2f} m over {len(run.gps_t)} fixes (median GPS sigma {sigma:.1f} m)")


def check_grade(runs: list[ScooterRun]) -> None:
    print("  8. grade (is there a slope for the slope input to act on?):")
    print(f"     {'run':8s} {'plane fit: amp m/s^2 -> deg, R2':>34s} {'imu grade std deg':>18s} {'coast drag m/s^2':>17s}")
    drags = []
    for run in runs:
        v = run.states[:, 3]
        dv = np.gradient(_moving_average(v, 15), run.t)
        coast = run.valid & (dv < 0.15)
        psi = run.states[:, 2]
        design = np.stack([np.ones(coast.sum()), np.cos(psi[coast]), np.sin(psi[coast])], axis=1)
        coef = np.linalg.lstsq(design, dv[coast], rcond=None)[0]
        amp = float(np.hypot(coef[1], coef[2]))
        r2 = 1.0 - np.var(dv[coast] - design @ coef) / np.var(dv[coast])
        imu = surface_angles(run.t, v, run.accel, run.states[:, 5], run.valid, source="imu")
        drags.append(-coef[0])
        print(f"     {run.name[-6:]:8s} {amp:12.3f} -> {np.degrees(np.arcsin(min(amp / GRAVITY, 1.0))):5.2f} deg, R2 {r2:4.2f}"
              f" {np.degrees(imu[run.valid, 0].std()):18.2f} {-coef[0]:17.3f}")
    print(f"     mean coasting drag {np.mean(drags):.3f} m/s^2 -> rolling_resistance {np.mean(drags) / GRAVITY:.3f} "
          f"(preset {make_scooter_config().rolling_resistance})")
    print("     (plane-fit amplitude under ~1 deg and R2 under 0.2 means the terrain is flat within the noise)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and checks for the Hiboy scooter logs.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--checks", action="store_true")
    args = parser.parse_args()

    runs = load_all(args.root)
    rows = [run_row(run, run.steer_zero_deg) for run in runs]
    print(f"{len(runs)} runs with motion (a stationary 6 s log is skipped)\n")
    print_table(rows)

    v_all = np.concatenate([run.states[run.valid, 3] for run in runs])
    ay_all = np.concatenate([(run.states[:, 3] * run.states[:, 5])[run.valid] for run in runs])
    print(f"\n  totals: {sum(r['duration_s'] for r in rows):.0f} s recorded, "
          f"{sum(r['moving_s'] for r in rows):.0f} s moving, {sum(r['distance_m'] for r in rows):.0f} m")
    print(f"  ground speed while moving: mean {v_all.mean():.2f}, median {np.median(v_all):.2f}, "
          f"p95 {np.percentile(v_all, 95):.2f}, max {v_all.max():.2f} m/s "
          f"({v_all.mean() * 2.237:.1f} mph mean, {v_all.max() * 2.237:.1f} mph max)")
    print(f"  lateral acceleration v*r: p95 {np.percentile(np.abs(ay_all), 95):.2f}, "
          f"max {np.abs(ay_all).max():.2f} m/s^2 ({np.abs(ay_all).max() / 9.81:.2f} g)")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "scooter_runs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {OUT / 'scooter_runs.csv'}")

    if args.checks:
        print("\nchecks:")
        check_speed_scale(runs)
        check_yaw_sign(runs)
        check_geometry(runs)
        check_understeer(runs)
        check_lag(runs)
        check_imu_axes(runs)
        check_dead_reckoning(runs)
        check_grade(runs)


if __name__ == "__main__":
    main()
