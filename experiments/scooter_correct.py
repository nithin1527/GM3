"""Write corrected copies of the scooter CSV logs.

    python -m experiments.scooter_correct                 # every data/<folder>/ -> data/<folder>_corrected/
    python -m experiments.scooter_correct --root data --scale 0.545 --gyro-bias -0.085

For every folder under ``--root`` that holds ``odrive_gps_imu_*.csv`` files, a
sibling ``<folder>_corrected`` is written with one CSV per input file, the same
rows at the same timestamps, and only physically meaningful columns in one
vehicle frame (x forward, y left, z up; angles positive left/counter-clockwise).
Nothing is resampled or smoothed except where noted. The corrections are the
ones ``gm3.dataset.scooter`` established and ``scooter_report --checks``
re-derives:

1. Wheel speed: ``speed_mps = -drive_velocity_mps * scale`` (forward positive;
   the log's sign is negative moving forward and its magnitude is 1.83x too
   high; ``scale`` = 0.545 was fitted against GPS). The same goes for the
   distance since start.
2. Yaw rate: ``yaw_rate_rad_s = -(imu_gyro_z - gyro_bias)``. The IMU is upside
   down, so its z rate is the negative of the vehicle's, and it carries a
   -0.085 deg/s rest bias.
3. IMU axes into the vehicle frame: the IMU's +y points forward and its +x to
   the left, so ``gyro_x = imu_gyro_y``, ``gyro_y = imu_gyro_x``,
   ``gyro_z = -imu_gyro_z`` and likewise for the accelerometer (a proper
   rotation; determinant +1).
4. Steering: ``steer_angle_deg`` is the motor angle. The encoder's straight-ahead
   reading is found per file by regressing yaw rate on speed x angle (its
   intercept), which is why it survives the ~83 deg index shift between runs.
   ``steer_motor_deg`` is the motor angle relative to that zero;
   ``steer_column_deg`` is the column angle, ``-(3.75 / 10.9) * motor``,
   positive left.
5. Body speed: the encoder is on the steered front wheel, so the body's forward
   speed is ``vx_mps = speed_mps * cos(column angle)``. ``vy_mps = lr * yaw_rate``
   is the no-slip sideslip at the centre of gravity (derived, not measured).
6. Pose: ``heading_rad`` integrates the corrected yaw rate from 0 and ``x_m``,
   ``y_m`` integrate the body velocity from the origin (dead reckoning; the
   frame is the run's own, with no absolute heading).
7. Surface angles: ``grade_deg`` (positive climbing) and ``bank_deg`` (positive
   left side uphill) from the accelerometer with the wheel-derived acceleration
   removed, low-passed at 3 s, median-subtracted per file. The fused attitude's
   pitch is not used: it reads every kick as a tilt.
8. GPS: ``gps_new_fix`` marks the rows where a fresh fix arrived (the logger
   repeats the last fix on every row), ``gps_time_s`` puts the fix on the
   ``time_s`` clock, and ``gps_east_m`` / ``gps_north_m`` are metres from the
   file's first fix. Latitude, longitude, altitude and the 1-sigma horizontal
   error are carried through.

``motor_current_A`` (the ODrive Iq) is passed through unchanged; it is zero on
the motor-off runs. Every folder also gets ``corrections.json`` with the
constants used and, per file, the steering zero found and a few summaries, and
a ``README.md`` describing the columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments import _bootstrap  # noqa: F401  (registers the `gm3` package)

from gm3.dataset.scooter import (
    DEFAULT_MIN_SPEED, GYRO_Z_BIAS, MOTOR_SIGN, STEER_RATIO, WHEEL_RADIUS, WHEEL_SPEED_SCALE, _moving_average,
    _steer_zero, steer_column, surface_angles,
)
from gm3.shared import make_scooter_config

COLUMNS = [
    ("timestamp", "wall-clock ISO timestamp from the logger"),
    ("time_s", "seconds since the file started (logger clock)"),
    ("speed_mps", "front-wheel ground speed, forward positive (scale and sign corrected)"),
    ("distance_m", "front-wheel distance since start, m (scale and sign corrected)"),
    ("wheel_omega_rad_s", "front-wheel angular speed, speed_mps / 0.108 m"),
    ("motor_current_A", "ODrive drive Iq, unchanged"),
    ("steer_motor_deg", "steering motor angle relative to straight-ahead"),
    ("steer_column_deg", "column steering angle, positive left = -(3.75/10.9) x motor"),
    ("vx_mps", "body forward speed at the CG = speed_mps x cos(column angle)"),
    ("vy_mps", "body lateral speed at the CG, no-slip estimate lr x yaw_rate (derived)"),
    ("yaw_rate_rad_s", "vehicle yaw rate, positive left, gyro bias removed"),
    ("heading_rad", "integrated yaw from 0 at the first row"),
    ("x_m", "dead-reckoned position, run frame (heading 0 at start)"),
    ("y_m", "dead-reckoned position, run frame"),
    ("gyro_x_rad_s", "roll rate in the vehicle frame (x forward)"),
    ("gyro_y_rad_s", "pitch rate in the vehicle frame (y left)"),
    ("gyro_z_rad_s", "yaw rate in the vehicle frame (z up), bias not removed"),
    ("accel_x_mps2", "specific force, vehicle x (forward)"),
    ("accel_y_mps2", "specific force, vehicle y (left)"),
    ("accel_z_mps2", "specific force, vehicle z (up; ~+9.8 at rest)"),
    ("grade_deg", "surface grade along the vehicle, positive climbing (estimated)"),
    ("bank_deg", "surface bank, positive left side uphill (estimated)"),
    ("moving", "1 when the smoothed speed exceeds 0.3 m/s"),
    ("gps_new_fix", "1 on the row where a new GPS fix arrived"),
    ("gps_time_s", "fix time on the time_s clock (only on new-fix rows)"),
    ("gps_latitude_deg", "unchanged"),
    ("gps_longitude_deg", "unchanged"),
    ("gps_altitude_m", "unchanged"),
    ("gps_east_m", "metres east of the file's first fix"),
    ("gps_north_m", "metres north of the file's first fix"),
    ("gps_sigma_m", "reported 1-sigma horizontal error, sqrt((cov_x + cov_y) / 2)"),
]


def read_raw(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    # A file cut off mid-write leaves a short last row (DictReader fills the
    # missing fields with None); rows without a valid time_s are dropped and
    # counted, everything else missing becomes NaN like any other gap.
    def number(text):
        try:
            return float(text)
        except (TypeError, ValueError):
            return math.nan
    rows = [row for row in rows if math.isfinite(number(row.get("time_s")))]
    if not rows:
        return [], {}
    timestamps = [row["timestamp"] or "" for row in rows]
    columns = {name: np.array([number(row[name]) for row in rows]) for name in rows[0] if name != "timestamp"}
    return timestamps, columns


def correct_file(path: Path, scale: float, gyro_bias: float) -> tuple[list[dict[str, str]], dict[str, float]]:
    timestamps, raw = read_raw(path)
    if not timestamps:
        return [], {}
    t = raw["time_s"]
    dt = np.gradient(t)
    lr = make_scooter_config().lr

    speed = MOTOR_SIGN * scale * raw["drive_velocity_mps"]
    distance = MOTOR_SIGN * scale * raw["drive_distance_since_start_m"]
    yaw_rate = -(raw["imu_gyro_z_rad_s"] - gyro_bias)
    gyro = np.stack([raw["imu_gyro_y_rad_s"], raw["imu_gyro_x_rad_s"], -raw["imu_gyro_z_rad_s"]], axis=1)
    accel = np.stack([raw["imu_accel_y_mps2"], raw["imu_accel_x_mps2"], -raw["imu_accel_z_mps2"]], axis=1)
    moving = _moving_average(speed, 15) > DEFAULT_MIN_SPEED

    theta = np.radians(raw["steer_angle_deg"])
    zero = _steer_zero(speed[moving], theta[moving], yaw_rate[moving]) if moving.sum() > 30 else 0.0
    delta = steer_column(theta, zero)
    vx = speed * np.cos(delta)
    vy = lr * yaw_rate

    def cumtrapz(values: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(t))])

    heading = cumtrapz(yaw_rate)
    x = cumtrapz(vx * np.cos(heading) - vy * np.sin(heading))
    y = cumtrapz(vx * np.sin(heading) + vy * np.cos(heading))
    slopes = surface_angles(t, vx, accel, yaw_rate, moving, source="imu") if moving.sum() > 30 else np.zeros((len(t), 2))

    stamps = raw["gps_ros_stamp_s"]
    has_fix = (raw["gps_received"] == 1) & np.isfinite(raw["gps_latitude_deg"])
    new_fix = has_fix.copy()
    new_fix[1:] &= stamps[1:] != stamps[:-1]
    clock_offset = raw["imu_ros_stamp_s"][0] - t[0]
    gps_time = np.where(new_fix, stamps - clock_offset, math.nan)
    lat, lon = raw["gps_latitude_deg"], raw["gps_longitude_deg"]
    if new_fix.any():
        lat0, lon0 = lat[new_fix][0], lon[new_fix][0]
        east = np.radians(lon - lon0) * 6_371_000.0 * math.cos(math.radians(lat0))
        north = np.radians(lat - lat0) * 6_371_000.0
    else:
        east = north = np.full(len(t), math.nan)
    east = np.where(new_fix, east, math.nan)
    north = np.where(new_fix, north, math.nan)
    sigma = np.where(new_fix, np.sqrt(0.5 * (raw["gps_cov_x_m2"] + raw["gps_cov_y_m2"])), math.nan)

    values = {
        "timestamp": timestamps, "time_s": t, "speed_mps": speed, "distance_m": distance,
        "wheel_omega_rad_s": speed / WHEEL_RADIUS, "motor_current_A": raw["drive_iq_A"],
        "steer_motor_deg": np.degrees(theta - zero), "steer_column_deg": np.degrees(delta),
        "vx_mps": vx, "vy_mps": vy, "yaw_rate_rad_s": yaw_rate, "heading_rad": heading, "x_m": x, "y_m": y,
        "gyro_x_rad_s": gyro[:, 0], "gyro_y_rad_s": gyro[:, 1], "gyro_z_rad_s": gyro[:, 2],
        "accel_x_mps2": accel[:, 0], "accel_y_mps2": accel[:, 1], "accel_z_mps2": accel[:, 2],
        "grade_deg": np.degrees(slopes[:, 0]), "bank_deg": np.degrees(slopes[:, 1]),
        "moving": moving.astype(int), "gps_new_fix": new_fix.astype(int), "gps_time_s": gps_time,
        "gps_latitude_deg": lat, "gps_longitude_deg": lon, "gps_altitude_m": raw["gps_altitude_m"],
        "gps_east_m": east, "gps_north_m": north, "gps_sigma_m": sigma,
    }

    def fmt(v) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return "" if not np.isfinite(v) else f"{float(v):.6f}"

    rows = [{name: fmt(values[name][i]) for name, _ in COLUMNS} for i in range(len(t))]
    with open(path, newline="") as handle:
        raw_rows = sum(1 for _ in handle) - 1
    summary = {
        "rows": int(len(t)), "dropped_rows": int(raw_rows - len(t)),
        "duration_s": float(t[-1] - t[0]), "moving_s": float((moving * dt).sum()),
        "distance_m": float(distance[-1] - distance[0]), "speed_max_mps": float(speed.max()),
        "steer_zero_motor_deg": float(math.degrees(zero)), "gps_fixes": int(new_fix.sum()),
        "motor_current_max_A": float(np.nanmax(np.abs(raw["drive_iq_A"]))),
        "grade_std_deg": float(np.degrees(slopes[moving, 0].std())) if moving.any() else 0.0,
    }
    return rows, summary


def write_readme(folder: Path, scale: float, gyro_bias: float) -> None:
    lines = ["# Corrected scooter logs", "",
             f"Generated by `python -m experiments.scooter_correct` from `{folder.name.removesuffix('_corrected')}/`. "
             "Same rows and timestamps as the originals; every column is in the vehicle frame "
             "(x forward, y left, z up, angles positive left). See the script docstring for each correction.", "",
             f"Constants: wheel-speed scale {scale}, gyro-z bias {math.degrees(gyro_bias):+.3f} deg/s, "
             f"steering ratio {STEER_RATIO:.4f}, wheel radius {WHEEL_RADIUS} m, lr {make_scooter_config().lr} m. "
             "Per-file steering zeros are in `corrections.json`.", "", "| column | meaning |", "|---|---|"]
    lines += [f"| `{name}` | {meaning} |" for name, meaning in COLUMNS]
    (folder / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write corrected copies of the scooter CSV logs.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--scale", type=float, default=WHEEL_SPEED_SCALE)
    parser.add_argument("--gyro-bias", type=float, default=math.degrees(GYRO_Z_BIAS), help="deg/s, IMU frame")
    args = parser.parse_args()
    gyro_bias = math.radians(args.gyro_bias)

    root = Path(args.root)
    folders = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith("_corrected")
                     and any(d.glob("odrive_gps_imu_*.csv")))
    for folder in folders:
        out = root / f"{folder.name}_corrected"
        out.mkdir(exist_ok=True)
        report = {"source": folder.name, "wheel_speed_scale": args.scale, "gyro_z_bias_deg_s": args.gyro_bias,
                  "steer_ratio": STEER_RATIO, "wheel_radius_m": WHEEL_RADIUS, "files": {}}
        print(f"{folder.name} -> {out.name}")
        for path in sorted(folder.glob("odrive_gps_imu_*.csv")):
            rows, summary = correct_file(path, args.scale, gyro_bias)
            if not rows:
                report["files"][path.name] = {"rows": 0, "note": "no complete rows; skipped"}
                print(f"  {path.name}: no complete rows, skipped")
                continue
            with open(out / path.name, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[name for name, _ in COLUMNS])
                writer.writeheader()
                writer.writerows(rows)
            report["files"][path.name] = summary
            dropped = f" ({summary['dropped_rows']} truncated row dropped)" if summary["dropped_rows"] else ""
            print(f"  {path.name}: {summary['rows']} rows{dropped}, {summary['duration_s']:.0f} s, moving {summary['moving_s']:.0f} s, "
                  f"{summary['distance_m']:.0f} m, v max {summary['speed_max_mps']:.2f} m/s, steer zero {summary['steer_zero_motor_deg']:+.1f} deg, "
                  f"{summary['gps_fixes']} fixes, Iq max {summary['motor_current_max_A']:.2f} A, grade std {summary['grade_std_deg']:.2f} deg")
        (out / "corrections.json").write_text(json.dumps(report, indent=2))
        write_readme(out, args.scale, gyro_bias)


if __name__ == "__main__":
    main()
