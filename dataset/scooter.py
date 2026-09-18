"""Hiboy scooter CSV logs (ODrive + GPS + IMU) as GM3 rollout tensors.

The platform is a Hiboy S2 (350 W front hub motor, 8.5 in tires) fitted with a
steering motor on the column and two caster wheels just ahead of the rear
wheel, so it cannot lean. The runs here were ridden with the hub motor off
(``drive_iq_A`` is identically zero), so the front wheel free-rolls and its
encoder is a ground-speed sensor, not a control input in the physical sense.
With ``motor_on=True`` GM3 takes it as ``omega`` and the tire model has to
reproduce the measured speed; by default both wheels free-roll.

Conventions applied here, each checked against the recordings
(``python -m experiments.scooter_report --checks`` re-derives them):

* **Wheel speed is logged 1.83x too high.** GPS chord length over odometer
  distance on straight spans reads 0.50-0.53 with the raw values, and a factor
  of two (an encoder CPR / pole-pair misconfiguration) was the first guess.
  Letting the dead-reckoned path find the scale that best matches the GPS
  fixes over each whole run gives 0.545 on every run (0.535-0.555), halving
  the path error against GPS from 0.76 to 0.42 m; with it the kinematic yaw
  gain implies a 0.95 m wheelbase. ``WHEEL_SPEED_SCALE`` carries the
  correction; ``experiments.scooter_sensors`` re-derives it.
* **The yaw gyro has a -0.085 deg/s bias** (stationary log; the GPS fit wants
  -0.10). Rigid alignment cannot remove a bias, because it integrates into a
  heading ramp, and it is worth 0.1 m of path error over a run.
  ``GYRO_Z_BIAS`` removes it.
* Forward motion is a negative drive velocity (``MOTOR_SIGN``).
* The IMU is mounted upside down (roll ~ 175 deg, ``accel_z`` ~ -9.7), so the
  vehicle yaw rate is ``-gyro_z`` (``YAW_SIGN``). GPS course rotation against
  the integrated gyro confirms the sign at corr -0.9. Its ``y`` axis points
  along the scooter (corr +0.5 with dv/dt) and its lateral axis is too noisy
  to use, so the accelerometer is carried for diagnostics only.
* ``steer_angle_deg`` is the steering *motor* angle. The motor pinion (3.75 cm)
  drives the column gear (10.9 cm), so the column turns ``STEER_RATIO`` times
  as far. A positive motor angle turns the scooter right, and GM3's ``delta``
  is positive left (``STEER_SIGN``).
* **The front wheel measures speed along the steered direction.** The hub
  motor encoder is on the steered wheel, so ``drive_velocity`` is the speed of
  that wheel's contact point, not the body's. With a non-slipping rear wheel
  the body speed is ``v_front * cos(delta)``, 23% less at 40 deg of column
  angle. Treating the wheel speed as body speed makes the effective wheelbase
  appear to grow with steering angle (0.89 m below 10 deg to 1.07 m at
  30-45 deg) and forces the driven front tire into 30% longitudinal slip in
  tight turns, which is where the model's gradients blew up. ``omega`` stays
  ``v_front / R`` because GM3 compares it against the tire-frame speed.
* **The steering encoder zero is not straight ahead, and it moved.** Regressing
  yaw rate on ``v * theta`` puts straight-ahead at about -12 deg (motor) for
  the first seven runs and about +71 deg for the last two, i.e. the index
  shifted ~83 deg between 16:34 and 16:35. Each run's zero is therefore
  estimated from its own kinematic regression at load time (the estimate does
  not depend on the gear ratio) and ``controls[:, 1]`` is the motor angle
  *relative to that zero*. A fit can still carry a residual global offset.
* Heading is the integrated gyro and position is wheel-plus-gyro dead
  reckoning: the GPS fix is ~1 Hz with a 3 m sigma, far too coarse for the
  1-3 s windows the fits use. GPS is kept alongside for run-level checks.
* Lateral velocity is not measured. ``vy`` is seeded with the no-slip value
  ``lr * r`` (the rear wheel rolls straight, so the CG's lateral velocity is
  its distance from the rear axle times the yaw rate); a window that started
  from ``vy = 0`` mid-corner would hand the model a spurious sideslip
  transient to explain.
* ``gamma``/``gamma_dot`` stay zero (casters). Surface angles come from the
  accelerometer with the wheel-derived acceleration removed
  (:func:`surface_angles`); the fused attitude's pitch is unusable because it
  reads every kick as a tilt.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gm3.shared.presets import make_scooter_config

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "scooter_data_kim_quad"

STEER_MOTOR_DIAMETER = 0.0375
STEER_GEAR_DIAMETER = 0.109
STEER_RATIO = STEER_MOTOR_DIAMETER / STEER_GEAR_DIAMETER   # column rad per motor rad
WHEEL_RADIUS = 0.108                                       # 8.5 in tire
WHEEL_SPEED_SCALE = 0.545                                  # true / logged ground speed, GPS-fitted
GYRO_Z_BIAS = math.radians(-0.085)                          # rad/s, IMU frame, from the stationary log
MOTOR_SIGN = -1.0
YAW_SIGN = -1.0
STEER_SIGN = -1.0

DEFAULT_HZ = 30.0
DEFAULT_MIN_SPEED = 0.3
GRAVITY = 9.81


@dataclass(frozen=True)
class RolloutWindows:
    """Fixed-horizon windows shaped for ``DiffGM3.rollout``."""

    surface: str
    dt: float
    initial_state: np.ndarray  # [B, 8]
    controls: np.ndarray       # [T, B, 2]  [omega, theta_motor]
    slopes: np.ndarray         # [T, B, 2]
    target_states: np.ndarray  # [T + 1, B, 8]

    def torch(self, dtype=None):
        import torch

        dtype = dtype or torch.get_default_dtype()
        as_tensor = lambda array: torch.as_tensor(array, dtype=dtype)  # noqa: E731
        return {
            "initial_state": as_tensor(self.initial_state),
            "controls": as_tensor(self.controls),
            "slopes": as_tensor(self.slopes),
            "target_states": as_tensor(self.target_states),
        }


def contiguous_runs(valid: np.ndarray, bag_index: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [begin, end) spans that are valid throughout and from one run."""
    runs: list[tuple[int, int]] = []
    begin = None
    for i in range(len(valid)):
        breaks = begin is not None and (not valid[i] or bag_index[i] != bag_index[begin])
        if breaks:
            runs.append((begin, i))
            begin = i if valid[i] else None
        elif begin is None and valid[i]:
            begin = i
    if begin is not None:
        runs.append((begin, len(valid)))
    return runs


def steer_column(theta_motor: np.ndarray, zero: float = 0.0, ratio: float = STEER_RATIO) -> np.ndarray:
    """Motor angle (rad) to GM3 steering angle (rad, positive left)."""
    return STEER_SIGN * ratio * (theta_motor - zero)


@dataclass
class ScooterRun:
    """One CSV log, resampled to a uniform grid.

    ``states`` is ``[N, 8]`` in GM3 layout with ``x, y, psi`` dead-reckoned
    from wheel speed and gyro. ``controls`` is ``[N, 2]`` = ``[omega,
    theta_motor]`` -- the *raw* motor angle, so a fit can own the zero and
    ratio; map it with :func:`steer_column` before handing it to GM3.
    """

    name: str
    dt: float
    steer_zero_deg: float   # motor angle at straight-ahead, estimated per run
    t: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    valid: np.ndarray
    slopes: np.ndarray      # [N, 2] body-frame surface angles [alpha_p, alpha_r], rad
    gyro: np.ndarray        # [N, 3] rates in the vehicle frame (x forward, y left, z up), bias not removed
    accel: np.ndarray       # [N, 3] specific force in the vehicle frame
    odometer: np.ndarray    # [N] scaled distance since start, m
    gps_t: np.ndarray       # [M]
    gps_xy: np.ndarray      # [M, 2] local east/north, m, origin at first fix
    gps_sigma: np.ndarray   # [M] horizontal 1-sigma, m

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def moving_seconds(self) -> float:
        return float(self.valid.sum() * self.dt)

    @property
    def distance(self) -> float:
        return float(self.odometer[-1] - self.odometer[0])

    @property
    def gps_path_length(self) -> float:
        if len(self.gps_xy) < 2:
            return float("nan")
        return float(np.linalg.norm(np.diff(self.gps_xy, axis=0), axis=1).sum())

    def windows(self, horizon: int = 45, stride: int | None = None) -> RolloutWindows:
        """Contiguous moving stretches cut into fixed-horizon windows."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        stride = horizon if stride is None else stride
        starts: list[int] = []
        for begin, end in contiguous_runs(self.valid, np.zeros(len(self.valid), dtype=int)):
            starts.extend(range(begin, end - horizon, stride))
        if not starts:
            raise ValueError(f"no {horizon}-step window fits in the moving samples of {self.name!r}")
        index = np.asarray(starts)
        steps = np.arange(horizon)[:, None] + index[None, :]
        return RolloutWindows(
            surface="scooter",
            dt=self.dt,
            initial_state=self.states[index],
            controls=self.controls[steps],
            slopes=self.slopes[steps],
            target_states=self.states[np.arange(horizon + 1)[:, None] + index[None, :]],
        )


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)[1:]  # drop the ISO timestamp column
        rows = [[float(x) if x not in ("", "nan") else math.nan for x in row[1:]] for row in reader]
    table = np.asarray(rows, dtype=float)
    return {name: table[:, i] for i, name in enumerate(header)}


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width) / width
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_run(
    path: Path | str,
    *,
    hz: float = DEFAULT_HZ,
    min_speed: float = DEFAULT_MIN_SPEED,
    speed_scale: float = WHEEL_SPEED_SCALE,
    smooth_window: int = 3,
    slope_source: str = "imu",
    gyro_bias: float = GYRO_Z_BIAS,
) -> ScooterRun:
    path = Path(path)
    raw = _read_csv(path)
    if "speed_mps" in raw:
        return _load_corrected(path, raw, hz=hz, min_speed=min_speed, smooth_window=smooth_window,
                               slope_source=slope_source)
    t_raw = raw["time_s"]
    dt = 1.0 / hz
    t = np.arange(t_raw[0], t_raw[-1], dt)

    def grid(name: str) -> np.ndarray:
        return np.interp(t, t_raw, raw[name])

    v_front = _moving_average(MOTOR_SIGN * speed_scale * grid("drive_velocity_mps"), smooth_window)
    r = YAW_SIGN * (grid("imu_gyro_z_rad_s") - gyro_bias)
    theta = np.radians(grid("steer_angle_deg"))
    moving = _moving_average(v_front, 15) > min_speed
    zero = _steer_zero(v_front[moving], theta[moving], r[moving]) if moving.sum() > 30 else 0.0
    delta = steer_column(theta, zero)
    vx = v_front * np.cos(delta)
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * dt)])
    vy = make_scooter_config().lr * r
    x_dot = vx * np.cos(psi) - vy * np.sin(psi)
    y_dot = vx * np.sin(psi) + vy * np.cos(psi)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (x_dot[1:] + x_dot[:-1]) * dt)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (y_dot[1:] + y_dot[:-1]) * dt)])
    zeros = np.zeros_like(t)
    states = np.stack([x, y, psi, vx, vy, r, zeros, zeros], axis=1)
    valid = moving
    controls = np.stack([v_front / WHEEL_RADIUS, theta - zero], axis=1)

    # IMU +y is forward, +x is left, z is down: swap x/y and negate z.
    gyro = np.stack([grid("imu_gyro_y_rad_s"), grid("imu_gyro_x_rad_s"), -grid("imu_gyro_z_rad_s")], axis=1)
    accel = np.stack([grid("imu_accel_y_mps2"), grid("imu_accel_x_mps2"), -grid("imu_accel_z_mps2")], axis=1)
    slopes = surface_angles(t, vx, accel, r, valid, source=slope_source)
    odometer = MOTOR_SIGN * speed_scale * grid("drive_distance_since_start_m")   # front-wheel path

    has_fix = (raw["gps_received"] == 1) & np.isfinite(raw["gps_latitude_deg"])
    stamps = raw["gps_ros_stamp_s"][has_fix]
    _, first = np.unique(stamps, return_index=True)
    lat = raw["gps_latitude_deg"][has_fix][first]
    lon = raw["gps_longitude_deg"][has_fix][first]
    cov = raw["gps_cov_x_m2"][has_fix][first] + raw["gps_cov_y_m2"][has_fix][first]
    clock_offset = raw["imu_ros_stamp_s"][0] - t_raw[0]
    gps_t = stamps[first] - clock_offset
    if len(lat):
        east = np.radians(lon - lon[0]) * 6_371_000.0 * math.cos(math.radians(lat[0]))
        north = np.radians(lat - lat[0]) * 6_371_000.0
        gps_xy = np.stack([east, north], axis=1)
    else:
        gps_xy = np.zeros((0, 2))

    return ScooterRun(
        name=path.stem.replace("odrive_gps_imu_", ""),
        dt=dt, steer_zero_deg=math.degrees(zero), t=t, states=states, controls=controls, valid=valid,
        slopes=slopes, gyro=gyro, accel=accel, odometer=odometer,
        gps_t=gps_t, gps_xy=gps_xy, gps_sigma=np.sqrt(0.5 * cov),
    )


def _load_corrected(path: Path, raw: dict[str, np.ndarray], *, hz: float, min_speed: float,
                    smooth_window: int, slope_source: str) -> ScooterRun:
    """Build a run from a ``scooter_correct`` CSV: every correction is already
    applied, so this only resamples, derives the pose and picks up the GPS rows."""
    t_raw = raw["time_s"]
    dt = 1.0 / hz
    t = np.arange(t_raw[0], t_raw[-1], dt)

    def grid(name: str) -> np.ndarray:
        return np.interp(t, t_raw, raw[name])

    v_front = _moving_average(grid("speed_mps"), smooth_window)
    r = grid("yaw_rate_rad_s")
    theta = np.radians(grid("steer_motor_deg"))           # already relative to straight-ahead
    moving = _moving_average(v_front, 15) > min_speed
    delta = steer_column(theta, 0.0)
    vx = v_front * np.cos(delta)
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * dt)])
    vy = make_scooter_config().lr * r
    x_dot = vx * np.cos(psi) - vy * np.sin(psi)
    y_dot = vx * np.sin(psi) + vy * np.cos(psi)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (x_dot[1:] + x_dot[:-1]) * dt)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (y_dot[1:] + y_dot[:-1]) * dt)])
    zeros = np.zeros_like(t)
    states = np.stack([x, y, psi, vx, vy, r, zeros, zeros], axis=1)
    controls = np.stack([v_front / WHEEL_RADIUS, theta], axis=1)
    gyro = np.stack([grid(f"gyro_{a}_rad_s") for a in "xyz"], axis=1)
    accel = np.stack([grid(f"accel_{a}_mps2") for a in "xyz"], axis=1)
    if slope_source == "imu":
        slopes = np.radians(np.stack([grid("grade_deg"), grid("bank_deg")], axis=1))
    else:
        slopes = surface_angles(t, vx, accel, r, moving, source=slope_source)
    fix = raw["gps_new_fix"] == 1
    gps_t = raw["gps_time_s"][fix]
    gps_xy = np.stack([raw["gps_east_m"][fix], raw["gps_north_m"][fix]], axis=1) if fix.any() else np.zeros((0, 2))
    return ScooterRun(
        name=path.stem.replace("odrive_gps_imu_", ""),
        dt=dt, steer_zero_deg=0.0, t=t, states=states, controls=controls, valid=moving,
        slopes=slopes, gyro=gyro, accel=accel, odometer=grid("distance_m"),
        gps_t=gps_t, gps_xy=gps_xy, gps_sigma=raw["gps_sigma_m"][fix],
    )


def _lowpass(values: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        a = math.exp(-(t[i] - t[i - 1]) / tau)
        out[i] = a * out[i - 1] + (1.0 - a) * values[i]
    return out


def surface_angles(t: np.ndarray, vx: np.ndarray, accel: np.ndarray, r: np.ndarray, moving: np.ndarray,
                   *, source: str = "imu", tau: float = 3.0) -> np.ndarray:
    """Body-frame surface angles ``[alpha_p, alpha_r]`` in radians, ``alpha_p > 0``
    climbing along body ``+x`` and ``alpha_r > 0`` with the ``+y`` (left) side uphill.
    ``accel`` is the specific force in the vehicle frame (x forward, y left, z up).

    ``"imu"``: the accelerometer's forward specific force is ``dv/dt + g sin(grade)``,
    so subtracting the wheel-derived acceleration and low-passing (``tau`` s)
    isolates the grade. This is not the fused attitude's pitch: that estimate
    treats every kick as a 6 deg tilt and reads +-10 deg on a flat quad. The
    lateral channel does the same with ``v * r`` removed. Each run's median is
    subtracted, which absorbs the IMU mounting tilt and also any grade that is
    constant over the run. About 1 deg of noise remains at ``tau = 3``.

    ``"plane"``: a planar ground fitted to the run's own coasting behaviour,
    ``dv/dt = c0 + A cos(psi) + B sin(psi)`` on samples that are not being
    kicked; the gradient ``(A, B)`` is rotated into the body frame. It does not
    depend on the IMU. On these runs the amplitude is 0.3-0.7 deg with R^2
    under 0.15, i.e. the terrain is flat within the noise.

    ``"none"``: zeros, the flat-ground dynamics.
    """
    if source == "none":
        return np.zeros((len(t), 2))
    dv = np.gradient(_moving_average(vx, 15), t)
    if source == "imu":
        forward = accel[:, 0]
        lateral = accel[:, 1]
        sin_p = _lowpass((forward - dv) / GRAVITY, t, tau)
        sin_r = _lowpass((lateral - vx * r) / GRAVITY, t, tau)
        alpha_p = np.arcsin(np.clip(sin_p, -1.0, 1.0))
        alpha_r = np.arcsin(np.clip(sin_r, -1.0, 1.0))
        alpha_p -= np.median(alpha_p[moving]) if moving.any() else 0.0
        alpha_r -= np.median(alpha_r[moving]) if moving.any() else 0.0
        return np.stack([alpha_p, alpha_r], axis=1)
    if source == "plane":
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t))])
        coast = moving & (dv < 0.15)
        if coast.sum() < 30:
            return np.zeros((len(t), 2))
        design = np.stack([np.ones(coast.sum()), np.cos(psi[coast]), np.sin(psi[coast])], axis=1)
        _, a, b = np.linalg.lstsq(design, dv[coast], rcond=None)[0]
        # dv/dt = -g sin(alpha_p) on the coast, so the world gradient is -(a, b) / g.
        gx, gy = -a / GRAVITY, -b / GRAVITY
        alpha_p = np.arcsin(np.clip(gx * np.cos(psi) + gy * np.sin(psi), -1.0, 1.0))
        alpha_r = np.arcsin(np.clip(-gx * np.sin(psi) + gy * np.cos(psi), -1.0, 1.0))
        return np.stack([alpha_p, alpha_r], axis=1)
    raise ValueError(f"unknown slope source {source!r}: use 'imu', 'plane' or 'none'")


def _steer_zero(v: np.ndarray, theta: np.ndarray, r: np.ndarray) -> float:
    """Motor angle at which the scooter runs straight: ``r = a v (theta - zero)``.

    Uses the front-wheel speed and a linear-in-theta form; both are fine for
    locating the zero (it is the intercept, unaffected by the gain's scale or
    its mild nonlinearity), and the zero has to be known before the body
    speed can be computed.
    """
    design = np.stack([v * theta, v], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(design, r, rcond=None)
    return float(-intercept / slope) if abs(slope) > 1e-9 else 0.0


def discover(root: Path | str | None = None) -> list[Path]:
    root = DEFAULT_ROOT if root is None else Path(root)
    return sorted(root.glob("*.csv"))


def load_all(root: Path | str | None = None, *, min_moving_seconds: float = 5.0, **kwargs) -> list[ScooterRun]:
    """Every run under ``root`` that actually moves, in recording order."""
    runs = [load_run(path, **kwargs) for path in discover(root)]
    return [run for run in runs if run.moving_seconds >= min_moving_seconds]


def concatenate_windows(parts: list[RolloutWindows]) -> RolloutWindows:
    if not parts:
        raise ValueError("no windows to concatenate")
    return RolloutWindows(
        surface="scooter",
        dt=parts[0].dt,
        initial_state=np.concatenate([p.initial_state for p in parts], axis=0),
        controls=np.concatenate([p.controls for p in parts], axis=1),
        slopes=np.concatenate([p.slopes for p in parts], axis=1),
        target_states=np.concatenate([p.target_states for p in parts], axis=1),
    )


def windows_of(runs: list[ScooterRun], horizon: int, stride: int | None = None) -> RolloutWindows:
    return concatenate_windows([run.windows(horizon, stride) for run in runs])


def kinematic_steer_fit(runs: list[ScooterRun], *, ratio: float = STEER_RATIO) -> dict[str, float]:
    """Regress yaw rate on ``vx * tan(delta)`` and ``vx`` over every moving sample.

    In the kinematic-bicycle limit ``r = vx * tan(delta) / L`` with ``delta``
    from the stated gear ratio, so the slope gives the implied wheelbase and
    the intercept any residual straight-ahead offset beyond the per-run zero
    the loader already removed (reported in motor degrees through the
    small-angle gain). R^2 near 1 means the tires are behaving kinematically
    at the recorded lateral accelerations.
    """
    rows, targets = [], []
    for run in runs:
        v = run.states[run.valid, 3]
        delta = steer_column(run.controls[run.valid, 1], 0.0, ratio)
        rows.append(np.stack([v * np.tan(delta), v], axis=1))
        targets.append(run.states[run.valid, 5])
    design = np.concatenate(rows)
    target = np.concatenate(targets)
    (slope, intercept), *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ np.array([slope, intercept])
    return {
        "gain": float(slope),
        "steer_zero_deg": float(math.degrees(-intercept / (slope * STEER_SIGN * ratio))),
        "wheelbase_implied": float(1.0 / abs(slope)),
        "r2": float(1.0 - residual.var() / target.var()),
        "samples": int(len(target)),
    }


def gps_errors(path_xy: np.ndarray, path_t: np.ndarray, run: ScooterRun) -> dict[str, float]:
    """Errors of a predicted path against the GPS fixes it spans, after the
    best rigid rotation + translation (Kabsch in 2D).

    The alignment removes the unknown initial heading and origin, which dead
    reckoning cannot know, and leaves shape error. ``ADE`` is the mean
    distance to the fixes, ``RMS`` the root mean square, ``FDE`` the distance
    at the last fix.
    """
    empty = {"ADE": float("nan"), "RMS": float("nan"), "FDE": float("nan"), "n": 0}
    if len(run.gps_t) < 3:
        return empty
    keep = (run.gps_t >= path_t[0]) & (run.gps_t <= path_t[-1])
    if keep.sum() < 3:
        return empty
    gps = run.gps_xy[keep]
    pred = np.stack([np.interp(run.gps_t[keep], path_t, path_xy[:, i]) for i in range(2)], axis=1)
    p0, g0 = pred - pred.mean(0), gps - gps.mean(0)
    u, _, vt = np.linalg.svd(p0.T @ g0)
    d = np.sign(np.linalg.det(u @ vt))
    aligned = p0 @ (u @ np.diag([1.0, d]) @ vt)
    distance = np.linalg.norm(aligned - g0, axis=1)
    # Frechet: the whole aligned path (thinned to ~5 Hz) against the fix polyline.
    rotation = u @ np.diag([1.0, d]) @ vt
    span = (path_t >= run.gps_t[keep][0]) & (path_t <= run.gps_t[keep][-1])
    step = max(1, int(round(0.2 / max(float(np.median(np.diff(path_t))), 1e-6))))
    curve = ((path_xy[span] - pred.mean(0)) @ rotation)[::step]
    return {"ADE": float(distance.mean()), "RMS": float(np.sqrt((distance ** 2).mean())),
            "FDE": float(distance[-1]), "frechet": discrete_frechet(curve, g0), "n": int(len(distance))}


def discrete_frechet(p: np.ndarray, q: np.ndarray) -> float:
    """Discrete Frechet distance between two polylines (Eiter & Mannila)."""
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    ca = np.empty_like(d)
    ca[0, 0] = d[0, 0]
    ca[1:, 0] = np.maximum.accumulate(d[1:, 0], axis=0)
    ca[1:, 0] = np.maximum(ca[1:, 0], ca[0, 0])
    ca[0, 1:] = np.maximum(np.maximum.accumulate(d[0, 1:]), ca[0, 0])
    for i in range(1, d.shape[0]):
        row = ca[i - 1]
        for j in range(1, d.shape[1]):
            ca[i, j] = max(min(row[j], row[j - 1], ca[i, j - 1]), d[i, j])
    return float(ca[-1, -1])


def rigid_align_rms(path_xy: np.ndarray, path_t: np.ndarray, run: ScooterRun) -> float:
    """RMS distance to the GPS fixes after rigid alignment; see :func:`gps_errors`."""
    return gps_errors(path_xy, path_t, run)["RMS"]
