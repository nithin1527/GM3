"""Recorded-data loaders for GM3 calibration.

Currently the Hiboy scooter CSV logs (ODrive drive/steer, GPS, IMU); see
:mod:`gm3.dataset.scooter` for the conventions each field goes through.
"""

from __future__ import annotations

from .scooter import (
    DEFAULT_ROOT,
    GYRO_Z_BIAS,
    STEER_RATIO,
    STEER_SIGN,
    WHEEL_RADIUS,
    WHEEL_SPEED_SCALE,
    RolloutWindows,
    ScooterRun,
    discover,
    discrete_frechet,
    gps_errors,
    kinematic_steer_fit,
    load_all,
    load_run,
    rigid_align_rms,
    steer_column,
    surface_angles,
    windows_of,
)

__all__ = [
    "DEFAULT_ROOT", "GYRO_Z_BIAS", "STEER_RATIO", "STEER_SIGN", "WHEEL_RADIUS", "WHEEL_SPEED_SCALE",
    "RolloutWindows", "ScooterRun", "discover", "discrete_frechet", "gps_errors", "kinematic_steer_fit",
    "load_all", "load_run", "rigid_align_rms", "steer_column", "surface_angles", "windows_of",
]
