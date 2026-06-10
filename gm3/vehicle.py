from __future__ import annotations

import numpy as np

from new_gm3.shared.tire import tire_flag_arrays, tire_physical_arrays
from new_gm3.shared.types import GM3State, VehicleConfig
from new_gm3.shared.utils import tire_count_by_axle


class GM3Vehicle:
    """Precomputed non-torch vehicle/tire arrays for pure GM3."""

    def __init__(self, config: VehicleConfig):
        self.config = config
        physical = tire_physical_arrays(config)
        flags = tire_flag_arrays(config)

        self.tire_x = physical["x"]
        self.tire_y = physical["y"]
        self.tire_radius = physical["radius"]
        self.tire_mu = physical["mu"]
        self.tire_cp = physical["cp"]
        self.tire_contact_length = physical["contact_length"]

        self.steerable = flags["steerable"]
        self.driven = flags["driven"]
        self.can_lean = flags["can_lean"]

        self.front = self.tire_x > 0.0
        self.rear = ~self.front
        self.front_count, self.rear_count = tire_count_by_axle(self.tire_x)
        self.lateral_load_mask = np.abs(self.tire_y) > config.eps

    def normal_loads(self, state: GM3State) -> np.ndarray:
        cfg = self.config
        wheelbase = max(cfg.wheelbase, cfg.eps)

        ax_est = state.vy * state.r
        ay_est = -state.vx * state.r
        longitudinal = cfg.mass * ax_est * cfg.cg_height / wheelbase

        front_static = cfg.mass * cfg.gravity * cfg.lr / wheelbase / self.front_count
        rear_static = cfg.mass * cfg.gravity * cfg.lf / wheelbase / self.rear_count

        loads = np.where(
            self.front,
            front_static - longitudinal / self.front_count,
            rear_static + longitudinal / self.rear_count,
        )

        if cfg.width > cfg.eps and self.lateral_load_mask.any():
            lateral = cfg.mass * ay_est * cfg.cg_height / max(cfg.width, cfg.eps)
            y_norm = self.tire_y / max(cfg.width * 0.5, cfg.eps)
            loads = loads - lateral * y_norm * self.lateral_load_mask

        return np.maximum(loads, cfg.min_normal_load)

    def steering_angles(self, delta: float) -> np.ndarray:
        cfg = self.config
        if cfg.steering_mode == "direct":
            angles = np.full_like(self.tire_x, float(delta), dtype=float)
        else:
            tan_delta = np.tan(delta)
            numerator = cfg.wheelbase * tan_delta
            denominator = cfg.wheelbase - self.tire_y * tan_delta
            angles = np.arctan2(numerator, denominator)
        return np.where(self.steerable, angles, 0.0)

    def aggregate_body_forces(
        self,
        *,
        steering_angles: np.ndarray,
        fx_tire: np.ndarray,
        fy_tire: np.ndarray,
        mz_tire: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        cos_delta = np.cos(steering_angles)
        sin_delta = np.sin(steering_angles)
        fx_body = fx_tire * cos_delta - fy_tire * sin_delta
        fy_body = fx_tire * sin_delta + fy_tire * cos_delta
        mz_body = self.config.align_gain * mz_tire + self.tire_x * fy_body - self.tire_y * fx_body

        return {
            "body_forces": np.stack([fx_body, fy_body, mz_body], axis=-1),
            "force_total": np.array([fx_body.sum(), fy_body.sum()], dtype=float),
            "moment_total": float(mz_body.sum()),
        }

