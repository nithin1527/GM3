from __future__ import annotations

import numpy as np

from gm3.shared.constants import MAX_ALPHA, MAX_KAPPA, MAX_SIGMA, MIN_KAPPA, SLIP_EPS
from gm3.shared.types import VehicleConfig


def body_to_tire_velocity(
    *,
    vx: float,
    vy: float,
    yaw_rate: float,
    tire_x: float,
    steering_angle: float,
) -> tuple[float, float]:
    vx_point = vx
    vy_point = vy + yaw_rate * tire_x
    cos_delta = np.cos(steering_angle)
    sin_delta = np.sin(steering_angle)
    vx_tire = vx_point * cos_delta + vy_point * sin_delta
    vy_tire = -vx_point * sin_delta + vy_point * cos_delta
    return float(vx_tire), float(vy_tire)


def slip(
    *,
    vx_tire: float,
    vy_tire: float,
    omega: float,
    tire_radius: float,
    eps: float,
) -> dict[str, float]:
    denom = max(abs(vx_tire), SLIP_EPS)
    kappa = (tire_radius * omega - vx_tire) / denom
    kappa = float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))
    alpha = float(np.arctan2(vy_tire, denom))

    one_plus_kappa = max(1.0 + kappa, eps)
    sigma_x = -kappa / one_plus_kappa
    sigma_y = np.tan(np.clip(alpha, -MAX_ALPHA, MAX_ALPHA)) / one_plus_kappa
    return {
        "sigma_x": float(np.clip(sigma_x, -MAX_SIGMA, MAX_SIGMA)),
        "sigma_y": float(np.clip(sigma_y, -MAX_SIGMA, MAX_SIGMA)),
        "kappa": kappa,
        "alpha": alpha,
    }


def brush_forces(
    *,
    config: VehicleConfig,
    sigma_x: float,
    sigma_y: float,
    normal_load: float,
    alpha: float,
    steering_angle: float,
    gamma: float,
    tire_radius: float,
    mu: float,
    cp: float,
    contact_length: float,
    can_lean: bool,
) -> dict[str, float]:
    normal_load = max(normal_load, config.min_normal_load)
    theta_y = (2.0 * cp * contact_length * contact_length) / (3.0 * mu * normal_load)
    theta_star = theta_y

    if can_lean:
        tan_delta = np.tan(steering_angle)
        sign_delta = np.sign(steering_angle)
        turn_radius = config.wheelbase / (abs(tan_delta) + config.eps)
        phi_spin = -sign_delta / (turn_radius + config.eps) + 0.9 * np.sin(gamma) / tire_radius
        denom = 1.0 - contact_length * phi_spin * theta_y * np.sign(alpha if abs(alpha) > config.eps else 1.0)
        if abs(denom) < config.eps:
            denom = config.eps if denom >= 0.0 else -config.eps
        theta_star = theta_y / denom

    sigma = float(np.hypot(sigma_x, sigma_y))
    if sigma < config.eps:
        return {"fx": 0.0, "fy": 0.0, "mz": 0.0}

    sigma_sliding = 1.0 / max(abs(theta_star), config.eps)
    t = theta_star * sigma

    if sigma <= sigma_sliding:
        force_total = mu * normal_load * t * (3.0 - 3.0 * t + t * t)
        mz = -mu * normal_load * contact_length * theta_star * sigma_y * (
            1.0 - 3.0 * t + 3.0 * t * t - t * t * t
        )
    else:
        force_total = mu * normal_load
        mz = 0.0

    return {
        "fx": float(-force_total * sigma_x / sigma),
        "fy": float(-force_total * sigma_y / sigma),
        "mz": float(mz),
    }

