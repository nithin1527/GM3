from __future__ import annotations

import torch

from new_gm3.diffgm3.torch_utils import smooth_abs
from new_gm3.shared.constants import MAX_ALPHA, MAX_KAPPA, MAX_SIGMA, MIN_KAPPA, SLIP_EPS


def body_to_tire_velocities(
    *,
    vx: torch.Tensor,
    vy: torch.Tensor,
    yaw_rate: torch.Tensor,
    tire_x: torch.Tensor,
    steering_angles: torch.Tensor,
) -> torch.Tensor:
    vx_point = vx.unsqueeze(-1)
    vy_point = vy.unsqueeze(-1) + yaw_rate.unsqueeze(-1) * tire_x.unsqueeze(0)
    cos_delta = torch.cos(steering_angles)
    sin_delta = torch.sin(steering_angles)
    vx_tire = vx_point * cos_delta + vy_point * sin_delta
    vy_tire = -vx_point * sin_delta + vy_point * cos_delta
    return torch.stack([vx_tire, vy_tire], dim=-1)


def slip(
    *,
    vx_tire: torch.Tensor,
    vy_tire: torch.Tensor,
    omega_tire: torch.Tensor,
    tire_radius: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = smooth_abs(vx_tire, SLIP_EPS)
    kappa = (tire_radius.unsqueeze(0) * omega_tire - vx_tire) / denom
    kappa = torch.clamp(kappa, min=MIN_KAPPA, max=MAX_KAPPA)
    alpha = torch.atan2(vy_tire, denom)
    alpha_safe = MAX_ALPHA * torch.tanh(alpha / MAX_ALPHA)
    one_plus = (1.0 + kappa).clamp_min(eps)
    sigma_x = -kappa / one_plus
    sigma_y = torch.tan(alpha_safe) / one_plus
    sigma_x = MAX_SIGMA * torch.tanh(sigma_x / MAX_SIGMA)
    sigma_y = MAX_SIGMA * torch.tanh(sigma_y / MAX_SIGMA)
    return sigma_x, sigma_y, kappa, alpha


def brush_forces(
    *,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    normal_loads: torch.Tensor,
    alpha: torch.Tensor,
    steering_angles: torch.Tensor,
    gamma: torch.Tensor,
    tire_radius: torch.Tensor,
    wheelbase: torch.Tensor,
    eps: torch.Tensor,
    min_normal_load: torch.Tensor,
    lean_mask: torch.Tensor,
    has_leaning_tires: bool,
    mu: torch.Tensor,
    cp: torch.Tensor,
    contact_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = mu.unsqueeze(0)
    cp = cp.unsqueeze(0)
    contact_length = contact_length.unsqueeze(0)
    normal_loads = normal_loads.clamp_min(min_normal_load)

    theta_y = (2.0 * cp * contact_length * contact_length) / (3.0 * mu * normal_loads)
    theta_star = theta_y

    if has_leaning_tires:
        tan_delta = torch.tan(steering_angles)
        sign_delta = torch.tanh(steering_angles / 1e-3)
        turn_radius = wheelbase.clamp_min(eps) / (smooth_abs(tan_delta, 1e-6) + eps)
        phi_spin = -sign_delta / (turn_radius + eps) + 0.9 * torch.sin(gamma).unsqueeze(-1) / tire_radius.unsqueeze(0)
        denom = 1.0 - contact_length * phi_spin * theta_y * torch.tanh(alpha * 10.0)
        denom_sign = torch.where(denom >= 0.0, torch.ones_like(denom), -torch.ones_like(denom))
        denom_safe = torch.where(smooth_abs(denom, 1e-8) > 1e-6, denom, 1e-6 * denom_sign)
        theta_candidate = theta_y / denom_safe
        theta_star = torch.where(lean_mask.unsqueeze(0), theta_candidate, theta_y)

    sigma = torch.sqrt(sigma_x.square() + sigma_y.square() + eps.square())
    sigma_sliding = 1.0 / torch.sqrt(theta_star.square() + eps.square())
    t = (theta_star * sigma).clamp(0.0, 1.5)

    force_adhesion = mu * normal_loads * t * (3.0 - 3.0 * t + t.square())
    force_sliding = mu * normal_loads
    gate = torch.sigmoid(20.0 * (sigma - sigma_sliding))
    force_total = (1.0 - gate) * force_adhesion + gate * force_sliding

    fx = -force_total * sigma_x / sigma
    fy = -force_total * sigma_y / sigma
    mz_raw = -mu * normal_loads * contact_length * theta_star * sigma_y * (
        1.0 - 3.0 * t + 3.0 * t.square() - t * t.square()
    )
    mz = (1.0 - gate) * mz_raw
    return fx, fy, mz

