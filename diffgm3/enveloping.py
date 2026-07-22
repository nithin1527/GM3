"""Batched, differentiable radial-interradial enveloping tire forces (torch).

Torch counterpart of ``gm3.shared.enveloping`` (numpy reference) and the frontend
``envelopingTire.ts``. Given a batch of vehicle poses and the calibrated
per-wheel spring parameters, it returns each wheel's vertical-load deviation from
flat (``delta_fz``) and its longitudinal drag (``fd``) over the active obstacle.
The free-element coupling is solved with vectorized Jacobi sweeps so the whole
thing runs as dense tensor ops over [batch, tire, element].
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from gm3.shared import terrain as T


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def obstacle_height(kind: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Surface height at world points (x, y) for the active obstacle (0 elsewhere)."""
    if kind == "speedbump":
        s = x - (T.BUMP_CX - T.BUMP_LEN / 2.0)
        on = (s >= 0.0) & (s <= T.BUMP_LEN) & (y.abs() <= T.BUMP_HW)
        w = math.pi / T.BUMP_LEN
        return torch.where(on, T.BUMP_H * torch.sin(w * s), torch.zeros_like(x))

    if kind == "pothole":
        dx = x - T.HOLE_CX
        r2 = dx * dx + y * y
        on = r2 < T.HOLE_R * T.HOLE_R
        one = 1.0 - r2 / (T.HOLE_R * T.HOLE_R)
        return torch.where(on, -T.HOLE_D * one * one, torch.zeros_like(x))

    if kind == "rough":
        on = (x >= T.ROUGH_X0) & (x <= T.ROUGH_X1) & (y.abs() <= T.ROUGH_HW)
        wx = _smoothstep(T.ROUGH_X0, T.ROUGH_X0 + 1.0, x) * (1.0 - _smoothstep(T.ROUGH_X1 - 1.0, T.ROUGH_X1, x))
        wy = 1.0 - _smoothstep(T.ROUGH_HW - 1.0, T.ROUGH_HW, y.abs())
        dx = x - T.ROUGH_X0
        base = torch.sin(T.ROUGH_KX1 * dx) + 0.5 * torch.sin(T.ROUGH_KX2 * dx + 1.3) * torch.cos(T.ROUGH_KY * y)
        return torch.where(on, T.ROUGH_A * wx * wy * base, torch.zeros_like(x))

    raise ValueError(f"unknown obstacle kind: {kind!r}")


def enveloping_wheel_forces(
    *,
    x: torch.Tensor,          # [B]
    y: torch.Tensor,          # [B]
    psi: torch.Tensor,        # [B]
    tire_x: torch.Tensor,     # [T]
    tire_y: torch.Tensor,     # [T]
    tire_radius: torch.Tensor,  # [T]
    sin_t: torch.Tensor,      # [N]
    cos_t: torch.Tensor,      # [N]
    is_first: torch.Tensor,   # [N] bool
    is_last: torch.Tensor,    # [N] bool
    C1: torch.Tensor,         # [T]
    C2: torch.Tensor,         # [T]
    k: torch.Tensor,          # [T]
    h_axle: torch.Tensor,     # [T]
    flat_fz: torch.Tensor,    # [T]
    kind: str,
    sweeps: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (delta_fz, fd), each [B, T]: load deviation from flat and drag."""
    cos_p = torch.cos(psi)[:, None]  # [B,1]
    sin_p = torch.sin(psi)[:, None]
    wx = x[:, None] + tire_x[None, :] * cos_p - tire_y[None, :] * sin_p  # [B,T]
    wy = y[:, None] + tire_x[None, :] * sin_p + tire_y[None, :] * cos_p  # [B,T]

    # Element contact points a horizontal distance R*cos(theta_i) along heading.
    off = tire_radius[None, :, None] * cos_t[None, None, :]  # [1,T,N]
    sx = wx[:, :, None] + off * cos_p[:, None]  # [B,T,N]
    sy = wy[:, :, None] + off * sin_p[:, None]
    Zp = obstacle_height(kind, sx, sy)  # [B,T,N]

    C1b = C1[None, :, None]
    kb = k[None, :, None]
    Ev = Zp - h_axle[None, :, None] + tire_radius[None, :, None] * sin_t[None, None, :]
    contact = Ev > 0.0
    Er = torch.where(contact, Ev / sin_t[None, None, :], torch.zeros_like(Ev))

    first = is_first[None, None, :]
    last = is_last[None, None, :]
    for _ in range(sweeps):
        left = F.pad(Er[..., :-1], (1, 0))
        right = F.pad(Er[..., 1:], (0, 1))
        interior = kb * (left + right) / (C1b + 2.0 * kb)
        end0 = kb * right / (C1b + kb)
        endn = kb * left / (C1b + kb)
        new = torch.where(first, end0, torch.where(last, endn, interior))
        Er = torch.where(contact, Er, new)

    left = F.pad(Er[..., :-1], (1, 0))
    right = F.pad(Er[..., 1:], (0, 1))
    coupling = torch.where(first, Er - right, torch.where(last, Er - left, 2.0 * Er - left - right))

    Fr = C1b * Er + C2[None, :, None] * Er * Er + kb * coupling
    Fv = (Fr * sin_t[None, None, :]).clamp_min(0.0)
    Fz = Fv.sum(dim=-1)  # [B,T]
    Fd = (Fv * (cos_t / sin_t)[None, None, :]).sum(dim=-1)
    return Fz - flat_fz[None, :], Fd
