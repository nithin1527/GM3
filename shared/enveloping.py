"""Radial-interradial spring tire model (Badalamenti & Doyle, 1988).

A tire is discretized into N radial spring elements around the wheel. Elements
touching the ground are compressed; linear "interradial" springs couple
neighbors so the tire envelopes short-wavelength obstacles smoothly instead of
stepping over them. This module holds the numpy reference solver and the
per-wheel calibration used at model-build time; the batched torch version lives
in ``gm3.diffgm3.enveloping`` and reproduces these results.

Convention (matches the frontend ``envelopingTire.ts``):
    theta_i  angle from horizontal to a radial element; 90 deg points down.
    h_axle   axle height above nominal flat ground (z = 0).
    Zp_i     ground-profile height under element i.
    E_vi = Zp_i - h_axle + R*sin(theta_i)   (compression, only where positive)
    E_ri = E_vi / sin(theta_i)
On flat ground the summed Fz is the calibrated static load and Fd ~ 0, so callers
apply the deviation from flat (deltaFz) and the raw drag Fd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Calibration defaults (see EnvelopingTuning in the frontend).
DEFAULT_N = 21
DEFAULT_SPAN_DEG = 60.0
DEFAULT_DELTA0 = 0.03
DEFAULT_KAPPA = 1.0
DEFAULT_C2RATIO = -0.05


@dataclass(frozen=True)
class EnvelopingParams:
    """Calibrated per-wheel parameters for the enveloping model."""

    N: int
    R: float
    h_axle: float
    C1: float
    C2: float
    k: float
    flat_fz: float
    sin_t: np.ndarray  # [N]
    cos_t: np.ndarray  # [N]


def element_angles(n: int, span_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = 90.0 - span_deg, 90.0 + span_deg
    deg = np.linspace(lo, hi, n) if n > 1 else np.array([90.0])
    th = np.deg2rad(deg)
    return np.sin(th), np.cos(th)


def solve_forces(
    *,
    sin_t: np.ndarray,
    cos_t: np.ndarray,
    R: float,
    h_axle: float,
    C1: float,
    C2: float,
    k: float,
    Zp: np.ndarray,
    sweeps: int = 40,
) -> tuple[float, float]:
    """Total vertical (Fz) and drag (Fd) force for one wheel over profile Zp[N]."""
    n = sin_t.shape[0]
    Ev = Zp - h_axle + R * sin_t
    contact = Ev > 0.0
    Er = np.where(contact, Ev / sin_t, 0.0)

    # Jacobi relaxation for the free elements: F_ri = 0 gives, for an interior
    # free element, (C1 + 2k) E = k (E_left + E_right); ends use one neighbor.
    for _ in range(sweeps):
        left = np.concatenate([[0.0], Er[:-1]])
        right = np.concatenate([Er[1:], [0.0]])
        new = k * (left + right) / (C1 + 2.0 * k)
        new[0] = k * Er[1] / (C1 + k)
        new[-1] = k * Er[-2] / (C1 + k)
        Er = np.where(contact, Er, new)

    left = np.concatenate([[0.0], Er[:-1]])
    right = np.concatenate([Er[1:], [0.0]])
    coupling = 2.0 * Er - left - right
    coupling[0] = Er[0] - Er[1]
    coupling[-1] = Er[-1] - Er[-2]

    Fr = C1 * Er + C2 * Er * Er + k * coupling
    Fv = np.maximum(Fr * sin_t, 0.0)
    Fz = float(Fv.sum())
    Fd = float((Fv * (cos_t / sin_t)).sum())
    return Fz, Fd


def calibrate_enveloping(
    R: float,
    static_load: float,
    *,
    n: int = DEFAULT_N,
    span_deg: float = DEFAULT_SPAN_DEG,
    delta0: float = DEFAULT_DELTA0,
    kappa: float = DEFAULT_KAPPA,
    c2ratio: float = DEFAULT_C2RATIO,
) -> EnvelopingParams:
    """Fit C1/C2/k so a flat-ground wheel at delta0 deflection carries static_load."""
    sin_t, cos_t = element_angles(n, span_deg)
    h_axle = R - delta0
    flat = np.zeros(n)
    load = max(static_load, 1e-3)

    C1 = 1.0
    for _ in range(6):
        k = kappa * C1
        C2 = (c2ratio * C1) / delta0
        fz, _ = solve_forces(sin_t=sin_t, cos_t=cos_t, R=R, h_axle=h_axle, C1=C1, C2=C2, k=k, Zp=flat)
        if fz > 1e-9:
            C1 *= load / fz
    k = kappa * C1
    C2 = (c2ratio * C1) / delta0
    flat_fz, _ = solve_forces(sin_t=sin_t, cos_t=cos_t, R=R, h_axle=h_axle, C1=C1, C2=C2, k=k, Zp=flat)
    return EnvelopingParams(N=n, R=R, h_axle=h_axle, C1=C1, C2=C2, k=k, flat_fz=flat_fz, sin_t=sin_t, cos_t=cos_t)
