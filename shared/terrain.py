"""Localized road obstacles (speed bump, pothole, rough patch).

Mirrors the frontend terrain in GMMSim ``src/render/slopes.ts`` so the backend
can sample the same ground profile when it envelopes obstacles. Distances are in
metres in the world/sim plane; obstacles sit a few metres ahead of the spawn
point along +x. ``obstacle_height`` returns the surface height (0 off the
obstacle) and is fully vectorized over numpy arrays.
"""

from __future__ import annotations

import math

import numpy as np

OBSTACLE_KINDS = ("speedbump", "pothole", "rough")

# Speed bump: a rounded ridge across a wide y-band (half-sine in x).
BUMP_CX, BUMP_LEN, BUMP_H, BUMP_HW = 8.0, 0.7, 0.04, 6.0
# Pothole: a smooth circular depression.
HOLE_CX, HOLE_R, HOLE_D = 8.0, 0.6, 0.1
# Rough patch: a band of short-wavelength ripples.
ROUGH_X0, ROUGH_X1, ROUGH_HW, ROUGH_A = 6.0, 14.0, 6.0, 0.025
ROUGH_KX1 = 2.0 * math.pi / 0.6
ROUGH_KX2 = 2.0 * math.pi / 0.35
ROUGH_KY = 2.0 * math.pi / 1.2


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def obstacle_height(kind: str | None, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Surface height at world points (x, y) for the active obstacle (0 elsewhere)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if kind is None:
        return np.zeros_like(x)

    if kind == "speedbump":
        s = x - (BUMP_CX - BUMP_LEN / 2.0)
        on = (s >= 0.0) & (s <= BUMP_LEN) & (np.abs(y) <= BUMP_HW)
        w = math.pi / BUMP_LEN
        return np.where(on, BUMP_H * np.sin(w * s), 0.0)

    if kind == "pothole":
        dx = x - HOLE_CX
        r2 = dx * dx + y * y
        on = r2 < HOLE_R * HOLE_R
        u2 = r2 / (HOLE_R * HOLE_R)
        one = 1.0 - u2
        return np.where(on, -HOLE_D * one * one, 0.0)

    if kind == "rough":
        on = (x >= ROUGH_X0) & (x <= ROUGH_X1) & (np.abs(y) <= ROUGH_HW)
        wx = _smoothstep(ROUGH_X0, ROUGH_X0 + 1.0, x) * (1.0 - _smoothstep(ROUGH_X1 - 1.0, ROUGH_X1, x))
        wy = 1.0 - _smoothstep(ROUGH_HW - 1.0, ROUGH_HW, np.abs(y))
        dx = x - ROUGH_X0
        base = np.sin(ROUGH_KX1 * dx) + 0.5 * np.sin(ROUGH_KX2 * dx + 1.3) * np.cos(ROUGH_KY * y)
        return np.where(on, ROUGH_A * wx * wy * base, 0.0)

    raise ValueError(f"unknown obstacle kind: {kind!r}")
