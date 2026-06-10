from __future__ import annotations

import numpy as np

from .types import VehicleConfig


def axle_masks(tire_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    front = tire_x > 0.0
    return front, ~front


def wheelbase(config: VehicleConfig) -> float:
    return config.lf + config.lr


def effective_roll_inertia(config: VehicleConfig) -> float:
    return config.effective_roll_inertia

