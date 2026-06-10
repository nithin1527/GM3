from __future__ import annotations

import numpy as np

from .types import VehicleConfig


def tire_positions(config: VehicleConfig) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([tire.x for tire in config.tires], dtype=float),
        np.array([tire.y for tire in config.tires], dtype=float),
    )


def tire_physical_arrays(config: VehicleConfig) -> dict[str, np.ndarray]:
    return {
        "x": np.array([tire.x for tire in config.tires], dtype=float),
        "y": np.array([tire.y for tire in config.tires], dtype=float),
        "radius": np.array([tire.radius for tire in config.tires], dtype=float),
        "mu": np.array([tire.mu for tire in config.tires], dtype=float),
        "cp": np.array([tire.cp for tire in config.tires], dtype=float),
        "contact_length": np.array([tire.contact_length for tire in config.tires], dtype=float),
    }


def tire_flag_arrays(config: VehicleConfig) -> dict[str, np.ndarray]:
    return {
        "steerable": np.array([tire.steerable for tire in config.tires], dtype=bool),
        "driven": np.array([tire.driven for tire in config.tires], dtype=bool),
        "can_lean": np.array([tire.can_lean for tire in config.tires], dtype=bool),
    }

