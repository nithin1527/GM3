from __future__ import annotations

from new_gm3.diffgm3.vehicle import DiffGM3Vehicle
from new_gm3.shared.types import VehicleConfig


class DiffGM3(DiffGM3Vehicle):
    """Public trainable PyTorch DiffGM3 model."""

    def __init__(self, config: VehicleConfig, dt: float = 0.05):
        super().__init__(config=config, dt=dt)


__all__ = ["DiffGM3"]

