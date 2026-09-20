from __future__ import annotations

from gm3.diffgm3.tire_models import TireModel
from gm3.diffgm3.vehicle import DiffGM3Vehicle
from gm3.shared.types import VehicleConfig


class DiffGM3(DiffGM3Vehicle):
    """Public trainable PyTorch DiffGM3 model.

    ``tire_model`` picks the tire law: ``"brush"`` (default, the original GM3
    tire), ``"fiala"``, ``"dugoff"``, ``"burckhardt"`` or ``"pacejka"``, or a
    ``TireModel`` module. Extra keywords go to the law, see ``tire_models``.
    """

    def __init__(self, config: VehicleConfig, dt: float = 0.05, *, tire_model: str | TireModel = "brush",
                 speed_epsilon: float | None = None, **tire_options):
        super().__init__(config=config, dt=dt, tire_model=tire_model, speed_epsilon=speed_epsilon, **tire_options)


__all__ = ["DiffGM3"]

