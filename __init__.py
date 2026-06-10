from __future__ import annotations

from new_gm3.gm3 import GM3
from new_gm3.shared import GM3Control, GM3State, TireConfig, VehicleConfig, make_bicycle_config, make_cart_config

__all__ = [
    "DiffGM3",
    "GM3",
    "GM3Control",
    "GM3State",
    "TireConfig",
    "VehicleConfig",
    "make_bicycle_config",
    "make_cart_config",
]


def __getattr__(name: str):
    if name == "DiffGM3":
        from new_gm3.diffgm3 import DiffGM3

        return DiffGM3
    raise AttributeError(f"module 'new_gm3' has no attribute {name!r}")

