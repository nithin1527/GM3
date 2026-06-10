from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "gm3":
    # Support running examples/tests from the repository root, where this
    # backend package shadows the repository package of the same name.
    _repo_root = Path(__file__).resolve().parent.parent
    __path__.append(str(_repo_root))
    sys.modules.setdefault("gm3.gm3", sys.modules[__name__])

from .model import GM3

if __name__ == "gm3":
    from gm3.shared import GM3Control, GM3State, TireConfig, VehicleConfig, make_bicycle_config, make_cart_config

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
            from gm3.diffgm3 import DiffGM3

            return DiffGM3
        raise AttributeError(f"module 'gm3' has no attribute {name!r}")

else:
    __all__ = ["GM3"]
