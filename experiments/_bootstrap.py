"""Make ``import gm3`` work when this repo is checked out under any folder name.

The package imports itself as ``gm3`` (e.g. ``from gm3.shared import ...``), so
the repo directory has to be importable under that name. If it already is,
nothing happens; otherwise the repo root is registered as the ``gm3`` package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install() -> None:
    if "gm3" in sys.modules:
        return
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))
    try:
        import gm3  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    spec = importlib.util.spec_from_file_location(
        "gm3", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not register {ROOT} as the 'gm3' package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gm3"] = module
    spec.loader.exec_module(module)


install()
