from __future__ import annotations

from typing import Iterable

import numpy as np

from .types import ControlLike, GM3Control, GM3State, StateLike


def state_from(state: StateLike) -> GM3State:
    if isinstance(state, GM3State):
        return state
    return GM3State.from_array(list(state))


def control_from(control: ControlLike) -> GM3Control:
    if isinstance(control, GM3Control):
        return control
    return GM3Control.from_array(list(control))


def states_to_array(states: Iterable[GM3State]) -> np.ndarray:
    return np.stack([state.as_array() for state in states], axis=0)


def tire_count_by_axle(tire_x: np.ndarray) -> tuple[int, int]:
    front_count = max(int((tire_x > 0.0).sum()), 1)
    rear_count = max(int((tire_x <= 0.0).sum()), 1)
    return front_count, rear_count

