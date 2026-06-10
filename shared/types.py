from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Union

import numpy as np

from .constants import CONTROL_FIELDS, DEFAULT_EPS, DEFAULT_GRAVITY, DEFAULT_MIN_NORMAL_LOAD, STATE_FIELDS


SteeringMode = Literal["direct", "ackermann"]


@dataclass(frozen=True)
class GM3State:
    """Canonical GM3 state.

    Pose is in the global frame. Velocities are in the vehicle body frame.
    """

    x: float
    y: float
    psi: float
    vx: float
    vy: float
    r: float
    gamma: float = 0.0
    gamma_dot: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in STATE_FIELDS], dtype=float)

    @classmethod
    def from_array(cls, values: Sequence[float]) -> "GM3State":
        if len(values) != len(STATE_FIELDS):
            raise ValueError(f"GM3State requires {len(STATE_FIELDS)} values, got {len(values)}")
        return cls(*(float(v) for v in values))


@dataclass(frozen=True)
class GM3Control:
    """Canonical GM3 control: wheel angular velocity and steering angle."""

    omega: float
    delta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.omega, self.delta], dtype=float)

    @classmethod
    def from_array(cls, values: Sequence[float]) -> "GM3Control":
        if len(values) != len(CONTROL_FIELDS):
            raise ValueError(f"GM3Control requires {len(CONTROL_FIELDS)} values, got {len(values)}")
        return cls(float(values[0]), float(values[1]))


@dataclass(frozen=True)
class TireConfig:
    """Physical tire configuration shared by GM3 and DiffGM3."""

    x: float
    y: float
    radius: float
    mu: float
    cp: float
    contact_length: float
    steerable: bool
    driven: bool
    can_lean: bool = False

    @property
    def R(self) -> float:
        return self.radius

    @property
    def a(self) -> float:
        return self.contact_length


@dataclass(frozen=True)
class VehicleConfig:
    """Vehicle-level configuration shared by GM3 and DiffGM3."""

    mass: float
    yaw_inertia: float
    lf: float
    lr: float
    width: float
    cg_height: float
    tires: tuple[TireConfig, ...]
    roll_inertia: float | None = None
    can_lean: bool = False
    align_gain: float = 0.3
    yaw_damping: float = 2.0
    roll_damping: float = 15.0
    steering_mode: SteeringMode = "ackermann"
    gravity: float = DEFAULT_GRAVITY
    min_normal_load: float = DEFAULT_MIN_NORMAL_LOAD
    eps: float = DEFAULT_EPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "tires", tuple(self.tires))
        if not self.tires:
            raise ValueError("VehicleConfig requires at least one tire")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.yaw_inertia <= 0.0:
            raise ValueError("yaw_inertia must be positive")
        if self.roll_inertia is not None and self.roll_inertia <= 0.0:
            raise ValueError("roll_inertia must be positive when provided")
        if self.lf < 0.0 or self.lr < 0.0 or self.wheelbase <= 0.0:
            raise ValueError("lf/lr must define a positive wheelbase")
        if self.width < 0.0:
            raise ValueError("width must be non-negative")
        if self.cg_height < 0.0:
            raise ValueError("cg_height must be non-negative")
        if self.steering_mode not in ("direct", "ackermann"):
            raise ValueError("steering_mode must be 'direct' or 'ackermann'")
        for tire in self.tires:
            if tire.radius <= 0.0:
                raise ValueError("tire radius must be positive")
            if tire.mu <= 0.0 or tire.cp <= 0.0 or tire.contact_length <= 0.0:
                raise ValueError("tire mu, cp, and contact_length must be positive")

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @property
    def effective_roll_inertia(self) -> float:
        if self.roll_inertia is not None:
            return self.roll_inertia
        return max(self.mass * self.cg_height * self.cg_height, self.eps)


StateLike = Union[GM3State, Sequence[float], np.ndarray]
ControlLike = Union[GM3Control, Sequence[float], np.ndarray]

