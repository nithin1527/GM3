from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Union

import numpy as np

from .constants import CONTROL_FIELDS, DEFAULT_EPS, DEFAULT_GRAVITY, DEFAULT_MIN_NORMAL_LOAD, STATE_FIELDS


SteeringMode = Literal["direct", "ackermann"]
DriveMode = Literal["single", "independent"]


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
    """Canonical GM3 control: wheel angular velocity and steering angle.

    ``omega`` is one speed shared by every driven tire. A vehicle configured
    with ``drive_mode="independent"`` instead takes one speed per driven tire,
    in tire order, so the flat layout widens to ``[omega_0, ..., omega_k, delta]``.
    """

    omega: float | tuple[float, ...]
    delta: float

    def __post_init__(self) -> None:
        if np.ndim(self.omega) == 0:
            object.__setattr__(self, "omega", float(self.omega))
        else:
            object.__setattr__(self, "omega", tuple(float(w) for w in self.omega))
            if not self.omega:
                raise ValueError("omega must hold at least one wheel speed")

    @property
    def omegas(self) -> tuple[float, ...]:
        """Wheel speeds exactly as supplied, without broadcasting."""
        return (self.omega,) if isinstance(self.omega, float) else self.omega

    def wheel_omegas(self, count: int) -> tuple[float, ...]:
        """Broadcast to one speed per drive input, sharing a single speed."""
        omegas = self.omegas
        if len(omegas) == 1:
            return omegas * count
        if len(omegas) != count:
            raise ValueError(f"GM3Control carries {len(omegas)} wheel speeds, expected {count}")
        return omegas

    def as_array(self) -> np.ndarray:
        return np.array([*self.omegas, self.delta], dtype=float)

    @classmethod
    def from_array(cls, values: Sequence[float]) -> "GM3Control":
        if len(values) < len(CONTROL_FIELDS):
            raise ValueError(f"GM3Control requires at least {len(CONTROL_FIELDS)} values, got {len(values)}")
        omegas = tuple(float(v) for v in values[:-1])
        return cls(omegas[0] if len(omegas) == 1 else omegas, float(values[-1]))


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
    drive_mode: DriveMode = "single"
    gravity: float = DEFAULT_GRAVITY
    min_normal_load: float = DEFAULT_MIN_NORMAL_LOAD
    eps: float = DEFAULT_EPS
    #: Rolling-resistance coefficient: each tire pushes back ``-c * Fz`` along
    #: its rolling direction. Zero (the default) leaves the original dynamics.
    rolling_resistance: float = 0.0

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
        if self.rolling_resistance < 0.0:
            raise ValueError("rolling_resistance must be non-negative")
        if self.steering_mode not in ("direct", "ackermann"):
            raise ValueError("steering_mode must be 'direct' or 'ackermann'")
        if self.drive_mode not in ("single", "independent"):
            raise ValueError("drive_mode must be 'single' or 'independent'")
        if self.drive_mode == "independent" and self.driven_count == 0:
            raise ValueError("drive_mode 'independent' requires at least one driven tire")
        for tire in self.tires:
            if tire.radius <= 0.0:
                raise ValueError("tire radius must be positive")
            if tire.mu <= 0.0 or tire.cp <= 0.0 or tire.contact_length <= 0.0:
                raise ValueError("tire mu, cp, and contact_length must be positive")

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @property
    def driven_count(self) -> int:
        return sum(1 for tire in self.tires if tire.driven)

    @property
    def n_control(self) -> int:
        """Control width: 2 for a shared drive, 1 + driven_count for an independent one."""
        return 2 if self.drive_mode == "single" else 1 + self.driven_count

    @property
    def driven_slots(self) -> tuple[int, ...]:
        """Per-tire index into the control's omega block; 0 for a shared drive.

        Skid-steer vehicles steer by driving each side at a different speed, so
        every driven tire needs its own slot. Free-rolling tires get slot 0 and
        are masked out downstream.
        """
        if self.drive_mode == "single":
            return (0,) * len(self.tires)
        slots: list[int] = []
        rank = 0
        for tire in self.tires:
            slots.append(rank if tire.driven else 0)
            rank += int(tire.driven)
        return tuple(slots)

    @property
    def effective_roll_inertia(self) -> float:
        if self.roll_inertia is not None:
            return self.roll_inertia
        return max(self.mass * self.cg_height * self.cg_height, self.eps)


StateLike = Union[GM3State, Sequence[float], np.ndarray]
ControlLike = Union[GM3Control, Sequence[float], np.ndarray]

