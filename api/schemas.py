"""Pydantic request/response schemas for the DiffGM3 API.

These mirror gm3.shared.types.VehicleConfig / TireConfig without touching them.
State layout: [x, y, psi, vx, vy, r, gamma, gamma_dot]
Control layout: [omega, delta]
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from gm3.shared.constants import DEFAULT_EPS, DEFAULT_GRAVITY, DEFAULT_MIN_NORMAL_LOAD

N_STATE = 8
N_CONTROL = 2
N_SLOPE = 2

PresetName = Literal["bicycle", "cart"]


class TireConfigModel(BaseModel):
    x: float
    y: float
    radius: float = Field(gt=0.0)
    mu: float = Field(gt=0.0)
    cp: float = Field(gt=0.0)
    contact_length: float = Field(gt=0.0)
    steerable: bool
    driven: bool
    can_lean: bool = False


class VehicleConfigModel(BaseModel):
    mass: float = Field(gt=0.0)
    yaw_inertia: float = Field(gt=0.0)
    lf: float = Field(ge=0.0)
    lr: float = Field(ge=0.0)
    width: float = Field(ge=0.0)
    cg_height: float = Field(ge=0.0)
    tires: list[TireConfigModel] = Field(min_length=1)
    roll_inertia: float | None = Field(default=None, gt=0.0)
    can_lean: bool = False
    align_gain: float = 0.3
    yaw_damping: float = 2.0
    roll_damping: float = 15.0
    steering_mode: Literal["direct", "ackermann"] = "ackermann"
    gravity: float = DEFAULT_GRAVITY
    min_normal_load: float = DEFAULT_MIN_NORMAL_LOAD
    eps: float = DEFAULT_EPS


class ModelSpec(BaseModel):
    """Selects the vehicle: either a named preset or a full custom config."""

    preset: PresetName | None = None
    config: VehicleConfigModel | None = None
    dt: float = Field(default=0.02, gt=0.0)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ModelSpec":
        if (self.preset is None) == (self.config is None):
            raise ValueError("provide exactly one of 'preset' or 'config'")
        return self


class StepRequest(BaseModel):
    model: ModelSpec
    state: list[float] = Field(min_length=N_STATE, max_length=N_STATE)
    control: list[float] = Field(min_length=N_CONTROL, max_length=N_CONTROL)
    slope: list[float] | None = Field(
        default=None,
        min_length=N_SLOPE,
        max_length=N_SLOPE,
        description="[alpha_p, alpha_r] surface angles in radians; omit for flat ground",
    )
    dt: float | None = Field(default=None, gt=0.0)
    return_aux: bool = False


class StepResponse(BaseModel):
    state: list[float]
    aux: dict[str, Any] | None = None


class RolloutRequest(BaseModel):
    model: ModelSpec
    initial_state: list[float] = Field(min_length=N_STATE, max_length=N_STATE)
    controls: list[list[float]] = Field(min_length=1)
    slopes: list[list[float]] | None = Field(
        default=None,
        description="[alpha_p, alpha_r] per step (same length as controls) or a single constant pair",
    )
    dt: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _control_shape(self) -> "RolloutRequest":
        for row in self.controls:
            if len(row) != N_CONTROL:
                raise ValueError(f"each control must have {N_CONTROL} values [omega, delta]")
        if self.slopes is not None:
            if len(self.slopes) not in (1, len(self.controls)):
                raise ValueError("slopes must have length 1 or match controls")
            for row in self.slopes:
                if len(row) != N_SLOPE:
                    raise ValueError(f"each slope must have {N_SLOPE} values [alpha_p, alpha_r]")
        return self


class RolloutResponse(BaseModel):
    states: list[list[float]]


class PresetsResponse(BaseModel):
    presets: dict[str, VehicleConfigModel]


class HealthResponse(BaseModel):
    status: str
    torch_version: str
    state_fields: list[str]
    control_fields: list[str]
