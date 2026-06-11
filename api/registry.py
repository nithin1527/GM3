"""Builds and caches DiffGM3 instances keyed by their config, for inference."""

from __future__ import annotations

from collections import OrderedDict

import torch

from gm3.diffgm3 import DiffGM3
from gm3.shared import TireConfig, VehicleConfig, make_bicycle_config, make_cart_config

from gm3.api.schemas import ModelSpec, TireConfigModel, VehicleConfigModel

PRESET_BUILDERS = {
    "bicycle": make_bicycle_config,
    "cart": make_cart_config,
}


def vehicle_config_from_model(model: VehicleConfigModel) -> VehicleConfig:
    tires = tuple(
        TireConfig(
            x=t.x,
            y=t.y,
            radius=t.radius,
            mu=t.mu,
            cp=t.cp,
            contact_length=t.contact_length,
            steerable=t.steerable,
            driven=t.driven,
            can_lean=t.can_lean,
        )
        for t in model.tires
    )
    return VehicleConfig(
        mass=model.mass,
        yaw_inertia=model.yaw_inertia,
        lf=model.lf,
        lr=model.lr,
        width=model.width,
        cg_height=model.cg_height,
        tires=tires,
        roll_inertia=model.roll_inertia,
        can_lean=model.can_lean,
        align_gain=model.align_gain,
        yaw_damping=model.yaw_damping,
        roll_damping=model.roll_damping,
        steering_mode=model.steering_mode,
        gravity=model.gravity,
        min_normal_load=model.min_normal_load,
        eps=model.eps,
    )


def vehicle_config_to_model(config: VehicleConfig) -> VehicleConfigModel:
    return VehicleConfigModel(
        mass=config.mass,
        yaw_inertia=config.yaw_inertia,
        lf=config.lf,
        lr=config.lr,
        width=config.width,
        cg_height=config.cg_height,
        tires=[
            TireConfigModel(
                x=t.x,
                y=t.y,
                radius=t.radius,
                mu=t.mu,
                cp=t.cp,
                contact_length=t.contact_length,
                steerable=t.steerable,
                driven=t.driven,
                can_lean=t.can_lean,
            )
            for t in config.tires
        ],
        roll_inertia=config.roll_inertia,
        can_lean=config.can_lean,
        align_gain=config.align_gain,
        yaw_damping=config.yaw_damping,
        roll_damping=config.roll_damping,
        steering_mode=config.steering_mode,
        gravity=config.gravity,
        min_normal_load=config.min_normal_load,
        eps=config.eps,
    )


def resolve_vehicle_config(spec: ModelSpec) -> VehicleConfig:
    if spec.preset is not None:
        return PRESET_BUILDERS[spec.preset]()
    assert spec.config is not None
    return vehicle_config_from_model(spec.config)


class ModelRegistry:
    """LRU cache of DiffGM3 instances so repeated requests reuse one module."""

    def __init__(self, max_size: int = 16):
        self._max_size = max_size
        self._cache: OrderedDict[str, DiffGM3] = OrderedDict()

    def get(self, spec: ModelSpec) -> DiffGM3:
        key = spec.model_dump_json()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        config = resolve_vehicle_config(spec)
        model = DiffGM3(config, dt=spec.dt)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self._cache[key] = model
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return model


def to_jsonable(value):
    """Recursively convert tensors (e.g. step aux output) to plain lists."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
