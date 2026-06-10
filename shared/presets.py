from __future__ import annotations

from .types import TireConfig, VehicleConfig


def make_bicycle_config() -> VehicleConfig:
    """Return a two-wheel front-steer/rear-drive micro-mobility preset."""

    lf = 0.45
    lr = 0.55
    tire_radius = 0.35
    tires = (
        TireConfig(
            x=lf,
            y=0.0,
            radius=tire_radius,
            mu=0.9,
            cp=12_500.0,
            contact_length=0.03,
            steerable=True,
            driven=False,
            can_lean=True,
        ),
        TireConfig(
            x=-lr,
            y=0.0,
            radius=tire_radius,
            mu=0.9,
            cp=12_500.0,
            contact_length=0.03,
            steerable=False,
            driven=True,
            can_lean=True,
        ),
    )
    return VehicleConfig(
        mass=80.0,
        yaw_inertia=10.0,
        roll_inertia=20.0,
        lf=lf,
        lr=lr,
        width=0.5,
        cg_height=1.0,
        tires=tires,
        can_lean=True,
        align_gain=0.1,
        yaw_damping=0.5,
        roll_damping=15.0,
        steering_mode="ackermann",
    )


def make_cart_config() -> VehicleConfig:
    """Return a four-wheel front-steer/rear-drive cart preset."""

    lf = 1.2
    lr = 1.5
    half_width = 0.75
    tire_radius = 0.30
    tires = (
        TireConfig(
            x=lf,
            y=half_width,
            radius=tire_radius,
            mu=0.9,
            cp=15_000.0,
            contact_length=0.15,
            steerable=True,
            driven=False,
        ),
        TireConfig(
            x=lf,
            y=-half_width,
            radius=tire_radius,
            mu=0.9,
            cp=15_000.0,
            contact_length=0.15,
            steerable=True,
            driven=False,
        ),
        TireConfig(
            x=-lr,
            y=half_width,
            radius=tire_radius,
            mu=0.9,
            cp=15_000.0,
            contact_length=0.15,
            steerable=False,
            driven=True,
        ),
        TireConfig(
            x=-lr,
            y=-half_width,
            radius=tire_radius,
            mu=0.9,
            cp=15_000.0,
            contact_length=0.15,
            steerable=False,
            driven=True,
        ),
    )
    return VehicleConfig(
        mass=1_500.0,
        yaw_inertia=2_500.0,
        roll_inertia=500.0,
        lf=lf,
        lr=lr,
        width=2.0 * half_width,
        cg_height=0.5,
        tires=tires,
        can_lean=False,
        align_gain=0.2,
        yaw_damping=0.5,
        roll_damping=20.0,
        steering_mode="ackermann",
    )

