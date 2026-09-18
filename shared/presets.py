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


def make_scooter_config(*, motor_on: bool = False) -> VehicleConfig:
    """Return the Hiboy S2 kick-scooter preset used for the scooter logs.

    ``motor_on=False`` (the recorded condition) leaves both wheels free-rolling,
    so ``omega`` is ignored and speed comes from rolling resistance, slope and
    whatever the rider does; ``motor_on=True`` marks the front hub as driven so
    ``omega`` is a drive input the tire tracks through longitudinal slip.

    A Hiboy S2 (350 W front hub motor, 8.5 in tires) with a steering motor on
    the column and two caster wheels just ahead of the rear wheel, so it does
    not lean. It is modelled single-track: the casters self-align and carry
    negligible lateral force, so their only effect, sharing the rear load, is
    absorbed into the rear tire's ``cp * a^2 / Fz`` which the fit owns anyway.

    Geometry: the wheelbase comes from the kinematic yaw gain of the recorded
    runs (``r = vx * tan(delta) / L`` gives 0.95 m with the GPS-calibrated
    wheel-speed scale, the same in every speed and steering-angle band, with
    the measured 3.75 / 10.9 steering gear ratio); the wheel radius is the
    tire size; the rider stands mid-deck.
    Mass and CG height assume a ~70 kg rider on a 14 kg scooter.

    Tires: ``contact_length`` is the brush half-length, 0.025 m for an 8.5 in
    pneumatic tire under ~420 N, and ``cp`` carries the stiffness:
    ``2 * cp * a^2`` = 4,500 N/rad, about 11 x Fz per radian, a typical small
    pneumatic tire. Only that product is identifiable (brush force in the
    linear region is ``2 * cp * a^2 * sigma``), so fits should free ``cp`` and
    leave the geometry alone. Inertia and
    damping are starting points for calibration. ``rolling_resistance`` is the
    coasting deceleration of the recorded runs, 0.13-0.30 m/s^2 with the motor
    off, over g.
    """

    lf = 0.475
    lr = 0.475
    tire_radius = 0.108
    tires = (
        TireConfig(
            x=lf,
            y=0.0,
            radius=tire_radius,
            mu=0.8,
            cp=3_600_000.0,
            contact_length=0.025,
            steerable=True,
            driven=motor_on,
        ),
        TireConfig(
            x=-lr,
            y=0.0,
            radius=tire_radius,
            mu=0.8,
            cp=3_600_000.0,
            contact_length=0.025,
            steerable=False,
            driven=False,
        ),
    )
    return VehicleConfig(
        mass=85.0,
        yaw_inertia=8.0,
        roll_inertia=20.0,
        lf=lf,
        lr=lr,
        width=0.40,
        cg_height=0.90,
        tires=tires,
        can_lean=False,
        align_gain=0.1,
        yaw_damping=0.5,
        roll_damping=15.0,
        steering_mode="direct",
        rolling_resistance=0.02,
    )
