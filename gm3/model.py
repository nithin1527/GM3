from __future__ import annotations

from typing import Any

import numpy as np

from gm3.gm3.tire import body_to_tire_velocity, brush_forces, slip
from gm3.gm3.vehicle import GM3Vehicle
from gm3.shared.types import ControlLike, GM3State, StateLike, VehicleConfig
from gm3.shared.utils import control_from, state_from


class GM3:
    """Pure, deterministic General Micro-Mobility Model implementation."""

    def __init__(self, config: VehicleConfig):
        self.config = config
        self.vehicle = GM3Vehicle(config)

    def derivative(
        self,
        state: StateLike,
        control: ControlLike,
        *,
        return_aux: bool = False,
    ) -> GM3State | tuple[GM3State, dict[str, Any]]:
        s = state_from(state)
        u = control_from(control)
        aux = self._forces_and_aux(s, u)

        cfg = self.config
        x_dot = s.vx * np.cos(s.psi) - s.vy * np.sin(s.psi)
        y_dot = s.vx * np.sin(s.psi) + s.vy * np.cos(s.psi)
        psi_dot = s.r

        ax = aux["force_total"][0] / cfg.mass + s.vy * s.r
        ay = aux["force_total"][1] / cfg.mass - s.vx * s.r
        r_dot = aux["moment_total"] / cfg.yaw_inertia - cfg.yaw_damping * s.r

        if cfg.can_lean:
            roll_moment = (
                cfg.mass * ay * cfg.cg_height * np.cos(s.gamma)
                - cfg.mass * cfg.gravity * cfg.cg_height * np.sin(s.gamma)
            )
            gamma_ddot = (roll_moment - cfg.roll_damping * s.gamma_dot) / cfg.effective_roll_inertia
        else:
            gamma_ddot = 0.0

        deriv = GM3State(
            x=float(x_dot),
            y=float(y_dot),
            psi=float(psi_dot),
            vx=float(ax),
            vy=float(ay),
            r=float(r_dot),
            gamma=float(s.gamma_dot if cfg.can_lean else 0.0),
            gamma_dot=float(gamma_ddot),
        )

        if return_aux:
            aux = dict(aux)
            aux["derivative"] = deriv.as_array()
            return deriv, aux
        return deriv

    def step(
        self,
        state: StateLike,
        control: ControlLike,
        dt: float,
        *,
        return_aux: bool = False,
    ) -> GM3State | tuple[GM3State, dict[str, Any]]:
        if dt <= 0:
            raise ValueError("dt must be positive")

        s = state_from(state)
        if return_aux:
            deriv, aux = self.derivative(s, control, return_aux=True)
        else:
            deriv = self.derivative(s, control)
            aux = None

        next_state = GM3State.from_array(s.as_array() + deriv.as_array() * float(dt))
        if not self.config.can_lean:
            next_state = GM3State(
                next_state.x,
                next_state.y,
                next_state.psi,
                next_state.vx,
                next_state.vy,
                next_state.r,
                0.0,
                0.0,
            )

        if return_aux:
            return next_state, aux
        return next_state

    def rollout(self, initial_state: StateLike, controls: list[ControlLike] | np.ndarray, dt: float) -> list[GM3State]:
        states = [state_from(initial_state)]
        current = states[0]
        for control in controls:
            current = self.step(current, control, dt)
            states.append(current)
        return states

    def _forces_and_aux(self, state: GM3State, control: Any) -> dict[str, Any]:
        vehicle = self.vehicle
        n_tires = len(self.config.tires)

        normal_loads = vehicle.normal_loads(state)
        steering_angles = vehicle.steering_angles(control.delta)

        fx_tire = np.zeros(n_tires, dtype=float)
        fy_tire = np.zeros(n_tires, dtype=float)
        mz_tire = np.zeros(n_tires, dtype=float)
        sigma_x = np.zeros(n_tires, dtype=float)
        sigma_y = np.zeros(n_tires, dtype=float)
        alpha = np.zeros(n_tires, dtype=float)
        kappa = np.zeros(n_tires, dtype=float)
        vx_tire = np.zeros(n_tires, dtype=float)
        vy_tire = np.zeros(n_tires, dtype=float)

        for i in range(n_tires):
            vx_i, vy_i = body_to_tire_velocity(
                vx=state.vx,
                vy=state.vy,
                yaw_rate=state.r,
                tire_x=vehicle.tire_x[i],
                steering_angle=steering_angles[i],
            )
            vx_tire[i] = vx_i
            vy_tire[i] = vy_i
            omega_i = control.omega if vehicle.driven[i] else vx_i / max(vehicle.tire_radius[i], self.config.eps)

            slip_values = slip(
                vx_tire=vx_i,
                vy_tire=vy_i,
                omega=omega_i,
                tire_radius=vehicle.tire_radius[i],
                eps=self.config.eps,
            )
            sigma_x[i] = slip_values["sigma_x"]
            sigma_y[i] = slip_values["sigma_y"]
            alpha[i] = slip_values["alpha"]
            kappa[i] = slip_values["kappa"]

            force_values = brush_forces(
                config=self.config,
                sigma_x=sigma_x[i],
                sigma_y=sigma_y[i],
                normal_load=normal_loads[i],
                alpha=alpha[i],
                steering_angle=steering_angles[i],
                gamma=state.gamma,
                tire_radius=vehicle.tire_radius[i],
                mu=vehicle.tire_mu[i],
                cp=vehicle.tire_cp[i],
                contact_length=vehicle.tire_contact_length[i],
                can_lean=vehicle.can_lean[i],
            )
            fx_tire[i] = force_values["fx"]
            fy_tire[i] = force_values["fy"]
            mz_tire[i] = force_values["mz"]

        aggregate = vehicle.aggregate_body_forces(
            steering_angles=steering_angles,
            fx_tire=fx_tire,
            fy_tire=fy_tire,
            mz_tire=mz_tire,
        )

        return {
            "normal_loads": normal_loads,
            "steering_angles": steering_angles,
            "tire_forces": np.stack([fx_tire, fy_tire, mz_tire], axis=-1),
            "body_forces": aggregate["body_forces"],
            "slip": np.stack([sigma_x, sigma_y, kappa, alpha], axis=-1),
            "tire_velocities": np.stack([vx_tire, vy_tire], axis=-1),
            "force_total": aggregate["force_total"],
            "moment_total": aggregate["moment_total"],
        }


__all__ = ["GM3"]

