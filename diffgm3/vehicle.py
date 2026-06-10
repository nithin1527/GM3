from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_gm3.diffgm3.tire import body_to_tire_velocities, brush_forces, slip
from new_gm3.diffgm3.torch_utils import bounded, raw_bounded, raw_positive
from new_gm3.shared.constants import ALIGN_GAIN_BOUNDS, CONTACT_LENGTH_BOUNDS, CP_BOUNDS, MU_BOUNDS
from new_gm3.shared.types import VehicleConfig


class DiffGM3Vehicle(nn.Module):
    """Torch vehicle core with trainable GM3 physical parameters."""

    def __init__(self, config: VehicleConfig, dt: float = 0.05):
        super().__init__()
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.config = config
        self.steering_mode = config.steering_mode
        self.can_lean = bool(config.can_lean)
        self.n_state = 8
        self.n_control = 2
        self.n_tires = len(config.tires)
        self._has_leaning_tires = any(tire.can_lean for tire in config.tires)

        dtype = torch.get_default_dtype()

        def buffer(name: str, value: Any, *, bool_tensor: bool = False) -> None:
            tensor_dtype = torch.bool if bool_tensor else dtype
            self.register_buffer(name, torch.as_tensor(value, dtype=tensor_dtype))

        buffer("default_dt", float(dt))
        buffer("mass", config.mass)
        buffer("lf", config.lf)
        buffer("lr", config.lr)
        buffer("wheelbase", config.lf + config.lr)
        buffer("width", config.width)
        buffer("cg_height", config.cg_height)
        buffer("gravity", config.gravity)
        buffer("eps", config.eps)
        buffer("min_normal_load", config.min_normal_load)

        tire_x = [tire.x for tire in config.tires]
        tire_y = [tire.y for tire in config.tires]
        tire_radius = [tire.radius for tire in config.tires]
        buffer("tire_x", tire_x)
        buffer("tire_y", tire_y)
        buffer("tire_radius", tire_radius)
        buffer("tire_x_row", [tire_x])
        buffer("tire_y_row", [tire_y])
        buffer("front_mask", [x > 0.0 for x in tire_x], bool_tensor=True)
        buffer("steerable_mask", [tire.steerable for tire in config.tires], bool_tensor=True)
        buffer("driven_mask", [tire.driven for tire in config.tires], bool_tensor=True)
        buffer("lean_mask", [tire.can_lean for tire in config.tires], bool_tensor=True)
        buffer("lateral_load_mask", [abs(tire.y) > config.eps for tire in config.tires], bool_tensor=True)
        buffer("front_mask_row", [[x > 0.0 for x in tire_x]], bool_tensor=True)
        buffer("steerable_mask_row", [[tire.steerable for tire in config.tires]], bool_tensor=True)
        buffer("driven_mask_row", [[tire.driven for tire in config.tires]], bool_tensor=True)
        buffer("lateral_load_mask_row", [[abs(tire.y) > config.eps for tire in config.tires]], bool_tensor=True)

        front_count = max(sum(1 for x in tire_x if x > 0.0), 1)
        rear_count = max(len(tire_x) - sum(1 for x in tire_x if x > 0.0), 1)
        buffer("front_count", float(front_count))
        buffer("rear_count", float(rear_count))
        width_safe = max(config.width, config.eps)
        y_norm = [tire.y / max(0.5 * width_safe, config.eps) for tire in config.tires]
        buffer("tire_y_norm", y_norm)
        buffer("tire_y_norm_row", [y_norm])

        self.raw_mu = nn.Parameter(torch.stack([raw_bounded(tire.mu, *MU_BOUNDS) for tire in config.tires]))
        self.raw_cp = nn.Parameter(torch.stack([raw_bounded(tire.cp, *CP_BOUNDS) for tire in config.tires]))
        self.raw_contact_length = nn.Parameter(
            torch.stack([raw_bounded(tire.contact_length, *CONTACT_LENGTH_BOUNDS) for tire in config.tires])
        )
        self.raw_yaw_inertia = nn.Parameter(raw_positive(config.yaw_inertia, minimum=1e-6))
        self.raw_roll_inertia = nn.Parameter(raw_positive(config.effective_roll_inertia, minimum=1e-6))
        self.raw_align_gain = nn.Parameter(raw_bounded(config.align_gain, *ALIGN_GAIN_BOUNDS))
        self.raw_yaw_damping = nn.Parameter(raw_positive(config.yaw_damping, minimum=1e-6))
        self.raw_roll_damping = nn.Parameter(raw_positive(config.roll_damping, minimum=1e-6))

    def physical_parameters(self, *, detach: bool = False) -> dict[str, torch.Tensor]:
        params = {
            "mu": bounded(self.raw_mu, *MU_BOUNDS),
            "cp": bounded(self.raw_cp, *CP_BOUNDS),
            "contact_length": bounded(self.raw_contact_length, *CONTACT_LENGTH_BOUNDS),
            "yaw_inertia": F.softplus(self.raw_yaw_inertia) + 1e-6,
            "roll_inertia": F.softplus(self.raw_roll_inertia) + 1e-6,
            "align_gain": bounded(self.raw_align_gain, *ALIGN_GAIN_BOUNDS),
            "yaw_damping": F.softplus(self.raw_yaw_damping) + 1e-6,
            "roll_damping": F.softplus(self.raw_roll_damping) + 1e-6,
        }
        if detach:
            return {name: value.detach() for name, value in params.items()}
        return params

    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        squeeze_state = state.ndim == 1
        if squeeze_state:
            state = state.unsqueeze(0)
        if control.ndim == 1:
            control = control.unsqueeze(0)
        if control.shape[0] == 1 and state.shape[0] > 1:
            control = control.expand(state.shape[0], -1)
        if state.shape[-1] != self.n_state:
            raise ValueError(f"state must have shape [B, {self.n_state}]")
        if control.shape[-1] != self.n_control:
            raise ValueError(f"control must have shape [B, {self.n_control}]")
        if control.shape[0] != state.shape[0]:
            raise ValueError("state and control batch sizes must match")

        params = self.physical_parameters()
        next_state, aux = self._step_with_params(state, control, dt, params=params, compute_aux=return_aux)

        if squeeze_state:
            next_state = next_state.squeeze(0)

        if return_aux:
            return next_state, aux
        return next_state

    def derivative(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        squeeze_state = state.ndim == 1
        if squeeze_state:
            state = state.unsqueeze(0)
        if control.ndim == 1:
            control = control.unsqueeze(0)
        if control.shape[0] == 1 and state.shape[0] > 1:
            control = control.expand(state.shape[0], -1)
        params = self.physical_parameters()
        derivative, aux = self._derivative_and_aux(state, control, params=params, compute_aux=return_aux)
        if squeeze_state:
            derivative = derivative.squeeze(0)
        if return_aux:
            return derivative, aux
        return derivative

    def rollout(self, initial_state: torch.Tensor, controls: torch.Tensor, dt: float | torch.Tensor | None = None) -> torch.Tensor:
        if controls.ndim != 3 or controls.shape[-1] != self.n_control:
            raise ValueError("controls must have shape [T, B, 2]")
        current = initial_state
        if current.ndim == 1:
            current = current.unsqueeze(0)
        states = [current]
        params = self.physical_parameters()
        for t in range(controls.shape[0]):
            current, _ = self._step_with_params(current, controls[t], dt, params=params, compute_aux=False)
            states.append(current)
        return torch.stack(states, dim=0)

    def _step_with_params(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: float | torch.Tensor | None,
        *,
        params: dict[str, torch.Tensor],
        compute_aux: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        dt_tensor = self.default_dt if dt is None else self._dt_like(dt, state)
        derivative, aux = self._derivative_and_aux(state, control, params=params, compute_aux=compute_aux)
        next_state = state + derivative * dt_tensor

        if not self.can_lean:
            next_state = next_state.clone()
            next_state[:, 6:8] = 0.0

        return next_state, aux

    def _derivative_and_aux(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        *,
        params: dict[str, torch.Tensor],
        compute_aux: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        _, _, psi, vx, vy, r, gamma, gamma_dot = state.unbind(dim=-1)
        omega, delta = control.unbind(dim=-1)

        normal_loads = self._normal_loads(vx, vy, r)
        steering_angles = self._steering_angles(delta)
        tire_velocities = body_to_tire_velocities(
            vx=vx,
            vy=vy,
            yaw_rate=r,
            tire_x=self.tire_x,
            steering_angles=steering_angles,
        )
        vx_tire = tire_velocities[..., 0]
        vy_tire = tire_velocities[..., 1]

        free_roll_omega = vx_tire / self.tire_radius.clamp_min(self.eps)
        omega_tire = torch.where(self.driven_mask_row, omega.unsqueeze(-1), free_roll_omega)

        sigma_x, sigma_y, kappa, alpha = slip(
            vx_tire=vx_tire,
            vy_tire=vy_tire,
            omega_tire=omega_tire,
            tire_radius=self.tire_radius,
            eps=self.eps,
        )
        fx_tire, fy_tire, mz_tire = brush_forces(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            normal_loads=normal_loads,
            alpha=alpha,
            steering_angles=steering_angles,
            gamma=gamma,
            tire_radius=self.tire_radius,
            wheelbase=self.wheelbase,
            eps=self.eps,
            min_normal_load=self.min_normal_load,
            lean_mask=self.lean_mask,
            has_leaning_tires=self._has_leaning_tires,
            mu=params["mu"],
            cp=params["cp"],
            contact_length=params["contact_length"],
        )

        cos_delta = torch.cos(steering_angles)
        sin_delta = torch.sin(steering_angles)
        fx_body = fx_tire * cos_delta - fy_tire * sin_delta
        fy_body = fx_tire * sin_delta + fy_tire * cos_delta
        mz_body = params["align_gain"] * mz_tire + self.tire_x_row * fy_body - self.tire_y_row * fx_body

        fx_total = fx_body.sum(dim=-1)
        fy_total = fy_body.sum(dim=-1)
        mz_total = mz_body.sum(dim=-1)

        x_dot = vx * torch.cos(psi) - vy * torch.sin(psi)
        y_dot = vx * torch.sin(psi) + vy * torch.cos(psi)
        psi_dot = r
        ax = fx_total / self.mass + vy * r
        ay = fy_total / self.mass - vx * r
        r_dot = mz_total / params["yaw_inertia"] - params["yaw_damping"] * r

        if self.can_lean:
            roll_moment = (
                self.mass * ay * self.cg_height * torch.cos(gamma)
                - self.mass * self.gravity * self.cg_height * torch.sin(gamma)
            )
            gamma_ddot = (roll_moment - params["roll_damping"] * gamma_dot) / params["roll_inertia"]
            gamma_rate = gamma_dot
        else:
            gamma_ddot = torch.zeros_like(gamma_dot)
            gamma_rate = torch.zeros_like(gamma)

        derivative = torch.stack([x_dot, y_dot, psi_dot, ax, ay, r_dot, gamma_rate, gamma_ddot], dim=-1)
        aux = None
        if compute_aux:
            aux = {
                "normal_loads": normal_loads,
                "steering_angles": steering_angles,
                "tire_forces": torch.stack([fx_tire, fy_tire, mz_tire], dim=-1),
                "body_forces": torch.stack([fx_body, fy_body, mz_body], dim=-1),
                "slip": torch.stack([sigma_x, sigma_y, kappa, alpha], dim=-1),
                "tire_velocities": tire_velocities,
                "force_total": torch.stack([fx_total, fy_total], dim=-1),
                "moment_total": mz_total,
                "physical_parameters": params,
            }
        return derivative, aux

    def _normal_loads(self, vx: torch.Tensor, vy: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        wheelbase = self.wheelbase.clamp_min(self.eps)
        ax_est = vy * r
        ay_est = -vx * r
        longitudinal = self.mass * ax_est * self.cg_height / wheelbase

        front_static = self.mass * self.gravity * self.lr / wheelbase / self.front_count
        rear_static = self.mass * self.gravity * self.lf / wheelbase / self.rear_count

        front_load = front_static - longitudinal.unsqueeze(-1) / self.front_count
        rear_load = rear_static + longitudinal.unsqueeze(-1) / self.rear_count
        loads = torch.where(self.front_mask_row, front_load, rear_load)

        lateral = self.mass * ay_est * self.cg_height / self.width.clamp_min(self.eps)
        lateral_delta = lateral.unsqueeze(-1) * self.tire_y_norm_row
        loads = torch.where(self.lateral_load_mask_row, loads - lateral_delta, loads)
        return self.min_normal_load + F.softplus(loads - self.min_normal_load)

    def _steering_angles(self, delta: torch.Tensor) -> torch.Tensor:
        delta_by_tire = delta.unsqueeze(-1)
        if self.steering_mode == "ackermann":
            wheelbase = self.wheelbase.clamp_min(self.eps)
            tan_delta = torch.tan(delta_by_tire)
            numerator = wheelbase * tan_delta
            denominator = wheelbase - self.tire_y_row * tan_delta
            delta_by_tire = torch.atan2(numerator, denominator)
        else:
            delta_by_tire = delta_by_tire.expand(-1, self.n_tires)
        return torch.where(self.steerable_mask_row, delta_by_tire, torch.zeros_like(delta_by_tire))

    def _dt_like(self, dt: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if isinstance(dt, torch.Tensor):
            return dt.to(device=reference.device, dtype=reference.dtype)
        return reference.new_tensor(float(dt))
