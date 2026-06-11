from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gm3.diffgm3.tire import body_to_tire_velocities, brush_forces, slip
from gm3.diffgm3.torch_utils import bounded, raw_bounded, raw_positive
from gm3.shared.constants import ALIGN_GAIN_BOUNDS, CONTACT_LENGTH_BOUNDS, CP_BOUNDS, MU_BOUNDS
from gm3.shared.types import VehicleConfig


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
        slope: torch.Tensor | None = None,
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
        slope = self._slope_like(slope, state)

        params = self.physical_parameters()
        next_state, aux = self._step_with_params(
            state, control, dt, params=params, compute_aux=return_aux, slope=slope
        )

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
        slope: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        squeeze_state = state.ndim == 1
        if squeeze_state:
            state = state.unsqueeze(0)
        if control.ndim == 1:
            control = control.unsqueeze(0)
        if control.shape[0] == 1 and state.shape[0] > 1:
            control = control.expand(state.shape[0], -1)
        slope = self._slope_like(slope, state)
        params = self.physical_parameters()
        derivative, aux = self._derivative_and_aux(
            state, control, params=params, compute_aux=return_aux, slope=slope
        )
        if squeeze_state:
            derivative = derivative.squeeze(0)
        if return_aux:
            return derivative, aux
        return derivative

    def rollout(
        self,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        slopes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if controls.ndim != 3 or controls.shape[-1] != self.n_control:
            raise ValueError("controls must have shape [T, B, 2]")
        current = initial_state
        if current.ndim == 1:
            current = current.unsqueeze(0)
        if slopes is not None:
            if slopes.shape[-1] != 2:
                raise ValueError("slopes must have 2 values per step [alpha_p, alpha_r]")
            if slopes.ndim == 1:
                slopes = slopes.unsqueeze(0).unsqueeze(0)
            elif slopes.ndim == 2:
                slopes = slopes.unsqueeze(0)
            if slopes.shape[0] == 1:
                slopes = slopes.expand(controls.shape[0], -1, -1)
            if slopes.shape[0] != controls.shape[0]:
                raise ValueError("slopes must match controls in the time dimension")
        states = [current]
        params = self.physical_parameters()
        for t in range(controls.shape[0]):
            slope_t = self._slope_like(None if slopes is None else slopes[t], current)
            current, _ = self._step_with_params(
                current, controls[t], dt, params=params, compute_aux=False, slope=slope_t
            )
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
        slope: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        dt_tensor = self.default_dt if dt is None else self._dt_like(dt, state)
        derivative, aux = self._derivative_and_aux(
            state, control, params=params, compute_aux=compute_aux, slope=slope
        )
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
        slope: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        _, _, psi, vx, vy, r, gamma, gamma_dot = state.unbind(dim=-1)
        omega, delta = control.unbind(dim=-1)

        # Surface-fixed gravity decomposition. alpha_p > 0 means climbing along
        # body +x; alpha_r > 0 means the body +y side is uphill. On flat ground
        # g_long = g_lat = 0 and g_normal = g, recovering the original dynamics.
        if slope is None:
            g_long = torch.zeros_like(vx)
            g_lat = torch.zeros_like(vx)
            g_normal = self.gravity * torch.ones_like(vx)
        else:
            alpha_p, alpha_r = slope.unbind(dim=-1)
            cos_ap = torch.cos(alpha_p)
            g_long = self.gravity * torch.sin(alpha_p)
            g_lat = self.gravity * torch.sin(alpha_r) * cos_ap
            g_normal = self.gravity * cos_ap * torch.cos(alpha_r)

        normal_loads = self._normal_loads(vx, vy, r, g_long=g_long, g_lat=g_lat, g_normal=g_normal)
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
        ax = fx_total / self.mass + vy * r - g_long
        ay = fy_total / self.mass - vx * r - g_lat
        r_dot = mz_total / params["yaw_inertia"] - params["yaw_damping"] * r

        if self.can_lean:
            # ay already carries the lateral gravity component on a banked
            # surface; the restoring term uses the surface-normal gravity, so
            # the equilibrium lean satisfies tan(gamma - alpha_r) ~ v^2 / (R g).
            roll_moment = (
                self.mass * ay * self.cg_height * torch.cos(gamma)
                - self.mass * g_normal * self.cg_height * torch.sin(gamma)
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

    def _normal_loads(
        self,
        vx: torch.Tensor,
        vy: torch.Tensor,
        r: torch.Tensor,
        *,
        g_long: torch.Tensor,
        g_lat: torch.Tensor,
        g_normal: torch.Tensor,
    ) -> torch.Tensor:
        wheelbase = self.wheelbase.clamp_min(self.eps)
        # Down-slope gravity transfers load: climbing (g_long > 0) unloads the
        # front axle, a bank (g_lat > 0, +y uphill) loads the downhill (-y)
        # tires (positive `lateral` subtracts from +y tires below).
        ax_est = vy * r + g_long
        ay_est = -vx * r + g_lat
        longitudinal = self.mass * ax_est * self.cg_height / wheelbase

        front_static = (self.mass * g_normal * self.lr / wheelbase / self.front_count).unsqueeze(-1)
        rear_static = (self.mass * g_normal * self.lf / wheelbase / self.rear_count).unsqueeze(-1)

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

    def _slope_like(self, slope: Any, state: torch.Tensor) -> torch.Tensor | None:
        """Normalize a slope input to [B, 2] (alpha_p, alpha_r) or None."""
        if slope is None:
            return None
        slope = torch.as_tensor(slope, dtype=state.dtype, device=state.device)
        if slope.ndim == 1:
            slope = slope.unsqueeze(0)
        if slope.shape[-1] != 2:
            raise ValueError("slope must have shape [B, 2] (alpha_p, alpha_r)")
        if slope.shape[0] == 1 and state.shape[0] > 1:
            slope = slope.expand(state.shape[0], -1)
        if slope.shape[0] != state.shape[0]:
            raise ValueError("slope and state batch sizes must match")
        return slope
