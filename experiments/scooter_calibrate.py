"""Fit DiffGM3 to the Hiboy scooter logs.

    python -m experiments.scooter_calibrate
    python -m experiments.scooter_calibrate --per-run --per-speed
    python -m experiments.scooter_calibrate --fit contact_length mu yaw_inertia yaw_damping align_gain
    python -m experiments.scooter_calibrate --open-loop --holdout 20260911_163418

Known and held fixed
--------------------
The steering gear ratio (3.75 cm pinion on a 10.9 cm gear), the 8.5 in wheel
radius, the wheelbase (0.89 m, from the kinematic yaw gain of every run with
that ratio), the 0.5 wheel-speed scale and each run's steering-encoder zero.
See ``gm3.dataset.scooter`` for how each was established.

Tuned
-----
Tire lateral stiffness (``cp``, with the brush half-length ``contact_length``
held at its physical 0.025 m; brush force in the linear region is
``2 cp a^2 sigma`` so only that product is identifiable), ``mu``,
``yaw_inertia``, ``yaw_damping``,
``align_gain``, a residual global steering zero, and, as a check on the
measured ratio rather than as a parameter, ``steer_ratio``.

Formulation
-----------
The hub motor was off, so the wheel encoder is a ground-speed sensor: the
longitudinal channel is set by the rider's kicks and rolling drag, neither of
which GM3 models. The rollout is therefore hybrid: psi,
vy and r integrate through the model while vx is pinned to measurement each
step, and the loss matches the yaw-rate and heading trajectories,
variance-normalized. ``--open-loop`` integrates vx too (the tire then has to
reproduce the wheel speed through longitudinal slip) for the strict version.

Gates
-----
1. held-out run loss must fall;
2. per-run fits must agree (``--per-run``);
3. per-speed-tercile fits must agree (``--per-speed``);
4. the fitted model must reproduce the measured steady-state yaw gain.
The bar is a kinematic bicycle ``r = v tan(delta) / L``: on these logs it
already explains 98-99% of yaw-rate variance, so the tire model only earns
its parameters in what is left.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments import _bootstrap  # noqa: F401  (registers the `gm3` package)

from gm3.dataset.scooter import (
    STEER_RATIO, STEER_SIGN, RolloutWindows, ScooterRun, _moving_average, gps_errors, kinematic_steer_fit,
    load_all, rigid_align_rms, windows_of,
)
from gm3.diffgm3 import DiffGM3
from gm3.shared import make_scooter_config

TRAINABLE = {
    "contact_length": "gm3.raw_contact_length", "cp": "gm3.raw_cp", "mu": "gm3.raw_mu",
    "yaw_inertia": "gm3.raw_yaw_inertia", "yaw_damping": "gm3.raw_yaw_damping",
    "align_gain": "gm3.raw_align_gain", "steer_zero": "steer_zero", "steer_ratio": "log_steer_ratio",
    # --tire-model other than brush (gm3.diffgm3.tire_models). "cy" is the cornering
    # stiffness of fiala / dugoff / pacejka; Burckhardt has none, its zero-slip slope
    # is (c1 c2 - c3) Fz, so "burckhardt_c2" frees the exponent alone (the analogue of
    # freeing stiffness with friction held) and "burckhardt_y" all of (c1, c2, c3).
    "cx": "gm3.tire_model.raw_cx", "cy": "gm3.tire_model.raw_cy",
    "shape_y": "gm3.tire_model.raw_shape_y", "curvature_y": "gm3.tire_model.raw_curvature_y",
    "burckhardt_c2": "gm3.tire_model.log_coefficients_y", "burckhardt_y": "gm3.tire_model.log_coefficients_y",
}
GRADIENT_MASKS = {"burckhardt_c2": (0.0, 1.0, 0.0)}
TIRE_FIT = {"brush": "cp", "fiala": "cy", "dugoff": "cy", "pacejka": "cy", "burckhardt": "burckhardt_c2"}
DEFAULT_FIT = ["cp", "yaw_inertia", "yaw_damping", "steer_zero"]
OUT = Path(__file__).resolve().parent / "out"


class ScooterModel(nn.Module):
    """DiffGM3 plus the steering-input map, so the zero and ratio are trainable."""

    def __init__(self, dt: float, steer_ratio: float = STEER_RATIO, motor_on: bool = False,
                 tire_model: str = "brush", **tire_options):
        super().__init__()
        self.gm3 = DiffGM3(make_scooter_config(motor_on=motor_on), dt=dt, tire_model=tire_model, **tire_options)
        self.tire_model = self.gm3.tire_model.name
        self.steer_zero = nn.Parameter(torch.tensor(0.0))
        self.log_steer_ratio = nn.Parameter(torch.log(torch.tensor(steer_ratio)))

    @property
    def steer_ratio(self) -> torch.Tensor:
        return self.log_steer_ratio.exp()

    def controls(self, raw: torch.Tensor) -> torch.Tensor:
        """``[omega, theta_motor]`` -> ``[omega, delta]``."""
        delta = STEER_SIGN * self.steer_ratio * (raw[..., 1] - self.steer_zero)
        return torch.stack([raw[..., 0], delta], dim=-1)

    def stable_substeps(self, v_min: float) -> int:
        """Euler substeps that keep the stiffest tire mode stable.

        The yaw and sideslip modes of a single-track vehicle relax at
        ``sum(C_i x_i^2) / (Jz v)`` and ``sum(C_i) / (m v)``; with a realistic
        4,500 N/rad tire that is ~150/s at 1.5 m/s and ~450/s at 0.5 m/s, well
        past explicit Euler's ``dt * lambda < 2`` at 30 Hz. Substep so that
        ``h * lambda <= 1`` for the current parameters and the slowest sample.
        """
        values = self.gm3.physical_parameters(detach=True)
        if self.tire_model == "brush":
            stiffness = 2.0 * values["cp"] * values["contact_length"] ** 2
        else:
            stiffness = self.gm3.cornering_stiffness().detach()
        x = torch.tensor([tire.x for tire in self.gm3.config.tires], device=stiffness.device)
        rate = max(float((stiffness * x ** 2).sum() / values["yaw_inertia"]),
                   float(stiffness.sum()) / self.gm3.config.mass) / max(v_min, 0.2)
        return int(min(max(math.ceil(rate * float(self.gm3.default_dt)), 1), 200))

    @property
    def device(self) -> torch.device:
        return self.steer_zero.device

    def simulate(self, windows: RolloutWindows, *, pin_vx: bool, keep=None,
                 substeps: int | None = None, use_slope: bool = True) -> torch.Tensor:
        device = self.device
        target = torch.as_tensor(windows.target_states).to(device)
        raw = torch.as_tensor(windows.controls).to(device)
        if keep is not None:
            target, raw = target[:, keep], raw[:, keep]
        controls = self.controls(raw)
        slopes = torch.as_tensor(windows.slopes).to(device)
        if not use_slope:
            slopes = torch.zeros_like(slopes)
        if keep is not None:
            slopes = slopes[:, keep]
        if substeps is None:
            substeps = self.stable_substeps(float(target[..., 3].min()))
        h = self.gm3.default_dt / substeps
        state = target[0].clone()
        trace = [state]
        for t in range(controls.shape[0]):
            for _ in range(substeps):
                state = state + self.gm3.derivative(state, controls[t], slope=slopes[t]) * h
            if pin_vx:
                state = torch.cat([state[:, :3], target[t + 1, :, 3:4], state[:, 4:]], dim=1)
            trace.append(state)
        return torch.stack(trace)

    def values(self) -> dict[str, float]:
        physical = self.gm3.physical_parameters(detach=True)
        out = {}
        for name, value in physical.items():
            if not isinstance(value, torch.Tensor):
                out[name] = float(value)
            elif "coefficients" in name:
                out.update({f"{name}.c{k + 1}": float(c) for k, c in enumerate(value)})
            else:
                out[name] = float(value.flatten()[0])
        out["cp_rear"] = float(physical["cp"][1])
        out["contact_length_rear"] = float(physical["contact_length"][1])
        # Zero-slip cornering stiffness: 2 cp a^2 for the brush (the adhesion term the
        # brush fits have always reported), read off the law itself for the others.
        if self.tire_model == "brush":
            stiffness = 2.0 * physical["cp"] * physical["contact_length"] ** 2
        else:
            stiffness = self.gm3.cornering_stiffness().detach()
        out["C_alpha"], out["C_alpha_rear"] = float(stiffness[0]), float(stiffness[1])
        out["steer_zero_deg"] = math.degrees(float(self.steer_zero.detach()))
        out["steer_ratio"] = float(self.steer_ratio.detach())
        return out


def yaw_loss(simulated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    psi_meas, r_meas = target[..., 2], target[..., 5]
    sigma_r = r_meas.std().clamp_min(1e-6)
    sigma_psi = (psi_meas - psi_meas[0]).std().clamp_min(1e-6)
    return (((simulated[..., 5] - r_meas) / sigma_r) ** 2).mean() + \
        (((simulated[..., 2] - psi_meas) / sigma_psi) ** 2).mean()


def loss_on(model: ScooterModel, windows: RolloutWindows, pin_vx: bool, keep=None,
            substeps: int | None = None) -> torch.Tensor:
    target = torch.as_tensor(windows.target_states).to(model.device)
    if keep is not None:
        target = target[:, keep]
    return yaw_loss(model.simulate(windows, pin_vx=pin_vx, keep=keep, substeps=substeps), target)


TIRE_PARAMETERS = ("gm3.raw_cp", "gm3.raw_contact_length", "gm3.raw_mu",
                   "gm3.tire_model.raw_cx", "gm3.tire_model.raw_cy",
                   "gm3.tire_model.raw_shape_y", "gm3.tire_model.raw_curvature_y")


def fit(model: ScooterModel, windows: RolloutWindows, names, steps: int, lr: float, *,
        pin_vx: bool, keep=None, quiet: bool = False, tie_tires: bool = True) -> ScooterModel:
    """Adam on the freed parameters.

    ``tie_tires`` gives the front and rear tire one shared value: the split
    between them is only visible through the understeer gradient, which on
    these logs is indistinguishable from zero, so freeing it lets the
    optimizer wander into oversteer configurations that fit nothing better.
    The tie is applied through the gradient (both entries get their sum), so
    Adam moves them identically from an identical start.

    The substep count is frozen for the whole fit (sized with 3x headroom on
    the current stiffness) so the loss is a smooth function of the
    parameters; a per-call count would jump as stiffness changes.
    """
    freed = {TRAINABLE[n] for n in names}
    masks = {TRAINABLE[n]: torch.tensor(GRADIENT_MASKS[n]) for n in names if n in GRADIENT_MASKS}
    missing = freed - {name for name, _ in model.named_parameters()}
    if missing:
        raise ValueError(f"the {model.tire_model!r} tire has no parameter {sorted(missing)}")
    target = windows.target_states if keep is None else windows.target_states[:, keep]
    substeps = 3 * model.stable_substeps(float(target[..., 3].min()))
    groups = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in freed)
        if name in freed:
            # The steering zero is in motor radians and the ratio in log space,
            # so Adam's per-step move of ~lr would be 3 deg / 5% at lr = 0.05.
            scale = {"steer_zero": 0.02, "log_steer_ratio": 0.1}.get(name, 1.0)
            groups.append({"params": [parameter], "lr": lr * scale})
    optimizer = torch.optim.Adam(groups)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_on(model, windows, pin_vx, keep, substeps=substeps)
        loss.backward()
        if tie_tires:
            for name, parameter in model.named_parameters():
                if name in TIRE_PARAMETERS and parameter.grad is not None:
                    parameter.grad[:] = parameter.grad.sum()
        for name, parameter in model.named_parameters():
            if name in masks and parameter.grad is not None:
                parameter.grad.mul_(masks[name])
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
        optimizer.step()
        if not quiet and (step % max(steps // 5, 1) == 0 or step == steps - 1):
            v = model.values()
            print(f"    step {step:4d}  loss={loss.item():.4e}  cp={v['cp']:10.3e}  C_alpha={cornering_stiffness(v):7.0f} N/rad  "
                  f"Jz={v['yaw_inertia']:.3f}  c_r={v['yaw_damping']:.3f}  zero={v['steer_zero_deg']:+.2f} deg")
    return model


def cornering_stiffness(values: dict[str, float]) -> float:
    return values["C_alpha"]


def kinematic_trace(windows: RolloutWindows, wheelbase: float, ratio: float = STEER_RATIO,
                    lr: float | None = None, hold_speed: bool = False) -> np.ndarray:
    """Kinematic bicycle on the measured speed: ``r = vx tan(delta) / L``.

    Propagated at the centre of gravity, so it carries the no-slip sideslip
    ``vy = lr * r`` that follows from a rear wheel that rolls straight. Without
    that term the trace is the rear-axle path while both the measurement and
    GM3 are at the CG, which shows up as a spurious position error of order
    ``lr * r * t`` (0.1-0.3 m over a 3 s corner) charged to the baseline.
    """
    lr = make_scooter_config().lr if lr is None else lr
    target = windows.target_states
    horizon = windows.controls.shape[0]
    trace = np.repeat(target[0:1], horizon + 1, axis=0).copy()
    for t in range(horizon):
        vx = target[0, :, 3] if hold_speed else target[t, :, 3]
        delta = STEER_SIGN * ratio * windows.controls[t, :, 1]
        r = vx * np.tan(delta) / wheelbase
        vy = lr * r
        psi = trace[t, :, 2]
        trace[t + 1, :, 0] = trace[t, :, 0] + (vx * np.cos(psi) - vy * np.sin(psi)) * windows.dt
        trace[t + 1, :, 1] = trace[t, :, 1] + (vx * np.sin(psi) + vy * np.cos(psi)) * windows.dt
        trace[t + 1, :, 2] = psi + r * windows.dt
        trace[t + 1, :, 3] = vx if hold_speed else target[t + 1, :, 3]
        trace[t + 1, :, 4] = vy
        trace[t + 1, :, 5] = r
    return trace


def errors(trace: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Yaw-rate, heading and position error over a ``[T+1, B, 8]`` pair.

    ADE averages the displacement over every step of a window and then over
    windows; FDE is the endpoint alone. Read both with the reference in mind:
    the position truth here is the wheel-plus-gyro dead-reckoned path, because
    GPS is 1 Hz with a 3 m sigma and cannot resolve a 3 s window. That
    reference consumes the same wheel speed the models do, so ADE mostly
    scores heading and sideslip divergence rather than distance travelled, and
    it credits the dead-reckoning assumption of zero sideslip. The whole-run
    GPS comparison in ``gps_rms`` is the independent position check.
    """
    displacement = np.linalg.norm(trace[..., :2] - target[..., :2], axis=-1)
    r_err = np.degrees(trace[1:, :, 5] - target[1:, :, 5])
    psi_err = np.degrees(np.abs(trace[-1, :, 2] - target[-1, :, 2]))
    v_err = trace[1:, :, 3] - target[1:, :, 3]
    return {"r_rmse_dps": float(np.sqrt((r_err ** 2).mean())),
            "psi_end_deg": float(np.median(psi_err)),
            "psi_end_p90_deg": float(np.percentile(psi_err, 90)),
            "v_rmse_mps": float(np.sqrt((v_err ** 2).mean())),
            "ADE_m": float(displacement[1:].mean()),
            "ADE_med_m": float(np.median(displacement[1:].mean(axis=0))),
            "FDE_m": float(np.median(displacement[-1]))}


def whole_run(run: ScooterRun) -> RolloutWindows:
    return whole_runs([run])[0]


def whole_runs(runs: list[ScooterRun]) -> tuple[RolloutWindows, np.ndarray]:
    """Every run as one batch, padded by holding the last sample, so a single
    rollout covers them all. Returns the windows and each run's length."""
    lengths = np.array([len(run) for run in runs])
    horizon = int(lengths.max()) - 1
    controls = np.zeros((horizon, len(runs), 2))
    slopes = np.zeros((horizon, len(runs), 2))
    targets = np.zeros((horizon + 1, len(runs), 8))
    for b, run in enumerate(runs):
        n = len(run)
        controls[: n - 1, b] = run.controls[:-1]
        controls[n - 1:, b] = run.controls[-1]
        slopes[: n - 1, b] = run.slopes[:-1]
        slopes[n - 1:, b] = run.slopes[-1]
        targets[:n, b] = run.states
        targets[n:, b] = run.states[-1]
    windows = RolloutWindows(surface="scooter", dt=runs[0].dt, initial_state=targets[0].copy(),
                             controls=controls, slopes=slopes, target_states=targets)
    return windows, lengths


def gps_rms(model: ScooterModel | None, run: ScooterRun, wheelbase: float) -> float:
    return gps_table({"m": model}, [run], wheelbase)["m"][0]["RMS"]


def gps_table(models: dict[str, ScooterModel | None], runs: list[ScooterRun], wheelbase: float,
              open_loop: tuple[str, ...] = (), no_slope: tuple[str, ...] = ()) -> dict[str, list[dict[str, float]]]:
    """Whole-run path error against the GPS fixes for each model on each run.

    ``None`` stands for the kinematic bicycle. Runs are rolled out as one
    batch. Every path is rigidly aligned to the fixes before scoring, because
    the initial heading is unobservable (the IMU yaw is not magnetometer
    referenced) and dead reckoning has no absolute origin.
    """
    windows, lengths = whole_runs(runs)
    out: dict[str, list[dict[str, float]]] = {}
    for name, model in models.items():
        if model is None:
            trace = kinematic_trace(windows, wheelbase)
        else:
            with torch.no_grad():
                trace = model.simulate(windows, pin_vx=name not in open_loop, use_slope=name not in no_slope).cpu().numpy()
        out[name] = [gps_errors(trace[:n, b, :2], run.t, run) for b, (run, n) in enumerate(zip(runs, lengths))]
    return out


def print_gps_table(models: dict[str, ScooterModel | None], runs: list[ScooterRun], wheelbase: float,
                    holdout: str, open_loop: tuple[str, ...] = (), no_slope: tuple[str, ...] = ()) -> dict[str, list[dict[str, float]]]:
    table = gps_table(models, runs, wheelbase, open_loop, no_slope)
    names = list(table)
    print(f"\n  whole-run path against GPS fixes after rigid alignment, ADE / Frechet in m (* = held out):")
    print(f"    {'run':10s} {'fixes':>5s}  " + "  ".join(f"{n:>22s}" for n in names))
    for b, run in enumerate(runs):
        star = "*" if run.name == holdout else " "
        cells = "  ".join(f"{table[n][b]['ADE']:10.3f} / {table[n][b]['frechet']:8.3f}" for n in names)
        print(f"    {run.name[-6:] + star:10s} {table[names[0]][b]['n']:5d}  {cells}")
    weights = np.array([table[names[0]][b]["n"] for b in range(len(runs))], dtype=float)
    pooled = "  ".join(f"{np.average([e['ADE'] for e in table[n]], weights=weights):10.3f} / "
                       f"{np.mean([e['frechet'] for e in table[n]]):8.3f}" for n in names)
    print(f"    {'pooled':10s} {int(weights.sum()):5d}  {pooled}")
    print("    (ADE = mean distance to the fixes, fix-weighted pooled; Frechet = discrete Frechet distance of the\n"
          "     whole path to the fix polyline, mean over runs)")
    return table


def dead_reckoning_gps(runs: list[ScooterRun]) -> list[dict[str, float]]:
    return [gps_errors(run.states[:, :2], run.t, run) for run in runs]


def steady_state_gain(model: ScooterModel, speed: float = 1.5, delta: float = 0.05, steps: int = 600) -> float:
    """Simulated steady-state r per unit column angle per unit speed = 1 / L_eff."""
    radius = model.gm3.config.tires[0].radius
    state = torch.tensor([[0.0, 0.0, 0.0, speed, 0.0, 0.0, 0.0, 0.0]], device=model.device)
    control = torch.tensor([[speed / radius, delta]], device=model.device)
    substeps = model.stable_substeps(speed)
    h = model.gm3.default_dt / substeps
    with torch.no_grad():
        for _ in range(steps * substeps):
            state = model.gm3(state, control, dt=h)
            state[:, 3] = speed
    return float(state[0, 5]) / (speed * delta)


def report_errors(label: str, model_pre: ScooterModel, model_fit: ScooterModel, windows: RolloutWindows,
                  wheelbase: float) -> dict[str, dict[str, float]]:
    target = windows.target_states
    rows = {"kinematic bicycle": errors(kinematic_trace(windows, wheelbase), target),
            "kinematic bicycle (const speed)": errors(kinematic_trace(windows, wheelbase, hold_speed=True), target)}
    with torch.no_grad():
        for name, model in (("GM3 preset", model_pre), ("GM3 fitted", model_fit)):
            rows[f"{name} (vx pinned)"] = errors(model.simulate(windows, pin_vx=True).cpu().numpy(), target)
            rows[f"{name} (open loop)"] = errors(model.simulate(windows, pin_vx=False).cpu().numpy(), target)
        rows["GM3 fitted (open loop, no slope)"] = errors(
            model_fit.simulate(windows, pin_vx=False, use_slope=False).cpu().numpy(), target)
    print(f"\n  {label}, {windows.controls.shape[1]} windows of {windows.controls.shape[0] * windows.dt:.1f} s:")
    print(f"    {'':34s} {'r RMSE':>9s} {'psi@end':>9s} {'psi p90':>9s} {'v RMSE':>8s} {'ADE':>8s} {'ADE med':>8s} {'FDE':>8s}")
    for name, row in rows.items():
        print(f"    {name:34s} {row['r_rmse_dps']:7.2f}d/s {row['psi_end_deg']:8.2f}d {row['psi_end_p90_deg']:8.2f}d "
              f"{row['v_rmse_mps']:6.3f}m/s {row['ADE_m']:7.3f}m {row['ADE_med_m']:7.3f}m {row['FDE_m']:7.3f}m")
    print("    (measured-speed rows: kinematic bicycle and vx pinned. No-speed-input rows: const speed, open loop)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit DiffGM3 to the Hiboy scooter logs.")
    parser.add_argument("--fit", nargs="+", default=DEFAULT_FIT, choices=sorted(TRAINABLE))
    parser.add_argument("--tire-model", default="brush", choices=sorted(TIRE_FIT),
                        help="tire law (gm3.diffgm3.tire_models); 'cp' in --fit becomes that law's stiffness parameter")
    parser.add_argument("--horizon", type=int, default=45, help="training horizon, steps at 30 Hz")
    parser.add_argument("--eval-horizon", type=int, default=90)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--free-tires", action="store_true",
                        help="let front and rear tire parameters differ (default: tied)")
    parser.add_argument("--open-loop", action="store_true", help="integrate vx too instead of pinning it")
    parser.add_argument("--holdout", default=None, help="run name to hold out (default: last)")
    parser.add_argument("--per-run", action="store_true")
    parser.add_argument("--per-speed", action="store_true")
    parser.add_argument("--motor-on", action="store_true",
                        help="treat the wheel speed as a drive input (default: both wheels free-roll, as recorded)")
    parser.add_argument("--slope", default="imu", choices=("imu", "plane", "none"),
                        help="surface-angle source fed to the model (see gm3.dataset.scooter.surface_angles)")
    parser.add_argument("--root", default=None)
    parser.add_argument("--tag", default="", help="suffix for the output files, e.g. the data folder")
    args = parser.parse_args()

    args.fit = [TIRE_FIT[args.tire_model] if name == "cp" else name for name in args.fit]
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    runs = load_all(args.root, slope_source=args.slope)
    suffix = f"_{args.tag}" if args.tag else ""
    holdout = args.holdout or runs[-1].name
    train_runs = [run for run in runs if run.name != holdout]
    validation = next((run for run in runs if run.name == holdout), None)
    if validation is None:
        raise SystemExit(f"no run named {holdout!r}; have {[r.name for r in runs]}")
    pin_vx = not args.open_loop
    config = make_scooter_config()
    wheelbase = config.wheelbase

    train_windows = windows_of(train_runs, args.horizon, args.stride)
    val_windows = validation.windows(args.horizon, args.stride)
    print(f"train {[r.name for r in train_runs]}\nvalidate {holdout!r}")
    grade = np.degrees(np.concatenate([run.slopes[run.valid, 0] for run in runs]))
    print(f"  {train_windows.controls.shape[1]} training windows of {args.horizon / 30:.2f} s, "
          f"fitting {args.fit}, vx {'pinned' if pin_vx else 'integrated'}, "
          f"front wheel {'driven by omega' if args.motor_on else 'free-rolling (motor off)'}, slope source {args.slope!r} "
          f"(grade std {grade.std():.2f} deg, p95 |grade| {np.percentile(np.abs(grade), 95):.2f} deg)\n")

    kinematic = kinematic_steer_fit(train_runs)
    print(f"  kinematic bar: R^2 {kinematic['r2']:.3f}, implied wheelbase {kinematic['wheelbase_implied']:.3f} m "
          f"(preset {wheelbase:.2f} m)\n")

    preset = ScooterModel(dt=train_windows.dt, motor_on=args.motor_on, tire_model=args.tire_model)
    model = ScooterModel(dt=train_windows.dt, motor_on=args.motor_on, tire_model=args.tire_model)
    with torch.no_grad():
        before = float(loss_on(model, val_windows, pin_vx))
        before_train = float(loss_on(model, train_windows, pin_vx))
    print(f"  preset loss: train {before_train:.4e}, held-out {before:.4e}")
    fit(model, train_windows, args.fit, args.steps, args.lr, pin_vx=pin_vx, tie_tires=not args.free_tires)
    with torch.no_grad():
        after = float(loss_on(model, val_windows, pin_vx))
        after_train = float(loss_on(model, train_windows, pin_vx))

    print("\n  gate 1 - held-out run:")
    verdict = "generalizes" if after < before else "OVERFIT"
    print(f"    train loss {before_train:.4e} -> {after_train:.4e};  held-out {before:.4e} -> {after:.4e}  ({verdict})")

    values_pre, values_fit = preset.values(), model.values()
    print("\n  parameters (* = fitted):")
    for name in ("mu", "cp", "contact_length", "yaw_inertia", "yaw_damping", "align_gain", "steer_zero_deg",
                 "steer_ratio"):
        key = name.replace("_deg", "")
        mark = "*" if key in args.fit else " "
        print(f"    {mark} {name:16s} {values_pre[name]:12.4f} -> {values_fit[name]:12.4f}")
    fz = config.mass * config.gravity / 2.0
    for label, values in (("preset", values_pre), ("fitted", values_fit)):
        c_alpha = cornering_stiffness(values)
        c_rear = values["C_alpha_rear"]
        k_us = config.mass / wheelbase * (config.lr / c_alpha - config.lf / c_rear)
        print(f"    {label}: C_alpha front {c_alpha:.0f} N/rad ({c_alpha / fz:.1f} / rad per N of load), "
              f"rear {c_rear:.0f}; understeer gradient {math.degrees(k_us):+.3f} deg per m/s^2")

    print("\n  gate 4 - steady-state yaw gain at 1.5 m/s (1 / L_eff):")
    measured = abs(kinematic["gain"])
    for label, m in (("preset", preset), ("fitted", model)):
        gain = steady_state_gain(m)
        print(f"    {label}: model {gain:.3f} -> L_eff {1 / gain:.3f} m;  measured {measured:.3f} -> "
              f"L_eff {1 / measured:.3f} m  (ratio {gain / measured:.3f})")

    eval_windows = validation.windows(args.eval_horizon, args.stride)
    rows = report_errors(f"held-out {holdout!r}", preset, model, eval_windows, wheelbase)
    print(f"\n  all runs, {args.eval_horizon / 30:.0f} s windows against the dead-reckoned path (* = held out)")
    print("    measured speed in, ADE m:                              | no speed input, v RMSE m/s / ADE m:")
    measured = ("kinematic bicycle", "GM3 preset", "GM3 fitted")
    blind = ("KBM const speed", "GM3 fitted, no slope", "GM3 fitted, slope")
    windowed: dict[str, list[dict[str, float]]] = {name: [] for name in measured + blind}
    counts = []
    for run in runs:
        w = run.windows(args.eval_horizon, args.stride)
        counts.append(w.controls.shape[1])
        with torch.no_grad():
            traces = {"kinematic bicycle": kinematic_trace(w, wheelbase),
                      "GM3 preset": preset.simulate(w, pin_vx=True).cpu().numpy(),
                      "GM3 fitted": model.simulate(w, pin_vx=True).cpu().numpy(),
                      "KBM const speed": kinematic_trace(w, wheelbase, hold_speed=True),
                      "GM3 fitted, no slope": model.simulate(w, pin_vx=False, use_slope=False).cpu().numpy(),
                      "GM3 fitted, slope": model.simulate(w, pin_vx=False).cpu().numpy()}
        for name, trace in traces.items():
            windowed[name].append(errors(trace, w.target_states))
        star = "*" if run.name == holdout else " "
        print(f"    {run.name[-6:] + star:8s} n={counts[-1]:3d}  "
              + "  ".join(f"{windowed[n][-1]['ADE_m']:.3f}" for n in measured) + "   | "
              + "  ".join(f"{windowed[n][-1]['v_rmse_mps']:.3f}/{windowed[n][-1]['ADE_m']:.3f}" for n in blind))
    print(f"    {'pooled':8s} n={sum(counts):3d}  "
          + "  ".join(f"{np.average([e['ADE_m'] for e in windowed[n]], weights=counts):.3f}" for n in measured) + "   | "
          + "  ".join(f"{np.average([e['v_rmse_mps'] for e in windowed[n]], weights=counts):.3f}/"
                      f"{np.average([e['ADE_m'] for e in windowed[n]], weights=counts):.3f}" for n in blind))
    print("    columns: " + ", ".join(measured) + " | " + ", ".join(blind))

    # The same no-speed-input comparison on coasting windows only: the rider
    # is not kicking or braking inside them, so speed is set by rolling
    # resistance and slope alone, which is what those two terms model.
    # The rider kicks almost continuously: at 3 s only one window in the whole
    # set is kick-free, so this uses 1.5 s windows and a 0.35 m/s^2 threshold
    # on the smoothed speed derivative.
    coasting: dict[str, list[np.ndarray]] = {name: [] for name in blind}
    n_coast = 0
    coast_horizon = min(args.eval_horizon, 45)
    for run in runs:
        w = run.windows(coast_horizon, args.stride)
        smooth = np.apply_along_axis(_moving_average, 0, w.target_states[:, :, 3], 9)
        accel = np.gradient(smooth, w.dt, axis=0)
        quiet = np.flatnonzero(np.abs(accel).max(axis=0) < 0.35)
        if len(quiet) == 0:
            continue
        n_coast += len(quiet)
        with torch.no_grad():
            traces = {"KBM const speed": kinematic_trace(w, wheelbase, hold_speed=True)[:, quiet],
                      "GM3 fitted, no slope": model.simulate(w, pin_vx=False, keep=quiet, use_slope=False).cpu().numpy(),
                      "GM3 fitted, slope": model.simulate(w, pin_vx=False, keep=quiet).cpu().numpy()}
        target = w.target_states[:, quiet]
        for name, trace in traces.items():
            coasting[name].append(np.stack([trace[1:, :, 3] - target[1:, :, 3],
                                            np.linalg.norm(trace[1:, :, :2] - target[1:, :, :2], axis=-1)]))
    if n_coast:
        print(f"    coasting windows only ({coast_horizon / 30:.1f} s, |dv/dt| < 0.35 m/s^2 throughout, n={n_coast}), v RMSE m/s / ADE m:  "
              + "  ".join(f"{name} {np.sqrt((np.concatenate(v, axis=2)[0] ** 2).mean()):.3f}/"
                          f"{np.concatenate(v, axis=2)[1].mean():.3f}" for name, v in coasting.items()))

    gps_models = {"kinematic bicycle": None, "GM3 preset": preset, "GM3 fitted": model}
    if args.motor_on:
        gps_models["GM3 fitted (open loop)"] = model
    gps = print_gps_table(gps_models, runs, wheelbase, holdout, open_loop=("GM3 fitted (open loop)",))
    dr = dead_reckoning_gps(runs)
    print(f"    reference: wheel + gyro dead reckoning ADE / Frechet "
          + "  ".join(f"{run.name[-6:]} {e['ADE']:.2f}/{e['frechet']:.2f}" for run, e in zip(runs, dr)))

    OUT.mkdir(exist_ok=True)
    with open(OUT / f"scooter_calibration{suffix}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "preset", "fitted", "fitted?"])
        for name in values_fit:
            writer.writerow([name, values_pre[name], values_fit[name], name.replace("_deg", "") in args.fit])
    summary = {"holdout": holdout, "fit": args.fit, "tire_model": args.tire_model, "pin_vx": pin_vx, "slope": args.slope, "motor_on": args.motor_on,
               "root": str(args.root), "tag": args.tag,
               "loss_before": before, "loss_after": after,
               "parameters": values_fit, "cornering_stiffness_front": cornering_stiffness(values_fit),
               "held_out_errors": rows,
               "windowed_vs_dead_reckoning": {name: dict(zip([r.name for r in runs], values))
                                              for name, values in windowed.items()},
               "gps_ade_whole_run": {name: dict(zip([r.name for r in runs], [e["ADE"] for e in table]))
                                     for name, table in gps.items()},
               "gps_frechet_whole_run": {name: dict(zip([r.name for r in runs], [e["frechet"] for e in table]))
                                         for name, table in gps.items()},
               "gps_ade_dead_reckoning": dict(zip([r.name for r in runs], [e["ADE"] for e in dr]))}
    (OUT / f"scooter_calibration{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {OUT / f'scooter_calibration{suffix}.csv'} and .json")

    if args.per_run:
        print("\n  gate 2 - per-run consistency:")
        tracked = args.fit[0]
        results = []
        for run in runs:
            local = fit(ScooterModel(dt=run.dt, motor_on=args.motor_on, tire_model=args.tire_model), run.windows(args.horizon, args.stride), args.fit,
                        args.steps, args.lr, pin_vx=pin_vx, quiet=True, tie_tires=not args.free_tires)
            v = local.values()
            results.append(cornering_stiffness(v) if tracked in ("contact_length", "cp") else v[tracked if tracked != "steer_zero" else "steer_zero_deg"])
            print(f"    {run.name:16s} C_alpha={cornering_stiffness(v):7.0f}  Jz={v['yaw_inertia']:.3f}  "
                  f"c_r={v['yaw_damping']:.3f}  mu={v['mu']:.3f}  zero={v['steer_zero_deg']:+.2f} deg  "
                  f"ratio={v['steer_ratio']:.4f}")
        lo, hi = min(results), max(results)
        spread = hi / lo if lo > 0 else float("inf")
        print(f"    {tracked} spread {lo:.3f}..{hi:.3f} (ratio {spread:.2f}x; accept under ~1.5x)")

    if args.per_speed:
        print("\n  gate 3 - per-speed-tercile consistency:")
        speed = train_windows.target_states[:-1, :, 3].mean(axis=0)
        edges = np.percentile(speed, [33, 67])
        bands = [speed <= edges[0], (speed > edges[0]) & (speed <= edges[1]), speed > edges[1]]
        for label, mask in zip(("slow", "mid", "fast"), bands):
            local = fit(ScooterModel(dt=train_windows.dt, motor_on=args.motor_on, tire_model=args.tire_model), train_windows, args.fit, args.steps, args.lr,
                        pin_vx=pin_vx, keep=np.flatnonzero(mask), quiet=True, tie_tires=not args.free_tires)
            v = local.values()
            print(f"    {label:5s} (v ~ {speed[mask].mean():.2f} m/s, n={mask.sum()})  C_alpha={cornering_stiffness(v):7.0f}  "
                  f"Jz={v['yaw_inertia']:.3f}  c_r={v['yaw_damping']:.3f}  zero={v['steer_zero_deg']:+.2f} deg")


if __name__ == "__main__":
    main()
