"""Plots and parameter tables for the per-tire-law scooter calibrations.

    python -m experiments.scooter_tire_model_plots --surfaces kim_quad slope_sidewalk

Reads what ``scooter_tire_models fit`` / ``profile`` left in
``out/tire_models[_<surfaces>]/`` and writes, next to it:

* ``parameters.md`` / ``parameters.csv`` - every law's parameters, all-data
  fit and held-out fit side by side, with what was fitted and what was held.
* ``plots/force_curves.png``  - each fitted law's lateral force against slip
  angle at the static front load, with the band the recordings actually load
  the tire to.
* ``plots/accuracy.png``      - held-out accuracy of every law against the
  kinematic bicycle and the unfitted brush preset, one panel per metric.
* ``plots/gps_per_run.png``   - whole-run ADE against the GPS fixes per run.
* ``plots/stiffness_profile.png`` - loss along the cornering-stiffness axis.
* ``plots/parameters.png``    - fitted parameters, held-out fit vs all-data fit.
* ``plots/holdout.png``       - held-out runs: yaw rate and heading error at 3 s.
* ``plots/paths.png``         - whole-run paths against the GPS fixes.
* ``plots/paths/<run>.png``   - (``--only path_per_run``) one bare figure per run, every law on it, no title
  and no legend; ``plots/paths/legend.png`` is the shared legend. Held-out runs end in ``_heldout``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import _bootstrap  # noqa: F401  (registers the `gm3` package)

from experiments import scooter_tire_models as stm
from experiments.scooter_calibrate import ScooterModel, errors, kinematic_trace, whole_runs
from experiments.scooter_plots import align_to_gps
from gm3.dataset.scooter import ScooterRun, gps_errors
from gm3.shared import make_scooter_config

# Categorical slots 1-5 of the dataviz reference palette, fixed per law; baselines are neutral ink.
COLORS = {"brush": "#2a78d6", "fiala": "#eb6834", "dugoff": "#1baf7a", "burckhardt": "#eda100", "pacejka": "#e87ba4"}
MARKERS = {"brush": "o", "fiala": "s", "dugoff": "^", "burckhardt": "D", "pacejka": "v"}
KBM, PRESET = "#52514e", "#b3b2ab"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.axisbelow": True,
})


def load_model(name: str, suffix: str, dt: float) -> ScooterModel:
    model = stm.make_model(name, dt)
    model.load_state_dict(torch.load(stm.OUT / f"{name}{suffix}.pt", map_location="cpu"))
    return model


# ---------------------------------------------------------------- parameter tables

def write_parameter_files(held: dict[str, dict], everything: dict[str, dict]) -> None:
    lines = ["# Tire-law parameters", ""]
    first = next(iter((everything or held).values()))
    surfaces = first.get("surfaces", {})
    lines += ["Data: " + "; ".join(f"{surface} ({len(runs)} runs)" for surface, runs in surfaces.items()) + ".",
              "Protocol: 1.5 s windows, vx pinned to the wheel speed, IMU slope input, Adam from the scooter preset, "
              f"{first['steps']} steps, front and rear tire tied. Bold = fitted, (held) = left at the preset.", ""]
    if everything:
        lines += ["## All-data fit (the reported parameters)", ""] + stm.parameter_table(everything) + [""]
    if held:
        lines += [f"## Held-out fit (runs {', '.join(first_held['holdouts'])} excluded)"
                  for first_held in [next(iter(held.values()))]]
        lines += [""] + stm.parameter_table(held) + [""]
    lines += ["## Preset (starting point of every fit)", ""]
    source = everything or held
    preset = {name: {**s, "parameters": s["preset_parameters"], "fit": []} for name, s in source.items()}
    lines += [row.replace("**", "").replace(" (held)", "") for row in stm.parameter_table(preset)] + [""]
    (stm.OUT / "parameters.md").write_text("\n".join(lines))

    with open(stm.OUT / "parameters.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "fit", "parameter", "preset", "fitted", "was_fitted"])
        for label, summaries in (("all_data", everything), ("held_out", held)):
            for name, s in summaries.items():
                fitted = set().union(*(stm.FITTED_KEYS[n] for n in s["fit"])) | {"C_alpha", "C_alpha_rear"}
                for key, value in s["parameters"].items():
                    relevant = key in stm.USED[name] or key in ("C_alpha", "C_alpha_rear", "yaw_inertia", "yaw_damping",
                                                                "steer_zero_deg", "steer_ratio")
                    if relevant:
                        writer.writerow([name, label, key, s["preset_parameters"].get(key), value, key in fitted])


# ---------------------------------------------------------------- plots

def lateral_force(model: ScooterModel, alpha: np.ndarray) -> tuple[np.ndarray, float]:
    """Front-tire ``-Fy`` over ``alpha`` (rad) at the static front load, free rolling."""
    g = model.gm3
    front = float(g.mass * g.gravity * g.lr / g.wheelbase / g.front_count)
    rear = float(g.mass * g.gravity * g.lf / g.wheelbase / g.rear_count)
    loads = torch.where(g.front_mask, torch.tensor(front), torch.tensor(rear)).reshape(1, g.n_tires).repeat(len(alpha), 1)
    slip = torch.as_tensor(alpha).reshape(-1, 1).repeat(1, g.n_tires)
    zeros = torch.zeros_like(loads)
    params = g.physical_parameters()
    with torch.no_grad():
        _, fy, _ = g._tire_forces(
            longitudinal_speed=zeros + 1.0, kappa=zeros, sigma_x=zeros, sigma_y=torch.tan(slip), normal_loads=loads,
            alpha=slip, steering_angles=zeros, gamma=zeros[:, 0], tire_radius=g.tire_radius, wheelbase=g.wheelbase,
            eps=g.eps, min_normal_load=g.min_normal_load, lean_mask=g.lean_mask, has_leaning_tires=g._has_leaning_tires,
            mu=params["mu"], cp=params["cp"], contact_length=params["contact_length"])
    index = int(torch.nonzero(g.front_mask)[0])
    return -fy[:, index].numpy(), front


def plot_force_curves(models: dict[str, ScooterModel], runs: list[ScooterRun], label: str) -> None:
    config = make_scooter_config()
    ay = np.concatenate([np.abs(run.states[run.valid, 3] * run.states[run.valid, 5]) for run in runs])
    demand = {q: config.mass * np.percentile(ay, q) * config.lr / config.wheelbase for q in (95, 100)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    # Zoomed panel: wide enough for the softest law to pass 1.6x the largest recorded load.
    softest = min(model.values()["C_alpha"] for model in models.values())
    zoom = max(0.25, math.ceil(4.0 * math.degrees(1.6 * demand[100] / softest)) / 4.0)
    for ax, top, title in ((axes[0], 25.0, "Full curve"), (axes[1], zoom, "The range the recordings reach")):
        alpha = np.radians(np.linspace(0.0, top, 400))
        for name, model in models.items():
            fy, front = lateral_force(model, alpha)
            ax.plot(np.degrees(alpha), fy, color=COLORS[name], lw=2, label=name)
        ax.axhspan(0, demand[95], color=INK2, alpha=0.10, lw=0)
        ax.axhline(demand[100], color=INK2, lw=1, ls=":")
        ax.set_xlabel("slip angle, deg")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, top)
    axes[0].axhline(0.8 * front, color=MUTED, lw=1, ls="--")
    axes[0].text(24.8, 0.8 * front - 6, "mu Fz (mu = 0.8, held)", ha="right", va="top", fontsize=8, color=INK2)
    axes[0].text(24.8, demand[100], "largest recorded lateral load", ha="right", va="bottom", fontsize=8, color=INK2)
    axes[0].text(24.8, demand[95] * 0.5, "below the p95 recorded load", ha="right", va="center", fontsize=8, color=INK2)
    axes[1].set_ylim(0, demand[100] * 1.6)
    axes[0].set_ylabel("front-tire lateral force, N")
    axes[1].legend(loc="lower right", ncol=1)
    fig.suptitle(f"Fitted lateral force laws at the static front load ({front:.0f} N), {label}", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(stm.OUT / "plots" / "force_curves.png", dpi=150)
    plt.close(fig)


def plot_accuracy(held: dict[str, dict]) -> None:
    first = next(iter(held.values()))
    rows = [("kinematic bicycle", first["kinematic"], KBM)]
    if "brush" in held:
        rows.append(("brush preset (unfitted)", held["brush"]["preset"], PRESET))
    rows += [(f"{name} fitted", s["fitted"], COLORS[name]) for name, s in held.items()]
    metrics = [("yaw loss, 1.5 s windows (x 1e-3)", lambda r: 1e3 * r["loss"]["held-out"], "{:.2f}"),
               ("yaw-rate RMSE, 3 s windows (deg/s)", lambda r: r["windows"]["held-out"]["r_rmse_dps"], "{:.2f}"),
               ("median |heading error| at 3 s (deg)", lambda r: r["windows"]["held-out"]["psi_end_deg"], "{:.2f}"),
               ("ADE vs dead reckoning, 3 s windows (m)", lambda r: r["windows"]["held-out"]["ADE_m"], "{:.3f}"),
               ("whole-run ADE vs GPS fixes (m)", lambda r: r["gps"]["held-out"]["ADE"], "{:.3f}"),
               ("whole-run Frechet vs GPS fixes (m)", lambda r: r["gps"]["held-out"]["frechet"], "{:.2f}")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (title, pick, form) in zip(axes.flat, metrics):
        values = [pick(result) for _, result, _ in rows]
        ax.barh(y, values, height=0.62, color=[c for _, _, c in rows], linewidth=0)
        for yi, value in zip(y, values):
            ax.text(value, yi, " " + form.format(value), va="center", ha="left", fontsize=8, color=INK)
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_yticks(y, [label for label, _, _ in rows])
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Held-out accuracy (runs {', '.join(h[-6:] for h in first['holdouts'])}); lower is better",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(stm.OUT / "plots" / "accuracy.png", dpi=150)
    plt.close(fig)


def plot_gps_per_run(held: dict[str, dict]) -> None:
    first = next(iter(held.values()))
    runs = first["kinematic"]["gps"]["all"]["runs"]
    surfaces = list(dict.fromkeys(row["surface"] for row in runs.values()))
    widths = [sum(1 for row in runs.values() if row["surface"] == s) for s in surfaces]
    fig, axes = plt.subplots(1, len(surfaces), figsize=(13, 4.4), sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, surface in zip(np.atleast_1d(axes), surfaces):
        names = [name for name, row in runs.items() if row["surface"] == surface]
        x = np.arange(len(names))
        ax.bar(x, [runs[n]["ADE"] for n in names], width=0.74, color=GRID, edgecolor=MUTED, linewidth=0.8,
               label="kinematic bicycle")
        offsets = np.linspace(-0.26, 0.26, len(held))
        for offset, (law, s) in zip(offsets, held.items()):
            ax.scatter(x + offset, [s["fitted"]["gps"]["all"]["runs"][n]["ADE"] for n in names], s=30,
                       color=COLORS[law], marker=MARKERS[law], edgecolor="white", linewidth=0.7, zorder=3,
                       label=f"{law} fitted")
        ax.set_xticks(x, [n[-6:] + ("*" if n in first["holdouts"] else "") for n in names], rotation=45, ha="right")
        ax.set_title(f"{surface} ({len(names)} runs)", fontsize=10)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("whole-run ADE to the GPS fixes, m")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.94), fontsize=9)
    fig.suptitle("Whole-run path error against GPS, every run (* = held out)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(stm.OUT / "plots" / "gps_per_run.png", dpi=150)
    plt.close(fig)


def plot_profiles(profiles: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, group in zip(axes, ("train", "held-out")):
        for name, profile in profiles.items():
            c = [row["C_alpha"] for row in profile["profile"]]
            loss = [1e3 * row[group] for row in profile["profile"]]
            ax.plot(c, loss, color=COLORS[name], lw=2, marker=MARKERS[name], ms=5, markeredgecolor="white",
                    markeredgewidth=0.8, label=name)
            ax.axvline(profile["fitted_C_alpha"], color=COLORS[name], lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("zero-slip cornering stiffness C_alpha, N/rad (dotted: each law's fitted value)")
        ax.set_title(f"{group} runs", fontsize=10)
    axes[0].set_ylabel("yaw loss x 1e-3 (other parameters at the fit)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Loss along the cornering-stiffness axis", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(stm.OUT / "plots" / "stiffness_profile.png", dpi=150)
    plt.close(fig)


def plot_parameters(held: dict[str, dict], everything: dict[str, dict]) -> None:
    keys = [("C_alpha", "C_alpha, N/rad", 4500.0), ("yaw_inertia", "yaw inertia, kg m^2", None),
            ("yaw_damping", "yaw damping", None), ("steer_zero_deg", "residual steer zero, deg motor", None)]
    names = list(everything or held)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (key, title, _) in zip(axes, keys):
        source = everything or held
        preset = next(iter(source.values()))["preset_parameters"][key]
        ax.axvline(preset, color=MUTED, lw=1, ls="--")
        for yi, name in zip(y, names):
            if name in held:
                ax.scatter(held[name]["parameters"][key], yi + 0.14, s=44, facecolor="white", edgecolor=COLORS[name],
                           linewidth=1.8, marker=MARKERS[name], zorder=3)
            if name in everything:
                ax.scatter(everything[name]["parameters"][key], yi - 0.14, s=44, color=COLORS[name],
                           marker=MARKERS[name], zorder=3)
        ax.set_yticks(y, names)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    kinds = [k for k, present in (("filled = all-data fit", everything), ("hollow = held-out fit", held)) if present]
    fig.suptitle("Fitted parameters: " + ", ".join(kinds) + ", dashed = preset", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(stm.OUT / "plots" / "parameters.png", dpi=150)
    plt.close(fig)


def window_times(run: ScooterRun, horizon: int) -> np.ndarray:
    starts = []
    from gm3.dataset.scooter import contiguous_runs
    for begin, end in contiguous_runs(run.valid, np.zeros(len(run.valid), dtype=int)):
        starts.extend(range(begin, end - horizon, horizon))
    return run.t[np.asarray(starts)[:, None] + np.arange(horizon + 1)[None, :]]


def plot_holdout(models: dict[str, ScooterModel], runs: list[ScooterRun], wheelbase: float, best: str,
                 horizon: int = 90) -> None:
    """``best`` is the law drawn against the bicycle in the two yaw-rate rows; the bottom row carries every law."""
    fig, axes = plt.subplots(3, len(runs), figsize=(6.5 * len(runs), 8.6), squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for col, run in enumerate(runs):
        windows = run.windows(horizon)
        t_win = window_times(run, horizon)
        target = windows.target_states
        traces = {"kinematic bicycle": kinematic_trace(windows, wheelbase)}
        with torch.no_grad():
            traces.update({name: model.simulate(windows, pin_vx=True).numpy() for name, model in models.items()})
        ax = axes[0, col]
        ax.plot(run.t, np.degrees(run.states[:, 5]), color=INK, lw=1.0, label="measured (gyro)")
        for name, color in (("kinematic bicycle", KBM), (best, COLORS[best])):
            for b in range(target.shape[1]):
                ax.plot(t_win[b], np.degrees(traces[name][:, b, 5]), color=color, lw=1.5, alpha=0.9,
                        ls="--" if name == "kinematic bicycle" else "-",
                        label=(name if name == "kinematic bicycle" else f"{name} fitted") if b == 0 else None)
        ax.set_title(f"held-out run {run.name[-6:]}: 3 s windows from the measured initial state", fontsize=10)
        ax.set_ylabel("yaw rate, deg/s")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax = axes[1, col]
        ax.axhline(0, color=MUTED, lw=0.8)
        for name, color in (("kinematic bicycle", KBM), (best, COLORS[best])):
            for b in range(target.shape[1]):
                ax.plot(t_win[b], np.degrees(traces[name][:, b, 5] - target[:, b, 5]), color=color, lw=1.1,
                        alpha=0.85, label=(name if name == "kinematic bicycle" else f"{name} fitted") if b == 0 else None)
        ax.set_ylabel("yaw-rate error, deg/s")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax = axes[2, col]
        for name, trace in traces.items():
            err = np.degrees(np.abs(trace[-1, :, 2] - target[-1, :, 2]))
            style = dict(color=KBM, ls="--", marker="x") if name == "kinematic bicycle" else \
                dict(color=COLORS[name], marker=MARKERS[name], markeredgecolor="white", markeredgewidth=0.6)
            ax.plot(t_win[:, -1], err, lw=1.3, ms=5, label=name, **style)
        ax.set_ylabel("|heading error| at 3 s, deg")
        ax.set_xlabel("time, s")
        ax.set_ylim(0, None)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(stm.OUT / "plots" / "holdout.png", dpi=150)
    plt.close(fig)


def plot_paths(models: dict[str, ScooterModel], runs: list[ScooterRun], wheelbase: float, holdouts: set[str]) -> None:
    windows, lengths = whole_runs(runs)
    traces = {"kinematic bicycle": kinematic_trace(windows, wheelbase)}
    with torch.no_grad():
        traces.update({name: model.simulate(windows, pin_vx=True).numpy() for name, model in models.items()})
    cols = 2 if len(runs) == 4 else min(len(runs), 3)
    rows = math.ceil(len(runs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.6 * rows), squeeze=False)
    for ax, (b, run) in zip(axes.flat, enumerate(runs)):
        n, gps, score = lengths[b], None, {}
        for name, trace in traces.items():
            path = trace[:n, b, :2]
            aligned, gps = align_to_gps(path, run.t, run)
            score[name] = gps_errors(path, run.t, run)["ADE"]
            kbm = name == "kinematic bicycle"
            ax.plot(aligned[:, 0], aligned[:, 1], color=KBM if kbm else COLORS[name], lw=1.4 if kbm else 1.6,
                    ls="--" if kbm else "-", label=name if kbm else f"{name} fitted")
        ax.scatter(gps[:, 0], gps[:, 1], s=10, color=INK, zorder=4, label="GPS fix")
        ax.set_aspect("equal")
        held = "  (held out)" if run.name in holdouts else ""
        laws = [f"{k} {v:.2f}" for k, v in score.items() if k != "kinematic bicycle"]
        ax.set_title(f"{run.name[-6:]}{held}   ADE to GPS, m:  KBM {score['kinematic bicycle']:.2f}\n"
                     + "   ".join(laws[:3]) + ("\n" + "   ".join(laws[3:]) if laws[3:] else ""), fontsize=8)
        ax.set_xlabel("east, m")
        ax.set_ylabel("north, m")
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 0.985), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(stm.OUT / "plots" / "paths.png", dpi=150)
    plt.close(fig)


def simulate_whole_runs(models: dict[str, ScooterModel], runs: list[ScooterRun], wheelbase: float,
                        cache: Path) -> dict[str, list[np.ndarray]]:
    """Whole-run ``[N, 2]`` paths per model per run, cached: the rollout takes minutes, restyling should not."""
    names = ["kinematic bicycle"] + list(models)
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if all(f"{name}|{run.name}" in stored for name in names for run in runs):
            return {name: [stored[f"{name}|{run.name}"] for run in runs] for name in names}
    windows, lengths = whole_runs(runs)
    traces = {"kinematic bicycle": kinematic_trace(windows, wheelbase)}
    with torch.no_grad():
        for name, model in models.items():
            traces[name] = model.simulate(windows, pin_vx=True).numpy()
            print(f"  simulated {name}", flush=True)
    out = {name: [trace[:n, b, :2] for b, n in enumerate(lengths)] for name, trace in traces.items()}
    np.savez_compressed(cache, **{f"{name}|{run.name}": path for name, paths in out.items()
                                  for run, path in zip(runs, paths)})
    return out


def plot_path_per_run(models: dict[str, ScooterModel], runs: list[ScooterRun], wheelbase: float,
                      holdouts: set[str], baseline: bool) -> None:
    folder = stm.OUT / "plots" / "paths"
    folder.mkdir(parents=True, exist_ok=True)
    paths = simulate_whole_runs(models, runs, wheelbase, folder / "traces.npz")
    if not baseline:
        paths.pop("kinematic bicycle")
    # Later laws are drawn thinner so that coincident paths stay visible underneath each other.
    widths = dict(zip(models, np.linspace(3.4, 1.2, len(models))))
    handles = []
    for b, run in enumerate(runs):
        # Canvas shaped like the path (long side 7 in, short side at least 3 in) so equal axes waste nothing.
        aligned_all = {name: align_to_gps(per_run[b], run.t, run) for name, per_run in paths.items()}
        points = np.concatenate([a for a, _ in aligned_all.values()] + [next(iter(aligned_all.values()))[1]])
        extent = np.maximum(points.max(0) - points.min(0), 1e-6)
        size = np.maximum(7.0 * extent / extent.max(), 3.0) + np.array([0.9, 0.7])   # room for the axis labels
        fig, ax = plt.subplots(figsize=tuple(size))
        handles, gps, start = [], None, None
        for name in paths:
            aligned, gps = aligned_all[name]
            kbm = name == "kinematic bicycle"
            line, = ax.plot(aligned[:, 0], aligned[:, 1], color=KBM if kbm else COLORS[name],
                            lw=1.3 if kbm else widths[name], ls="--" if kbm else "-", zorder=4 if kbm else 2,
                            label=name if kbm else f"{name} fitted")
            handles.append(line)
            start = aligned[0] if start is None or not kbm else start
        handles.append(ax.scatter(gps[:, 0], gps[:, 1], s=14, color=INK, zorder=5, label="GPS fix"))
        handles.append(ax.scatter([start[0]], [start[1]], marker="x", s=80, color="#e34948", linewidths=2.2,
                                  zorder=6, label="start"))
        ax.set_aspect("equal")
        ax.margins(0.04)
        ax.set_xlabel("east, m")
        ax.set_ylabel("north, m")
        fig.tight_layout()
        fig.savefig(folder / f"{run.name[-6:]}{'_heldout' if run.name in holdouts else ''}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    fig = plt.figure(figsize=(2.6, 0.3 * len(handles) + 0.3))
    fig.legend(handles=handles, loc="center", ncol=1, fontsize=10, handlelength=2.6)
    fig.savefig(folder / "legend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig = plt.figure(figsize=(12, 0.5))
    fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=10, handlelength=2.6)
    fig.savefig(folder / "legend_horizontal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--surfaces", nargs="+", default=None, choices=sorted(stm.SURFACE_SLUGS))
    parser.add_argument("--slope", default="imu", choices=("imu", "plane", "none"))
    parser.add_argument("--only", nargs="+", default=None,
                        choices=("force_curves", "parameters", "accuracy", "gps_per_run", "holdout", "paths", "stiffness_profile",
                                 "path_per_run"),
                        help="draw just these (holdout and paths re-simulate every law and take minutes)")
    parser.add_argument("--no-baseline", action="store_true", help="path_per_run: leave the kinematic bicycle out")
    parser.add_argument("--paths", nargs="+", default=None,
                        help="run names (suffixes) for paths.png; default: the held-out runs and each surface's longest run")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    stm.OUT = stm.output_dir(args.surfaces)
    (stm.OUT / "plots").mkdir(parents=True, exist_ok=True)

    held, everything = stm.load_summaries(""), stm.load_summaries("_all")
    if not held and not everything:
        raise SystemExit(f"nothing fitted in {stm.OUT}")
    write_parameter_files(held, everything)
    print(f"wrote {stm.OUT / 'parameters.md'} and .csv")

    surfaces = stm.load_surfaces(args.slope, args.surfaces)
    every = [run for runs in surfaces.values() for run in runs]
    wheelbase = make_scooter_config().wheelbase
    label = " + ".join(surfaces)
    want = lambda name: args.only is None or name in args.only  # noqa: E731
    # The all-data fits are the reported parameters; fall back to the held-out fits when they have not been run.
    suffix, source = ("_all", everything) if everything else ("", held)
    which = "all-data fit" if everything else "held-out fit"
    if want("force_curves"):
        plot_force_curves({name: load_model(name, suffix, every[0].dt) for name in source}, every, f"{label}, {which}")
    if want("parameters"):
        plot_parameters(held, everything)
    if held:
        holdouts = set(next(iter(held.values()))["holdouts"])
        models = {name: load_model(name, "", every[0].dt) for name in held}
        if want("accuracy"):
            plot_accuracy(held)
        if want("gps_per_run"):
            plot_gps_per_run(held)
        best = min(held, key=lambda name: held[name]["fitted"]["loss"]["held-out"])
        if want("holdout"):
            plot_holdout(models, [run for run in every if run.name in holdouts], wheelbase, best)
        if args.paths:
            chosen = [run for run in every if any(run.name.endswith(p) for p in args.paths)]
        else:
            longest = {max(runs, key=lambda r: r.moving_seconds).name for runs in surfaces.values()}
            chosen = [run for run in every if run.name in holdouts | longest]
        if want("paths"):
            plot_paths(models, chosen, wheelbase, holdouts)
        if args.only and "path_per_run" in args.only:   # opt-in: twenty figures and a long rollout
            plot_path_per_run(models, every, wheelbase, holdouts, baseline=not args.no_baseline)
    profiles = {name: json.loads((stm.OUT / f"{name}_profile.json").read_text()) for name in stm.MODELS
                if (stm.OUT / f"{name}_profile.json").exists()}
    if profiles and want("stiffness_profile"):
        plot_profiles(profiles)
    for path in sorted((stm.OUT / "plots").rglob("*.png")):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
