"""Plot the fitted GM3 scooter model against the kinematic bicycle on every run.

    python -m experiments.scooter_plots
    python -m experiments.scooter_plots --params experiments/out/scooter_calibration.json

Writes to ``experiments/out/plots/``:

* ``scooter_paths.png``   - whole-run path per run: GPS fixes as the reference,
  with the kinematic bicycle and fitted GM3 each rigidly aligned to them; the
  panel title carries each model's ADE against the fixes.
* ``scooter_holdout.png`` - held-out run: measured vs predicted yaw rate over
  3 s open-loop windows, and the heading error at each window end.
* ``scooter_errors.png``  - per-run yaw-rate RMSE, heading error at 3 s and
  whole-run ADE against GPS for both models (the held-out run is marked; the
  rest were training data).
* ``scooter_residual.png`` - yaw-rate error by steering angle, both models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import _bootstrap  # noqa: F401  (registers the `gm3` package)

from experiments.scooter_calibrate import ScooterModel, errors, gps_table, kinematic_trace, whole_runs
from gm3.dataset.scooter import STEER_RATIO, STEER_SIGN, ScooterRun, gps_errors, load_all
from gm3.diffgm3.torch_utils import raw_bounded, raw_log_bounded, raw_positive
from gm3.shared import make_scooter_config
from gm3.shared.constants import CONTACT_LENGTH_BOUNDS, CP_BOUNDS

OUT = Path(__file__).resolve().parent / "out"
PLOTS = OUT / "plots"   # reassigned per --tag in main()

# Categorical slots from the dataviz reference palette; measured data is ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"
SERIES = {"GM3 fitted": BLUE, "kinematic bicycle": ORANGE, "measured": INK2}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.titlecolor": INK, "axes.titleweight": "semibold",
})


def fitted_model(dt: float, params_path: Path) -> ScooterModel:
    meta = json.loads(params_path.read_text())
    values = meta["parameters"]
    model = ScooterModel(dt=dt, motor_on=meta.get("motor_on", False))
    with torch.no_grad():
        model.gm3.raw_contact_length[0] = raw_bounded(values["contact_length"], *CONTACT_LENGTH_BOUNDS)
        model.gm3.raw_contact_length[1] = raw_bounded(values["contact_length_rear"], *CONTACT_LENGTH_BOUNDS)
        model.gm3.raw_cp[0] = raw_log_bounded(values["cp"], *CP_BOUNDS)
        model.gm3.raw_cp[1] = raw_log_bounded(values["cp_rear"], *CP_BOUNDS)
        model.gm3.raw_yaw_inertia.copy_(raw_positive(values["yaw_inertia"], 1e-6))
        model.gm3.raw_yaw_damping.copy_(raw_positive(values["yaw_damping"], 1e-6))
        model.steer_zero.fill_(math.radians(values["steer_zero_deg"]))
        model.log_steer_ratio.fill_(math.log(values["steer_ratio"]))
    return model


def align_to_gps(path_xy: np.ndarray, path_t: np.ndarray, run: ScooterRun) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align a predicted path to the GPS fixes; return (aligned path, gps)."""
    keep = (run.gps_t >= path_t[0]) & (run.gps_t <= path_t[-1])
    gps = run.gps_xy[keep]
    at_fix = np.stack([np.interp(run.gps_t[keep], path_t, path_xy[:, i]) for i in range(2)], axis=1)
    p_mean, g_mean = at_fix.mean(0), gps.mean(0)
    u, _, vt = np.linalg.svd((at_fix - p_mean).T @ (gps - g_mean))
    rotation = u @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return (path_xy - p_mean) @ rotation + g_mean, gps


def whole_run_paths(model: ScooterModel, runs: list[ScooterRun], wheelbase: float) -> list[dict[str, np.ndarray]]:
    windows, lengths = whole_runs(runs)
    kbm = kinematic_trace(windows, wheelbase)
    with torch.no_grad():
        gm3 = model.simulate(windows, pin_vx=True).numpy()
    return [{"kinematic bicycle": kbm[:n, b, :2], "GM3 fitted": gm3[:n, b, :2]} for b, n in enumerate(lengths)]


def plot_paths(runs, model, wheelbase, holdout):
    cols = 3
    rows = math.ceil(len(runs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows))
    short = {"kinematic bicycle": "KBM", "GM3 fitted": "GM3"}
    all_paths = whole_run_paths(model, runs, wheelbase)
    for ax, run, paths in zip(axes.flat, runs, all_paths):
        gps, score, start = None, {}, None
        for name, path in paths.items():
            aligned, gps = align_to_gps(path, run.t, run)
            score[name] = gps_errors(path, run.t, run)
            ax.plot(aligned[:, 0], aligned[:, 1], color=SERIES[name], lw=1.8, label=name)
            start = aligned[0] if start is None else start
        ax.scatter(gps[:, 0], gps[:, 1], s=12, color=INK2, zorder=3, label="GPS fix")
        ax.scatter([start[0]], [start[1]], marker="x", s=70, color="#e34948", linewidths=2.2, zorder=5, label="start")
        ax.set_aspect("equal")
        tag = "  (held out)" if run.name == holdout else ""
        ax.set_title(f"{run.name[-6:]}{tag}\n"
                     + "ADE  " + "  ".join(f"{short[k]} {v['ADE']:.2f}" for k, v in score.items()) + " m\n"
                     + "Frechet  " + "  ".join(f"{short[k]} {v['frechet']:.2f}" for k, v in score.items()) + " m",
                     fontsize=8)
        ax.set_xlabel("east, m")
        ax.set_ylabel("north, m")
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965), fontsize=9)
    fig.suptitle("Whole-run paths against the GPS fixes (each path rigidly aligned to the fixes; GPS sigma ~3 m)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOTS / "scooter_paths.png", dpi=150)
    plt.close(fig)


def windowed(model, run, wheelbase, horizon=90):
    windows = run.windows(horizon)
    with torch.no_grad():
        gm3 = model.simulate(windows, pin_vx=True).numpy()
    kbm = kinematic_trace(windows, wheelbase)
    return windows, gm3, kbm


def plot_holdout(run, model, wheelbase):
    windows, gm3, kbm = windowed(model, run, wheelbase)
    target = windows.target_states
    starts = np.array([np.searchsorted(run.t, run.t[0]) for _ in range(1)])  # placeholder
    # Window start times: match initial states back to the run by position index.
    idx = np.array([int(np.argmin(np.abs(run.states[:, 0] - s[0]) + np.abs(run.states[:, 1] - s[1])))
                    for s in windows.initial_state])
    t_win = run.t[idx[:, None] + np.arange(target.shape[0])[None, :]]  # [B, T+1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(run.t, np.degrees(run.states[:, 5]), color=SERIES["measured"], lw=1.2, label="measured (gyro)")
    for name, trace in (("kinematic bicycle", kbm), ("GM3 fitted", gm3)):
        for b in range(target.shape[1]):
            ax1.plot(t_win[b], np.degrees(trace[:, b, 5]), color=SERIES[name], lw=1.6,
                     label=name if b == 0 else None)
    ax1.set_ylabel("yaw rate, deg/s")
    ax1.set_title(f"Held-out run {run.name[-6:]}: 3 s open-loop windows from measured initial state", fontsize=10)
    ax1.legend(loc="upper right", ncol=3)

    for name, trace, offset in (("kinematic bicycle", kbm, -0.25), ("GM3 fitted", gm3, 0.25)):
        err = np.degrees(np.abs(trace[-1, :, 2] - target[-1, :, 2]))
        ax2.bar(t_win[:, -1] + offset * 0.6, err, width=0.55, color=SERIES[name], label=name, linewidth=0)
    ax2.set_ylabel("|heading error| at 3 s, deg")
    ax2.set_xlabel("time, s")
    ax2.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "scooter_holdout.png", dpi=150)
    plt.close(fig)


def plot_errors(runs, model, wheelbase, holdout):
    rows = []
    for run in runs:
        windows, gm3, kbm = windowed(model, run, wheelbase)
        rows.append((run.name[-6:] + ("*" if run.name == holdout else ""),
                     errors(kbm, windows.target_states), errors(gm3, windows.target_states)))
    gps = gps_table({"kinematic bicycle": None, "GM3 fitted": model}, runs, wheelbase)
    for b, row in enumerate(rows):
        row[1]["gps_ADE"] = gps["kinematic bicycle"][b]["ADE"]
        row[2]["gps_ADE"] = gps["GM3 fitted"][b]["ADE"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(rows))
    for ax, key, label in zip(axes, ("r_rmse_dps", "psi_end_deg", "gps_ADE"),
                              ("yaw-rate RMSE, 3 s windows, deg/s", "median |heading error| at 3 s, deg",
                               "whole-run ADE to GPS fixes, m")):
        for name, offset, pick in (("kinematic bicycle", -0.2, 1), ("GM3 fitted", 0.2, 2)):
            values = [row[pick][key] for row in rows]
            ax.bar(x + offset, values, width=0.36, color=SERIES[name], label=name, linewidth=0)
        ax.set_xticks(x, [row[0] for row in rows], rotation=45, ha="right")
        ax.set_title(label, fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93), fontsize=9)
    fig.suptitle("Per-run error (* = held out; others were training runs)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(PLOTS / "scooter_errors.png", dpi=150)
    plt.close(fig)


def plot_residual(runs, model, wheelbase):
    angle, err = [], {"kinematic bicycle": [], "GM3 fitted": []}
    for run in runs:
        windows, gm3, kbm = windowed(model, run, wheelbase)
        target = windows.target_states
        delta = np.degrees(np.abs(STEER_SIGN * STEER_RATIO * windows.controls[:, :, 1]))
        angle.append(delta.ravel())
        err["kinematic bicycle"].append(np.degrees(kbm[1:, :, 5] - target[1:, :, 5]).ravel())
        err["GM3 fitted"].append(np.degrees(gm3[1:, :, 5] - target[1:, :, 5]).ravel())
    angle = np.concatenate(angle)
    edges = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60])
    positions = np.arange(len(edges) - 1)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, values, marker in (("kinematic bicycle", err["kinematic bicycle"], "o"), ("GM3 fitted", err["GM3 fitted"], "s")):
        values = np.concatenate(values)
        rmse = [np.sqrt(np.mean(values[(angle >= lo) & (angle < hi)] ** 2)) for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(positions, rmse, color=SERIES[name], lw=2, marker=marker, ms=5, label=name,
                markeredgecolor="white", markeredgewidth=1)
    counts = [int(((angle >= lo) & (angle < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    ax.set_xticks(positions, [f"{lo}-{hi}\nn={n}" for lo, hi, n in zip(edges[:-1], edges[1:], counts)], fontsize=8)
    ax.set_xlabel("|column steering angle|, deg (samples per bin)")
    ax.set_ylabel("yaw-rate RMSE, deg/s")
    ax.set_ylim(0, None)
    ax.set_title("Yaw-rate residual by steering angle, all runs (the two models coincide)", fontsize=10)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS / "scooter_residual.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="GM3 vs kinematic bicycle plots for the scooter logs.")
    parser.add_argument("--params", default=None, help="calibration JSON (default: out/scooter_calibration[_tag].json)")
    parser.add_argument("--root", default=None, help="data folder (default: the one recorded in the JSON, else the loader default)")
    parser.add_argument("--tag", default="", help="reads out/scooter_calibration_<tag>.json, writes out/plots/<tag>/")
    args = parser.parse_args()

    global PLOTS
    torch.set_default_dtype(torch.float64)
    suffix = f"_{args.tag}" if args.tag else ""
    params = Path(args.params) if args.params else OUT / f"scooter_calibration{suffix}.json"
    meta = json.loads(params.read_text())
    root = args.root or (meta.get("root") if meta.get("root") not in (None, "None") else None)
    runs = load_all(root, slope_source=meta.get("slope", "imu"))
    if args.tag:
        PLOTS = OUT / "plots" / args.tag
    holdout = meta["holdout"]
    model = fitted_model(runs[0].dt, params)
    wheelbase = make_scooter_config().wheelbase
    PLOTS.mkdir(parents=True, exist_ok=True)

    plot_holdout(next(r for r in runs if r.name == holdout), model, wheelbase)
    plot_errors(runs, model, wheelbase, holdout)
    plot_residual(runs, model, wheelbase)
    plot_paths(runs, model, wheelbase, holdout)
    for name in ("scooter_holdout", "scooter_errors", "scooter_residual", "scooter_paths"):
        print(f"wrote {PLOTS / name}.png")


if __name__ == "__main__":
    main()
