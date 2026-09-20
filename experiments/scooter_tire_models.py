"""Calibrate every DiffGM3 tire law on all the corrected scooter logs and compare them.

    python -m experiments.scooter_tire_models fit --model fiala            # holds out one run per surface
    python -m experiments.scooter_tire_models fit --model fiala --all-data # the reported parameters
    python -m experiments.scooter_tire_models profile --model fiala        # loss along the stiffness axis
    python -m experiments.scooter_tire_models report                       # tables from whatever has been fitted

One set of parameters per tire law (brush, fiala, dugoff, burckhardt,
pacejka), fitted on the three ``data/*_corrected`` folders pooled: concrete
(kim quad), asphalt (AV Williams lot) and the sloped sidewalk. ``--surfaces``
picks a subset (``--surfaces kim_quad slope_sidewalk``); the outputs then go
to ``out/tire_models_<surfaces>/`` so the pooled and subset fits can coexist.
The protocol is
``scooter_calibrate``'s, unchanged, so the brush row is comparable with
CALIBRATION.md: 1.5 s windows, vx pinned to the wheel speed, both wheels
free-rolling, IMU slope input, variance-normalized yaw-rate + heading loss,
Adam from the scooter preset, front and rear tire tied.

Every law starts from the same zero-slip cornering stiffness, the preset's
4,500 N/rad (Burckhardt by choosing c2, since its stiffness is
``(c1 c2 - c3) Fz``), and frees the same things: its stiffness parameter,
``yaw_inertia``, ``yaw_damping`` and the residual steering zero. Friction and
curve-shape parameters are held (``--free all`` frees them too, as a
sensitivity check): these logs never load a tire past about a third of
``mu Fz``, see CALIBRATION.md.

``fit`` without ``--all-data`` holds out the last run of each folder and is
where the accuracy numbers come from; ``--all-data`` refits on every run and
is where the reported parameters come from.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments import _bootstrap  # noqa: F401  (registers the `gm3` package)

from experiments.scooter_calibrate import (
    TIRE_FIT, ScooterModel, errors, fit, gps_table, kinematic_trace, loss_on, steady_state_gain, yaw_loss,
)
from gm3.dataset.scooter import ScooterRun, kinematic_steer_fit, load_all, windows_of
from gm3.diffgm3.tire_models import BURCKHARDT_ROADS
from gm3.shared import make_scooter_config

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parent / "out" / "tire_models"   # reassigned per --surfaces in main()
SURFACES = {"concrete": "scooter_data_kim_quad_corrected",
            "asphalt": "scooter_data_av_williams_lot_corrected",
            "sloped sidewalk": "scooter_data_slope_sidewalk_corrected"}
# --surfaces takes the data-folder slugs.
SURFACE_SLUGS = {folder.replace("scooter_data_", "").replace("_corrected", ""): surface
                 for surface, folder in SURFACES.items()}
MODELS = ("brush", "fiala", "dugoff", "burckhardt", "pacejka")
SHARED_FIT = ["yaw_inertia", "yaw_damping", "steer_zero"]
# What --free all adds to the stiffness parameter: everything else the law's lateral force reads.
SHAPE_FIT = {"brush": ["mu"], "fiala": ["mu"], "dugoff": ["mu"], "pacejka": ["mu", "shape_y", "curvature_y"],
             "burckhardt": ["burckhardt_y"]}


def select_surfaces(slugs: list[str] | None) -> dict[str, str]:
    """``SURFACES`` restricted to the ``--surfaces`` slugs, in table order."""
    if not slugs:
        return dict(SURFACES)
    chosen = {SURFACE_SLUGS[slug] for slug in slugs}
    return {surface: folder for surface, folder in SURFACES.items() if surface in chosen}


def output_dir(slugs: list[str] | None) -> Path:
    base = Path(__file__).resolve().parent / "out"
    if not slugs or set(slugs) == set(SURFACE_SLUGS):
        return base / "tire_models"
    return base / ("tire_models_" + "_".join(slug for slug in SURFACE_SLUGS if slug in set(slugs)))


def load_surfaces(slope: str = "imu", slugs: list[str] | None = None) -> dict[str, list[ScooterRun]]:
    return {surface: load_all(DATA / folder, slope_source=slope) for surface, folder in select_surfaces(slugs).items()}


def prepare(model: ScooterModel, device: str, compile_model: bool) -> ScooterModel:
    """Move to ``device`` and optionally CUDA-graph the integrator's inner call.

    The rollout is ``T * substeps`` sequential ``derivative`` calls on a tiny
    ``[B, 8]`` state, so it is bound by kernel-launch overhead, not arithmetic:
    plain CUDA measured *slower* than the CPU (1152 vs 577 ms/step) because
    consumer fp64 is slow and each launch still costs microseconds.
    ``torch.compile(mode="reduce-overhead")`` replays the whole derivative as a
    CUDA graph and is 5.3x faster than the CPU at an identical loss.
    """
    model = model.to(device)
    if compile_model:
        model.gm3.derivative = torch.compile(model.gm3.derivative, mode="reduce-overhead")
    return model


def make_model(name: str, dt: float) -> ScooterModel:
    if name != "burckhardt":
        return ScooterModel(dt=dt, tire_model=name)
    # Dry asphalt's c1 and c3, with c2 chosen so the curve starts from the same
    # cornering stiffness as the other laws instead of the road table's 12,600 N/rad.
    config = make_scooter_config()
    stiffness = 2.0 * config.tires[0].cp * config.tires[0].contact_length ** 2
    load = config.mass * config.gravity * config.lr / config.wheelbase
    c1, _, c3 = BURCKHARDT_ROADS["dry_asphalt"]
    coefficients = (c1, (stiffness / load + c3) / c1, c3)
    return ScooterModel(dt=dt, tire_model=name, coefficients_x=coefficients, coefficients_y=coefficients)


def fit_names(name: str, free: str) -> list[str]:
    names = [TIRE_FIT[name]] + SHARED_FIT
    if free == "all":
        names = [n for n in names if not (n == "burckhardt_c2" and "burckhardt_y" in SHAPE_FIT[name])]
        names += SHAPE_FIT[name]
    return names


def pooled_gps(table: list[dict[str, float]]) -> dict[str, float]:
    weights = np.array([row["n"] for row in table], dtype=float)
    return {"ADE": float(np.average([row["ADE"] for row in table], weights=weights)),
            "frechet": float(np.mean([row["frechet"] for row in table])), "fixes": int(weights.sum())}


def evaluate(model: ScooterModel | None, surfaces: dict[str, list[ScooterRun]], holdouts: set[str],
             horizon: int, eval_horizon: int) -> dict:
    """Every accuracy number for one model; ``None`` is the kinematic bicycle."""
    wheelbase = make_scooter_config().wheelbase

    def trace(windows):
        if model is None:
            return kinematic_trace(windows, wheelbase)
        with torch.no_grad():
            return model.simulate(windows, pin_vx=True).cpu().numpy()

    def loss(windows):
        return float(yaw_loss(torch.as_tensor(trace(windows)), torch.as_tensor(windows.target_states)))

    out: dict = {"loss": {}, "windows": {}, "gps": {}}
    # All three surfaces pooled: one train / held-out / all triple, no per-surface
    # groups. A tire's cornering stiffness is a property of the tire and its load,
    # not the ground, and CALIBRATION.md found the three surfaces agree to within
    # 10%, so splitting them multiplied the evaluation cost without adding a fact.
    groups = {"train": [r for runs in surfaces.values() for r in runs if r.name not in holdouts],
              "held-out": [r for runs in surfaces.values() for r in runs if r.name in holdouts],
              "all": [r for runs in surfaces.values() for r in runs]}
    for label, runs in groups.items():
        if not runs:
            continue
        out["loss"][label] = loss(windows_of(runs, horizon))
        windows = windows_of(runs, eval_horizon)
        out["windows"][label] = {**errors(trace(windows), windows.target_states), "n": int(windows.controls.shape[1])}

    every = [r for runs in surfaces.values() for r in runs]
    surface_of = {run.name: surface for surface, runs in surfaces.items() for run in runs}
    table = gps_table({"m": model}, every, wheelbase)["m"]
    out["gps"]["all"] = {**pooled_gps(table),
                         "runs": {run.name: {"ADE": row["ADE"], "frechet": row["frechet"], "n": row["n"],
                                             "surface": surface_of[run.name]}
                                  for run, row in zip(every, table)}}
    gps_held = [row for run, row in zip(every, table) if run.name in holdouts]
    if gps_held:
        out["gps"]["held-out"] = pooled_gps(gps_held)
    if len(surfaces) > 1:
        for surface in surfaces:
            rows = [row for run, row in zip(every, table) if surface_of[run.name] == surface]
            if rows:
                out["gps"][surface] = pooled_gps(rows)
    if model is not None:
        out["steady_state_gain"] = steady_state_gain(model)
    return out


def command_fit(args: argparse.Namespace) -> None:
    surfaces = load_surfaces(args.slope, args.surfaces)
    holdouts = set() if args.all_data else {runs[-1].name for runs in surfaces.values()}
    every = [run for runs in surfaces.values() for run in runs]
    train = windows_of([run for run in every if run.name not in holdouts], args.horizon)
    names = fit_names(args.model, args.free)
    tag = f"{args.model}{'_all' if args.all_data else ''}{'_free' if args.free == 'all' else ''}"
    print(f"[{tag}] {len(every)} runs, {train.controls.shape[1]} training windows of {args.horizon / 30:.1f} s, "
          f"held out {sorted(holdouts) or 'nothing'}, fitting {names}, {args.steps} steps", flush=True)

    model = prepare(make_model(args.model, train.dt), args.device, args.compile)
    preset_values = model.values()
    preset = evaluate(model, surfaces, holdouts, args.horizon, args.eval_horizon)
    start = time.time()
    fit(model, train, names, args.steps, args.lr, pin_vx=True)
    seconds = time.time() - start
    fitted = evaluate(model, surfaces, holdouts, args.horizon, args.eval_horizon)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "all_data": args.all_data, "free": args.free, "fit": names,
               "surfaces": {surface: [run.name for run in runs] for surface, runs in surfaces.items()},
               "holdouts": sorted(holdouts), "steps": args.steps, "lr": args.lr, "fit_seconds": seconds,
               "training_windows": int(train.controls.shape[1]),
               "kinematic_gain": abs(kinematic_steer_fit(every)["gain"]),
               "preset_parameters": preset_values, "parameters": model.values(),
               "preset": preset, "fitted": fitted,
               "kinematic": evaluate(None, surfaces, holdouts, args.horizon, args.eval_horizon)}
    (OUT / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}.pt")
    held = "held-out" if holdouts else "all"
    print(f"[{tag}] {seconds / 60:.1f} min; loss train {preset['loss']['train']:.4e} -> {fitted['loss']['train']:.4e}, "
          f"{held} {preset['loss'][held]:.4e} -> {fitted['loss'][held]:.4e}; C_alpha {model.values()['C_alpha']:.0f}", flush=True)


def command_profile(args: argparse.Namespace) -> None:
    """Loss along the cornering-stiffness axis with everything else at the fit."""
    surfaces = load_surfaces(args.slope, args.surfaces)
    holdouts = {runs[-1].name for runs in surfaces.values()}
    every = [run for runs in surfaces.values() for run in runs]
    groups = {"train": windows_of([r for r in every if r.name not in holdouts], args.horizon),
              "held-out": windows_of([r for r in every if r.name in holdouts], args.horizon)}
    model = prepare(make_model(args.model, groups["train"].dt), args.device, args.compile)
    model.load_state_dict(torch.load(OUT / f"{args.model}.pt", map_location=args.device))
    fitted = model.values()["C_alpha"]
    parameter = dict(model.named_parameters())[{"brush": "gm3.raw_cp", "burckhardt": "gm3.tire_model.log_coefficients_y"}
                                               .get(args.model, "gm3.tire_model.raw_cy")]
    rows = []
    for target in args.values:
        # Bisect the raw coordinate: every law's stiffness is monotone in it.
        index = 1 if args.model == "burckhardt" else slice(None)
        low, high = -40.0, 40.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            with torch.no_grad():
                parameter[index] = middle
            low, high = (middle, high) if model.values()["C_alpha"] < target else (low, middle)
        with torch.no_grad():
            rows.append({"C_alpha": model.values()["C_alpha"],
                         **{label: float(loss_on(model, windows, True)) for label, windows in groups.items()}})
        print(f"[{args.model}] C_alpha {rows[-1]['C_alpha']:9.0f}  train {rows[-1]['train']:.4e}  "
              f"held-out {rows[-1]['held-out']:.4e}", flush=True)
    (OUT / f"{args.model}_profile.json").write_text(json.dumps({"model": args.model, "fitted_C_alpha": fitted,
                                                                 "profile": rows}, indent=2))


def load_summaries(suffix: str) -> dict[str, dict]:
    found = {}
    for name in MODELS:
        path = OUT / f"{name}{suffix}.json"
        if path.exists():
            found[name] = json.loads(path.read_text())
    return found


PARAMETER_ROWS = (
    ("C_alpha", "C_alpha, zero-slip cornering stiffness (N/rad)", "{:,.0f}"),
    ("cp", "cp (N/m^2)", "{:.3g}"), ("contact_length", "contact_length a (m)", "{:.3f}"),
    ("tire.cy", "cy (N/rad)", "{:,.0f}"), ("tire.cx", "cx (N per unit kappa)", "{:,.0f}"),
    ("tire.coefficients_y.c1", "Burckhardt c1", "{:.3f}"), ("tire.coefficients_y.c2", "Burckhardt c2", "{:.2f}"),
    ("tire.coefficients_y.c3", "Burckhardt c3", "{:.3f}"),
    ("tire.shape_y", "Pacejka C_y", "{:.2f}"), ("tire.curvature_y", "Pacejka E_y", "{:.2f}"),
    ("mu", "mu", "{:.2f}"), ("align_gain", "align_gain", "{:.2f}"),
    ("yaw_inertia", "yaw_inertia (kg m^2)", "{:.2f}"), ("yaw_damping", "yaw_damping (1/s)", "{:.3f}"),
    ("steer_zero_deg", "residual steer zero (deg motor)", "{:+.3f}"),
)
# Parameters each law's force actually reads; the rest of the framework's are inert for it.
USED = {"brush": {"cp", "contact_length", "mu", "align_gain"}, "fiala": {"tire.cy", "tire.cx", "mu"},
        "dugoff": {"tire.cy", "tire.cx", "mu"},
        "burckhardt": {"tire.coefficients_y.c1", "tire.coefficients_y.c2", "tire.coefficients_y.c3"},
        "pacejka": {"tire.cy", "tire.cx", "mu", "tire.shape_y", "tire.curvature_y"}}
FITTED_KEYS = {"cp": {"cp"}, "cy": {"tire.cy"}, "burckhardt_c2": {"tire.coefficients_y.c2"},
               "burckhardt_y": {"tire.coefficients_y.c1", "tire.coefficients_y.c2", "tire.coefficients_y.c3"},
               "mu": {"mu"}, "shape_y": {"tire.shape_y"}, "curvature_y": {"tire.curvature_y"},
               "yaw_inertia": {"yaw_inertia"}, "yaw_damping": {"yaw_damping"}, "steer_zero": {"steer_zero_deg"}}


def parameter_table(summaries: dict[str, dict]) -> list[str]:
    names = list(summaries)
    lines = ["| parameter | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, form in PARAMETER_ROWS:
        cells = []
        for name in names:
            values, summary = summaries[name]["parameters"], summaries[name]
            fitted = set().union(*(FITTED_KEYS[n] for n in summary["fit"]))
            always = key in ("C_alpha", "yaw_inertia", "yaw_damping", "steer_zero_deg")
            if key not in values or not (always or key in USED[name]):
                cells.append("-")
            elif key in fitted or key == "C_alpha":
                cells.append(f"**{form.format(values[key])}**")
            else:
                cells.append(f"{form.format(values[key])} (held)")
        if any(cell != "-" for cell in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def accuracy_table(summaries: dict[str, dict], group: str, gps_group: str) -> list[str]:
    first = next(iter(summaries.values()))
    rows = [("kinematic bicycle", first["kinematic"])]
    rows += [(f"{name} preset", s["preset"]) for name, s in summaries.items() if name == "brush"]
    rows += [(f"{name} fitted", s["fitted"]) for name, s in summaries.items()]
    lines = ["| model | yaw loss (1.5 s) | r RMSE (deg/s) | psi @ 3 s med / p90 (deg) | ADE (m) | FDE (m) | GPS ADE / Frechet (m) |",
             "|---|---|---|---|---|---|---|"]
    for label, result in rows:
        w, g = result["windows"][group], result["gps"][gps_group]
        lines.append(f"| {label} | {result['loss'][group]:.4e} | {w['r_rmse_dps']:.3f} | {w['psi_end_deg']:.2f} / "
                     f"{w['psi_end_p90_deg']:.2f} | {w['ADE_m']:.4f} | {w['FDE_m']:.4f} | {g['ADE']:.3f} / {g['frechet']:.3f} |")
    return lines


def command_report(args: argparse.Namespace) -> None:
    held, everything = load_summaries(""), load_summaries("_all")
    free = load_summaries("_all_free")
    lines: list[str] = []
    if everything:
        first = next(iter(everything.values()))
        runs = sum(len(names) for names in first.get("surfaces", {}).values()) or 26
        surfaces = ", ".join(first.get("surfaces", {})) or "all surfaces"
        lines += [f"## Parameters, fitted on all {runs} corrected runs ({surfaces}; bold = fitted)", ""]
        lines += parameter_table(everything) + [""]
    if free:
        lines += ["## Sensitivity: friction and shape parameters freed as well", ""] + parameter_table(free) + [""]
    if held:
        n = next(iter(held.values()))
        lines += [f"## Accuracy on the held-out runs ({', '.join(n['holdouts'])})", ""]
        lines += accuracy_table(held, "held-out", "held-out") + [""]
        lines += ["## Accuracy on the training runs of the same fits", ""] + accuracy_table(held, "train", "all") + [""]
        surfaces = [s for s in n["fitted"]["gps"] if s not in ("all", "held-out")]
        if surfaces:
            lines += ["## Whole-run GPS ADE / Frechet (m) by surface, held-out fits, every run", "",
                      "| model | " + " | ".join(surfaces) + " | all |", "|---|" + "---|" * (len(surfaces) + 1)]
            rows = [("kinematic bicycle", n["kinematic"]), ("brush preset", held["brush"]["preset"])] if "brush" in held \
                else [("kinematic bicycle", n["kinematic"])]
            rows += [(f"{name} fitted", s["fitted"]) for name, s in held.items()]
            for label, result in rows:
                lines.append(f"| {label} | " + " | ".join(f"{result['gps'][s]['ADE']:.3f} / {result['gps'][s]['frechet']:.3f}"
                                                          for s in surfaces + ["all"]) + " |")
            lines.append("")
        lines += ["## Generalization and parameter stability", "",
                  "| model | train loss preset -> fitted | held-out loss preset -> fitted | C_alpha held-out fit / all-data fit | "
                  "Jz | yaw gain ratio | fit time (min) |", "|---|---|---|---|---|---|---|"]
        for name, s in held.items():
            other = everything.get(name, {}).get("parameters", {})
            lines.append(
                f"| {name} | {s['preset']['loss']['train']:.4e} -> {s['fitted']['loss']['train']:.4e} | "
                f"{s['preset']['loss']['held-out']:.4e} -> {s['fitted']['loss']['held-out']:.4e} | "
                f"{s['parameters']['C_alpha']:,.0f} / {other.get('C_alpha', float('nan')):,.0f} | "
                f"{s['parameters']['yaw_inertia']:.2f} / {other.get('yaw_inertia', float('nan')):.2f} | "
                f"{s['fitted']['steady_state_gain'] / s['kinematic_gain']:.4f} | {s['fit_seconds'] / 60:.0f} |")
        lines.append("")
    profiles = {name: json.loads((OUT / f"{name}_profile.json").read_text()) for name in MODELS
                if (OUT / f"{name}_profile.json").exists()}
    if profiles:
        targets = [row["C_alpha"] for row in next(iter(profiles.values()))["profile"]]
        lines += ["## Loss along the cornering-stiffness axis (train / held-out, x 1e-3, other parameters at the fit)", "",
                  "| model | " + " | ".join(f"{t:,.0f}" for t in targets) + " |", "|---|" + "---|" * len(targets)]
        for name, profile in profiles.items():
            lines.append(f"| {name} | " + " | ".join(f"{1e3 * r['train']:.2f} / {1e3 * r['held-out']:.2f}"
                                                     for r in profile["profile"]) + " |")
        lines.append("")
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=MODELS)
        p.add_argument("--horizon", type=int, default=45, help="training horizon, steps at 30 Hz")
        p.add_argument("--slope", default="imu", choices=("imu", "plane", "none"))
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
        p.add_argument("--compile", action="store_true",
                       help="CUDA-graph the integrator (5.3x on this GPU; see prepare())")
    p = sub.choices["fit"]
    p.add_argument("--all-data", action="store_true", help="no held-out runs: the fit the reported parameters come from")
    p.add_argument("--free", default="identifiable", choices=("identifiable", "all"))
    p.add_argument("--eval-horizon", type=int, default=90)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.01)
    sub.choices["profile"].add_argument("--values", type=float, nargs="+",
                                        default=[1000, 2000, 4500, 7000, 10000, 15000, 20000, 30000])
    sub.add_parser("report")
    for p in sub.choices.values():
        p.add_argument("--surfaces", nargs="+", default=None, choices=sorted(SURFACE_SLUGS),
                       help="data folders to pool (default: all three); outputs go to out/tire_models_<surfaces>/")
    args = parser.parse_args()

    global OUT
    OUT = output_dir(args.surfaces)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    if getattr(args, "threads", None):
        torch.set_num_threads(args.threads)
    {"fit": command_fit, "profile": command_profile, "report": command_report}[args.command](args)


if __name__ == "__main__":
    main()
