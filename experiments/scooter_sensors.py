"""GPS-anchored calibration of the scooter's sensor constants.

    python -m experiments.scooter_sensors
    python -m experiments.scooter_sensors --steps 400 --per-run-scale

No tire model is involved. Two kinematic paths are rolled out per run and
compared with the GPS fixes after a rigid alignment that is part of the loss:

* the dead-reckoned path (gyro heading, wheel speed), which identifies the
  wheel-speed scale and a per-run yaw-rate bias;
* the kinematic-bicycle path (steering heading), which additionally
  identifies the wheelbase for the measured gear ratio and a per-run steering
  zero.

Both share the speed scale. The loss is a Huber norm of the distance to each
fix, so a few bad fixes do not steer the answer. Alignment is a differentiable
2-D Procrustes step (SVD), so the unobservable initial heading and origin are
never fitted. The result is a check on the loader's constants
(``WHEEL_SPEED_SCALE``, ``GYRO_Z_BIAS``, the preset wheelbase and the per-run
zeros the loader estimates); it is not applied automatically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments import _bootstrap  # noqa: F401  (registers the `gm3` package)

from gm3.dataset.scooter import GYRO_Z_BIAS, STEER_RATIO, STEER_SIGN, WHEEL_SPEED_SCALE, load_all
from gm3.shared import make_scooter_config

OUT = Path(__file__).resolve().parent / "out"


def align_loss(path: torch.Tensor, path_t: np.ndarray, gps_t: np.ndarray, gps_xy: torch.Tensor,
               delta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Huber loss of the rigidly aligned path against the fixes, and the raw distances."""
    idx = np.clip(np.searchsorted(path_t, gps_t) - 1, 0, len(path_t) - 2)
    w = torch.as_tensor((gps_t - path_t[idx]) / (path_t[idx + 1] - path_t[idx]))[:, None]
    at_fix = path[idx] * (1 - w) + path[idx + 1] * w
    p0, g0 = at_fix - at_fix.mean(0), gps_xy - gps_xy.mean(0)
    u, _, vh = torch.linalg.svd(p0.T @ g0)
    d = torch.sign(torch.det(u @ vh)).detach()
    rotation = u @ torch.diag(torch.stack([torch.ones((), dtype=d.dtype), d])) @ vh
    distance = torch.linalg.norm(p0 @ rotation - g0, dim=1)
    huber = torch.where(distance < delta, 0.5 * distance ** 2, delta * (distance - 0.5 * delta))
    return huber.mean(), distance


def integrate(vx: torch.Tensor, r: torch.Tensor, dt: float, lr: float) -> torch.Tensor:
    psi = torch.cumsum(r, 0) * dt
    vy = lr * r
    xd = vx * torch.cos(psi) - vy * torch.sin(psi)
    yd = vx * torch.sin(psi) + vy * torch.cos(psi)
    return torch.stack([torch.cumsum(xd, 0), torch.cumsum(yd, 0)], 1) * dt


def main() -> None:
    parser = argparse.ArgumentParser(description="GPS-anchored calibration of the scooter sensor constants.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--per-run-scale", action="store_true", help="let the speed scale differ per run")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    # Load with the loader's corrections switched off so the fit sees raw sensors.
    runs = load_all(args.root, slope_source="none", speed_scale=1.0, gyro_bias=0.0)
    config = make_scooter_config()
    n = len(runs)
    log_scale = torch.nn.Parameter(torch.full((n if args.per_run_scale else 1,), math.log(WHEEL_SPEED_SCALE)))
    bias = torch.nn.Parameter(torch.zeros(n))                    # rad/s added to r (= -gyro bias)
    log_wheelbase = torch.nn.Parameter(torch.tensor(math.log(config.wheelbase)))
    zero = torch.nn.Parameter(torch.zeros(n))                    # residual motor-rad zero per run
    groups = [{"params": [log_scale, log_wheelbase], "lr": args.lr},
              {"params": [bias], "lr": args.lr * 0.02}, {"params": [zero], "lr": args.lr * 0.1}]
    optimizer = torch.optim.Adam(groups)

    data = []
    for run in runs:
        data.append((torch.as_tensor(run.states[:, 3]), torch.as_tensor(run.states[:, 5]),
                     torch.as_tensor(run.controls[:, 1]), torch.as_tensor(run.gps_xy), run))

    def evaluate():
        total, dr_dist, kbm_dist = 0.0, [], []
        for i, (v_front, r_meas, theta, gps, run) in enumerate(data):
            scale = log_scale[i if args.per_run_scale else 0].exp()
            r = r_meas + bias[i]
            delta = STEER_SIGN * STEER_RATIO * (theta - zero[i])
            vx = scale * v_front * torch.cos(delta)
            dr = integrate(vx, r, run.dt, config.lr)
            r_kbm = vx * torch.tan(delta) / log_wheelbase.exp()
            kbm = integrate(vx, r_kbm, run.dt, config.lr)
            l1, d1 = align_loss(dr, run.t, run.gps_t, gps)
            l2, d2 = align_loss(kbm, run.t, run.gps_t, gps)
            total = total + (l1 + l2) * len(run.gps_t)
            dr_dist.append(d1.detach()); kbm_dist.append(d2.detach())
        weight = sum(len(run.gps_t) for run in runs)
        return total / weight, torch.cat(dr_dist), torch.cat(kbm_dist)

    with torch.no_grad():
        loss0, dr0, kbm0 = evaluate()
    print(f"start: scale {WHEEL_SPEED_SCALE:.3f}, wheelbase {config.wheelbase:.3f} m, no bias, loader zeros;  "
          f"ADE to GPS: dead reckoning {dr0.mean():.3f} m, kinematic bicycle {kbm0.mean():.3f} m")
    for step in range(args.steps):
        optimizer.zero_grad()
        loss, _, _ = evaluate()
        loss.backward()
        optimizer.step()
        if step % max(args.steps // 5, 1) == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}  scale {log_scale.exp().mean().item():.4f}  "
                  f"wheelbase {log_wheelbase.exp().item():.3f} m  mean bias {math.degrees(bias.mean().item()):+.3f} deg/s")
    with torch.no_grad():
        _, dr1, kbm1 = evaluate()
    print(f"\nfitted: ADE to GPS: dead reckoning {dr1.mean():.3f} m, kinematic bicycle {kbm1.mean():.3f} m")
    print(f"  wheel-speed scale {log_scale.exp().mean().item():.4f}  (loader {WHEEL_SPEED_SCALE})"
          + ("  per run: " + " ".join(f"{s:.3f}" for s in log_scale.exp().tolist()) if args.per_run_scale else ""))
    print(f"  wheelbase {log_wheelbase.exp().item():.3f} m for ratio {STEER_RATIO:.4f}  (preset {config.wheelbase:.3f})")
    # r = -(gyro - b_gyro) = -gyro + b_gyro, and the fit adds ``bias`` to -gyro, so bias IS the gyro bias.
    print(f"  gyro-z bias per run, deg/s, IMU frame: "
          + " ".join(f"{run.name[-6:]} {math.degrees(b):+.3f}" for run, b in zip(runs, bias.tolist()))
          + f"   (loader {math.degrees(GYRO_Z_BIAS):+.3f}, stationary log -0.085)")
    print(f"  residual steer zero per run, motor deg: " + " ".join(f"{math.degrees(z):+.2f}" for z in zero.tolist()))
    OUT.mkdir(exist_ok=True)
    (OUT / "scooter_sensors.json").write_text(json.dumps({
        "wheel_speed_scale": log_scale.exp().tolist(), "wheelbase": log_wheelbase.exp().item(),
        "gyro_z_bias_deg_s": {run.name: math.degrees(b) for run, b in zip(runs, bias.tolist())},
        "steer_zero_residual_deg": {run.name: math.degrees(z) for run, z in zip(runs, zero.tolist())},
        "ade_dead_reckoning": [float(dr0.mean()), float(dr1.mean())],
        "ade_kinematic_bicycle": [float(kbm0.mean()), float(kbm1.mean())]}, indent=2))
    print(f"  wrote {OUT / 'scooter_sensors.json'}")


if __name__ == "__main__":
    main()
