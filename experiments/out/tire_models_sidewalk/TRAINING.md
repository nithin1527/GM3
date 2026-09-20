# How the tire-law parameters were trained

Code: `experiments/scooter_tire_models.py` (driver), `experiments/scooter_calibrate.py` (`ScooterModel`, `fit`,
`yaw_loss`), `dataset/scooter.py` (loader), `diffgm3/tire_models.py` (the laws). Per-fit logs are in `logs/`,
full results in `<law>.json` (held-out fit) and `<law>_all.json` (all-data fit), weights in the `.pt` files.

## 1. Data

- Source: `data/scooter_data_kim_quad_corrected` (9 usable recordings, concrete) and
  `data/scooter_data_slope_sidewalk_corrected` (11 usable recordings, sidewalk with grade), pooled into one
  training set. Recordings with under 5 s of motion are dropped by the loader.
- The corrected CSVs already carry every sensor correction: wheel-speed scale 0.545, forward sign, gyro-z bias
  -0.085 deg/s, IMU axes rotated into the vehicle frame, per-recording steering-encoder zero, column angle from
  the 3.75/10.9 gear ratio, body speed `vx = v_wheel cos(delta)`.
- Resampled to a uniform 30 Hz grid; wheel speed smoothed with a 3-sample moving average.
- State per sample `[x, y, psi, vx, vy, r, gamma, gamma_dot]`: `r` is the gyro yaw rate, `psi` its integral,
  `x, y` wheel-plus-gyro dead reckoning, `vy` seeded with the no-slip value `lr * r`, lean states zero.
- Inputs per sample: steering motor angle (mapped to column angle inside the model), wheel angular speed
  (ignored, motor off), and the surface grade and bank from the IMU (`grade_deg`, `bank_deg` columns).
- Windows: only samples where the 15-sample-smoothed speed exceeds 0.3 m/s; each contiguous moving stretch is
  cut into non-overlapping windows of 45 steps (1.5 s). 617 windows from all 20 recordings; 547 when the last
  recording of each folder (163726, 191928) is held out.

## 2. Model being fitted

`ScooterModel` = DiffGM3 with the Hiboy scooter preset (single track, wheelbase 0.95 m, lf = lr = 0.475 m,
85 kg, wheel radius 0.108 m, rolling resistance 0.02, both wheels free-rolling) plus a trainable steering map
`delta = -ratio * (theta_motor - steer_zero)`. The tire law is swapped by name; everything else is identical
across the five fits.

Rollout of one window: start from the measured state, integrate the model's own derivative with explicit Euler,
and after every 30 Hz step overwrite `vx` with the measured value (the rider's kicks set speed, which the model
does not represent). Heading, lateral velocity and yaw rate are free. Each 30 Hz step is split into 129 Euler
substeps: `3 x stable_substeps`, where `stable_substeps` keeps `h * lambda <= 1` for the stiffest yaw/sideslip
mode at the preset stiffness and the slowest sample (clamped to 0.2 m/s). The count is frozen for the whole
fit so the loss is smooth in the parameters.

## 3. What was trained and what was held

Trained, four quantities per law:

| quantity | parametrization | start |
|---|---|---|
| tire stiffness: brush `cp`; Fiala / Dugoff / Pacejka `cy`; Burckhardt `c2` | `cp`: log-scaled logit between 1 and 1e8 N/m²; `cy`: log-scaled logit between 1 and 1e7 N/rad; `c2`: log | 4,500 N/rad zero-slip stiffness for every law |
| `yaw_inertia` | softplus | 8.0 kg m² |
| `yaw_damping` | softplus | 0.5 |
| residual steering zero | raw, motor radians | 0 |

Burckhardt has no stiffness parameter; its zero-slip stiffness is `(c1 c2 - c3) Fz`, so `c1 = 1.280` and
`c3 = 0.520` (dry asphalt) are held, the starting `c2 = 8.84` is chosen to give 4,500 N/rad, and a gradient
mask `(0, 1, 0)` lets only `c2` move. Front and rear tire are tied: after backprop both entries of a tire
parameter get the sum of their gradients, so they move identically.

Held at the preset: `mu = 0.8`, brush half-length `a = 0.025 m` and `align_gain = 0.1`, longitudinal stiffness
`cx = 4,500`, Pacejka `C_y = 1.30`, `E_y = -0.20`, steering ratio 0.344, all geometry and mass.

## 4. Loss

Over all windows and all 46 time points:

`loss = mean(((r_sim - r_meas) / sigma_r)^2) + mean(((psi_sim - psi_meas) / sigma_psi)^2)`

with `sigma_r` the standard deviation of the measured yaw rate over the batch and `sigma_psi` the standard
deviation of the measured heading change from each window's start. Position and speed are not in the loss.

## 5. Optimizer

- Adam, PyTorch defaults (betas 0.9 / 0.999, eps 1e-8), learning rate 0.01, no schedule, no weight decay.
- Learning rate scaled by 0.02 for the steering zero (it is in motor radians, so 0.01 would be 0.6 deg a step).
- 150 steps, full batch (every window in every step), one rollout + backprop through the whole Euler
  integration per step (45 x 129 = 5,805 derivative calls).
- Gradient-norm clipping at 10.
- float64, seed 0, CPU, one thread. No early stopping; the parameters at step 150 are the result.
- Cost: about 20-28 minutes and 5 GB of memory per fit.

## 6. The two fits per law

| | training recordings | windows | used for |
|---|---|---|---|
| held-out fit (`<law>.json`) | 18 (163726 and 191928 excluded) | 547 | every accuracy number, profiles, plots of paths |
| all-data fit (`<law>_all.json`) | all 20 | 617 | the reported parameters |

Both start from the same preset and use identical settings.

## 7. Training trajectories (all-data fits, from `logs/<law>_all.log`)

| step | brush C_alpha / Jz / loss x1e-3 | pacejka | burckhardt |
|---|---|---|---|
| 0 | 4,624 / 7.99 / 24.86 | 4,685 / 7.99 / 24.72 | 4,545 / 7.99 / 25.18 |
| 30 | 8,511 / 7.74 / 24.00 | 11,665 / 7.75 / 23.87 | 6,080 / 7.71 / 24.46 |
| 60 | 11,199 / 7.56 / 23.87 | 17,063 / 7.58 / 23.79 | 7,572 / 7.49 / 24.16 |
| 90 | 13,121 / 7.39 / 23.82 | 20,940 / 7.41 / 23.76 | 8,926 / 7.32 / 24.02 |
| 120 | 14,731 / 7.23 / 23.79 | 24,162 / 7.24 / 23.75 | 10,189 / 7.16 / 23.93 |
| 149 | 16,113 / 7.06 / 23.77 | 26,842 / 7.06 / 23.73 | 11,350 / 7.02 / 23.88 |

Fiala and Dugoff follow Pacejka closely. Nothing has converged at step 150: stiffness is still rising and
inertia and damping still falling for every law, at a shrinking loss gain (0.02e-3 over the last 30 steps for
Pacejka). That matches the loss profile: the loss is flat above about 10,000 N/rad, so the stiffness is a lower
bound and its final value reflects where 150 Adam steps reached in each law's coordinate, which is why `cy`
laws end near 26,000, the brush at 16,000 and Burckhardt at 11,000. Inertia and damping move at the same rate
in every law because they share a parametrization.

## 8. Evaluation after training

- Yaw loss on train and held-out windows (1.5 s), before and after.
- 3 s windows: yaw-rate RMSE, heading error at the window end, ADE / FDE against the dead-reckoned path.
- Whole recordings rolled out once, rigidly aligned to the ~1 Hz GPS fixes: ADE and discrete Frechet distance.
- Steady-state yaw gain at 1.5 m/s against the measured kinematic gain.
- `profile`: loss re-evaluated at fixed stiffness values 1,000-30,000 N/rad with the other parameters at the fit.
- Baseline throughout: kinematic bicycle `r = vx tan(delta) / L` on the same inputs.

## 9. Reproduce

```bash
python -m experiments.scooter_tire_models fit --model brush --surfaces kim_quad slope_sidewalk --threads 1
python -m experiments.scooter_tire_models fit --model brush --all-data --surfaces kim_quad slope_sidewalk --threads 1
python -m experiments.scooter_tire_models profile --model brush --surfaces kim_quad slope_sidewalk
python -m experiments.scooter_tire_models report --surfaces kim_quad slope_sidewalk
python -m experiments.scooter_tire_model_plots --surfaces kim_quad slope_sidewalk
```

Repeat the first three for `fiala`, `dugoff`, `burckhardt`, `pacejka`.
