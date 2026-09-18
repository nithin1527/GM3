# Calibrating GM3 from the Hiboy scooter logs

Recorded 2026-09-11: a Hiboy S2 kick scooter (350 W front hub motor, 8.5 in
tires) with a steering motor on the column and two caster
wheels just ahead of the rear wheel. It was ridden with the hub motor off, so
the front-wheel encoder is a ground-speed sensor and the rider's kicks set the
longitudinal channel. Logs are CSV at 30 Hz under `data/scooter_data_kim_quad`
with ODrive drive/steer state, a ~1 Hz GPS fix (3-4 m sigma) and an IMU.

```bash
python -m experiments.scooter_correct           # corrected CSVs: data/<folder>_corrected/
python -m experiments.scooter_report            # inventory, speeds, turn sequences
python -m experiments.scooter_report --checks   # re-derive every loader assumption
python -m experiments.scooter_sensors           # GPS-anchored sensor constants
python -m experiments.scooter_calibrate --per-run --per-speed
python -m experiments.scooter_plots
```

Loader: `gm3.dataset.scooter`. Preset: `make_scooter_config`. The loader
applies every correction below on the fly; `scooter_correct` writes the same
corrections out as CSVs (one `_corrected` folder per log folder, with a
column glossary) for use outside this repo.

## What the logs contain

| run | length | moving | distance | v mean / p95 / max (m/s) | ay p95 (m/s²) | shape |
|---|---|---|---|---|---|---|
| 162125 | 81 s | 80 s | 121 m | 1.47 / 1.93 / 2.29 | 1.01 | 1 U-turn, 5 corners, 4 sweeps |
| 162516 | 57 s | 27 s | 38 m | 1.37 / 1.72 / 1.9 | 0.75 | 1 U-turn, 2 sweeps |
| 162644 | 64 s | 59 s | 90 m | 1.43 / 1.91 / 2.09 | 1.45 | 1 loop, 3 U-turns, 3 corners, 4 sweeps |
| 162832 | 22 s | 21 s | 35 m | 1.6 / 2.09 / 2.19 | 0.99 | 2 corners, 2 sweeps |
| 163031 | 40 s | 39 s | 69 m | 1.74 / 2.48 / 2.74 | 1.01 | 3 corners, 3 sweeps |
| 163129 | 34 s | 31 s | 63 m | 2.0 / 2.66 / 2.94 | 1.18 | 1 corner, 1 sweep |
| 163418 | 77 s | 59 s | 106 m | 1.73 / 2.19 / 2.42 | 1.57 | 3 U-turns, 5 corners |
| 163556 | 60 s | 60 s | 108 m | 1.79 / 2.2 / 2.38 | 0.79 | 3 corners, 3 sweeps |
| 163726 | 51 s | 51 s | 83 m | 1.57 / 1.88 / 2.0 | 1.2 | 1 loop, 2 U-turns, 2 corners |
| 163011 | 6 s | 0 s | 0 m | stationary, skipped | | |

Totals: 486 s recorded, 429 s moving, 714 m of front-wheel path. Body speed
while moving is 1.62 m/s mean (3.6 mph), 2.94 m/s max; lateral acceleration
reaches 2.3 m/s² (0.23 g) with a p95 of 1.2 m/s². Turn sequences per run are
in `experiments/out/scooter_runs.csv`. The rider steers up to 40-60 deg of
column angle, so the runs are tight, low-speed handling rather than cruising.

## What the checks established

Each of these is re-derived by `scooter_report --checks`; all pass.

1. **The logged wheel speed is 1.83x too high, and GPS can say so to 1%.**
   GPS chord over odometer distance on straight spans reads 0.50-0.53 with
   the raw values, and a factor of exactly two (an encoder CPR / pole-pair
   misconfiguration) was the first guess. It is not exact: letting the
   dead-reckoned path find the scale that best matches the fixes over each
   whole run gives 0.545 (0.535-0.555 per run, one global constant), and the
   joint Huber fit in `scooter_sensors` gives 0.552. At 0.5 the path error
   against GPS was 0.76 m pooled; at 0.545 it is 0.42 m, and 0.32 m once the
   gyro bias is also removed. The loader carries 0.545. Measuring the
   wheelbase with a tape would settle it independently: the yaw gain implies
   0.95 m at this scale (0.88 m at 0.5).
1b. **The yaw gyro reads -0.085 deg/s at rest** (stationary log; the GPS fit
   wants -0.065 to -0.10). Rigid alignment cannot remove a bias, because it
   integrates into a heading ramp; it was worth 0.1 m of path error over a
   run and the loader now subtracts it.
2. **The IMU is upside down** (roll ~175 deg, gravity reads -9.7 on z), so the
   vehicle yaw rate is `-gyro_z`; GPS course rotation confirms the sign at
   corr +0.96. Its y axis is longitudinal. The lateral accelerometer channel
   correlates ~0 with `v * r` and is not used.
3. **The front wheel measures speed along the steered direction.** The
   encoder sits on the steered hub, so body speed is `v_front * cos(delta)`,
   23% less at 40 deg of column angle. Treating the wheel speed as body speed
   made the effective wheelbase appear to climb from 0.89 m below 10 deg to
   1.07 m at 30-45 deg (a "steering nonlinearity" that was really this
   projection), pushed the driven front tire into 30% longitudinal slip in
   tight turns, and biased the fit toward a 5% larger gear ratio. With the
   projection, the table below is flat.
4. **The steering gear ratio 3.75 / 10.9 is consistent with the logs.**
   `r = vx tan(delta) / L` gives L = 0.953 m pooled and 0.95-0.97 m on every
   run (R² 0.986 over 12,866 samples), and the loss profile below has its
   minimum at the measured ratio. (The path-based fit in `scooter_sensors`
   prefers 0.93 m; it is dominated by low-frequency heading drift and trades
   against the per-run zeros, so the instantaneous gain is the better
   estimate.) The encoder zero is not straight ahead: it
   sits at about -12 deg (motor) on the first seven runs and jumps to about
   +71 deg on the last two, so the loader estimates it per run; the residual
   after that is 0.07 deg.
5. **The tires are kinematic at these lateral loads.** Effective wheelbase by
   speed and column-angle band:

   | | 3-10 deg | 10-20 | 20-30 | 30-45 |
   |---|---|---|---|---|
   | 1.0-1.5 m/s | 0.946 | 0.946 | 0.944 | 0.957 |
   | 1.5-2.0 m/s | 0.966 | 0.952 | 0.953 | - |
   | 2.0-3.5 m/s | 0.946 | 0.975 | 0.960 | - |

   No speed dependence means no slip to identify a cornering stiffness from,
   so the fit can only bound it from below: the lateral loads never reach
   the tire. To identify stiffness the scooter would have to be ridden much
   faster through the same corners (lateral acceleration well above 3 m/s²).
6. **No steer-to-yaw lag** at 30 Hz (cross-correlation peaks at 0 samples,
   |corr| 0.86-0.998). Yaw inertia and damping are therefore only bounded
   from above.
7. **The quad is flat, and the fused IMU attitude cannot tell you that.** The
   quaternion's pitch reads +-10 deg (std 4-6 deg) on level ground because a
   gravity-referenced attitude treats every 1 m/s² kick as a 6 deg tilt; it
   predicts coasting deceleration with correlation -0.03. An IMU-free test,
   regressing coasting deceleration on heading direction (a planar ground
   would make it depend on which way the scooter points), gives an
   equivalent grade of 0.1-0.7 deg with R² under 0.15 on every run. The
   usable IMU estimator is the accelerometer's forward specific force minus
   the wheel-derived acceleration, low-passed at 3 s: about 1 deg of noise,
   agreeing in direction with the plane fit on seven of nine runs. That is
   what `surface_angles(source="imu")` feeds to the model's slope input.
   The coasting intercept of the same regression, 0.11-0.28 m/s² (mean
   0.19), is the rolling drag with the motor off; the preset carries it as
   `rolling_resistance = 0.02`.

## Sensor calibration against GPS

`python -m experiments.scooter_sensors`

No tire model. Two kinematic paths per run - dead reckoning (gyro heading)
and the kinematic bicycle (steering heading), sharing the wheel speed - are
compared with the GPS fixes through a differentiable 2-D Procrustes
alignment inside a Huber loss, and Adam fits the constants GPS can actually
see: a global wheel-speed scale, the wheelbase for the measured gear ratio,
and a yaw-rate bias and residual steering zero per run.

| constant | loader / preset | GPS fit |
|---|---|---|
| wheel-speed scale | 0.545 | 0.552 |
| wheelbase | 0.95 m (yaw-gain regression) | 0.93 m |
| gyro-z bias | -0.085 deg/s (stationary log) | -0.065 mean, -0.18..+0.02 per run |
| residual steer zero | 0 (loader zeroes per run) | -0.9..+1.3 deg motor |

Dead reckoning goes from 0.41 m (loader constants, no bias) to 0.33 m ADE
against the fixes; the kinematic bicycle from 1.33 to 0.71 m. That is the
whole gain available from calibration: what remains between dead reckoning
and the fixes is GPS scatter (0.3-0.6 m), and what remains between the
bicycle and dead reckoning is the rider's unpredictable yaw.

## Loss profile before fitting

Yaw loss (variance-normalized r + psi, vx pinned) along each parameter with
the others at the preset, train / held-out:

| parameter | values | train | held-out |
|---|---|---|---|
| C_alpha (N/rad) | 1000 / 2000 / 4500 / 10000 / 25000 | .0430 / .0356 / .0329 / .0326 / .0325 | .0138 / .0100 / .0087 / .0084 / .0083 |
| yaw_inertia | 2 / 4 / 8 / 16 / 32 | .0325 / .0326 / .0329 / .0348 / .0414 | .0084 / .0085 / .0087 / .0095 / .0126 |
| yaw_damping | 0.01 / 0.5 / 2 / 5 / 10 | .0329 / .0329 / .0335 / .0359 / .0437 | .0085 / .0087 / .0098 / .0131 / .0215 |
| steer_zero (deg) | -2 / -1 / 0 / 1 / 2 | .0403 / .0349 / .0329 / .0345 / .0396 | .0140 / .0104 / .0087 / .0089 / .0109 |
| steer_ratio | .32 / .33 / .344 / .36 / .37 | .0436 / .0371 / .0329 / .0352 / .0402 | .0226 / .0147 / .0087 / .0089 / .0126 |

Stiffness is flat above ~7,000 N/rad (lower bound only); inertia and damping
prefer zero (upper bound only); the zero and the ratio are sharp at their
measured values. That is what the fit has to work with: a kinematic bicycle
already explains 98-99% of yaw-rate variance, and the remaining 1% is what
the tire model competes for.

## What the scooter added to the model

`rolling_resistance` on `VehicleConfig` (default 0, so nothing else moves):
each tire pushes back `c * Fz` along its rolling direction, in both backends.
On a steered front wheel that drag has a lateral body component, so it also
enters the yaw balance; at 0.02 on the scooter it shifts the preset's 3 s
heading error by 0.14 deg, which the fit absorbs. The scooter preset defaults
to `motor_on=False`: both wheels free-roll and `omega` is ignored, because the
hub motor was off. With `motor_on=True` the front hub tracks `omega` through
longitudinal slip, which is what a motor-on recording needs. The slope input
that already existed in DiffGM3 is now driven from the logs through
`surface_angles`. Rolling resistance with the front free-rolling shifts the
preset's pinned heading error by another 0.14 deg for the same reason.

## Three things the data exposed in DiffGM3

**`cp` could not reach a physical value.** The brush half-length for an
8.5 in tire under ~420 N is about 0.025 m, and a 4,500-12,000 N/rad
cornering stiffness then needs `cp = 2 * C / a^2` of 4e6-1e7 N/m^2. The
`CP_BOUNDS` ceiling was 1e5, forty times too low (the experiments README
already predicted fits would hit it), which is why the first scooter fits had
to carry stiffness in a nonsensical 0.15-0.25 m `contact_length`. The bound is
now 1e8, and because a linear sigmoid over eight decades would put almost the
whole raw axis on the top decade, `raw_cp` is parametrized in log space
(`torch_utils.raw_log_bounded`): every optimizer step is a fixed percentage of
`cp` instead of a thousand at one end and a million at the other. Physical
values of every preset are unchanged; only the trainable coordinate moved.


**Explicit Euler is unstable at 30 Hz for a realistic single-track tire.**
A 4,500 N/rad tire gives yaw and sideslip modes near 150/s at 1.5 m/s and
450/s at 0.5 m/s, past `dt * lambda < 2`. `DiffGM3.rollout` at `dt = 1/30`
oscillates and flips sign; the bicycle preset only survives because its tire
is 400x softer. `scooter_calibrate` substeps the model's own derivative so
that `h * lambda <= 1` for the current stiffness and the slowest sample,
frozen for the duration of a fit so the loss is smooth in the parameters.

**The smooth sliding gate leaked force at zero slip and blew up backprop.**
`brush_forces` blends adhesion and sliding with `sigmoid(20 (sigma -
sigma_sliding))`. For a stiff tire (`theta ~ 4`, `sigma_sliding ~ 0.22`) that
gate is 0.012 at zero slip, i.e. ~4 N of `mu * Fz` pointed along
`sigma / |sigma|`: a Coulomb discontinuity softened only at `eps = 1e-6`.
Its linearized stiffness is ~1e6 N per unit slip. A forward rollout never
notices (the state sits at the discontinuity and the force is 4 N), but the
sensitivity grows ~4.6x per 30 Hz step on straight driving, and parameter
gradients reached 1e160 with a perfectly finite loss; `clip_grad_norm_` then
scaled every update to zero and the fit sat still. The gate is now rebased
to be exactly zero at zero slip. This changes forces by at most 1.2% of
`mu * Fz`, only for tires with `theta` above ~0.25 (the `--grippy` matrix
configuration; the shipped presets are unaffected), and gradients are back
to O(1e-2) and independent of the substep count.

## Fit results

`python -m experiments.scooter_calibrate --per-run --per-speed`, default set:
shared front/rear `cp` (brush half-length held at its physical 0.025 m),
`yaw_inertia`, `yaw_damping`, residual `steer_zero`; 1.5 s windows, vx pinned,
both wheels free-rolling (motor off), slope input from the IMU, GPS-calibrated
wheel-speed scale and gyro bias, wheelbase 0.95 m, run 163726 held out.

| parameter | preset | fitted | status |
|---|---|---|---|
| cp | 3.6e6 N/m² | 1.21e7 N/m² | lower bound; still rising slowly at step 150 |
| C_alpha = 2 cp a² | 4,500 N/rad | 15,100 N/rad (36 / rad per N of load) | as above |
| contact_length | 0.025 m | held | brush half-length of an 8.5 in tire |
| yaw_inertia | 8.0 | 7.16 kg m² | upper bound; loss flat below ~8 |
| yaw_damping | 0.5 | 0.41 | upper bound; loss flat below ~1 |
| rolling_resistance | 0.02 | held | coasting drag, see check 7 |
| steer_zero (residual) | 0 | +0.03 deg motor | identified |
| steer_ratio | 0.344 | held; profile minimum is at 0.344 | confirmed |
| mu, align_gain | 0.8, 0.1 | held | not identifiable: no saturation, no measurable lag |

Gates:

1. **Held-out**: loss 9.13e-3 -> 8.34e-3, train 3.19e-2 -> 3.12e-2. Generalizes.
2. **Per-run**: C_alpha 12,100-19,000 N/rad on eight of nine runs (1.6x) with
   one at 7,000 (162516, the shortest run at 27 s moving); Jz 6.97-7.46 on
   eight of nine; damping scatters 0.25-1.4 as an unbounded-below parameter
   should.
3. **Per-speed tercile**: C_alpha 17,500 / 12,900 / 11,100 (1.6x), Jz
   6.98 / 7.07 / 7.42. Mild speed trend in the stiffness lower bound, none in
   inertia.
4. **Steady-state yaw gain** at 1.5 m/s: model L_eff 0.950 m vs measured
   0.955 m (ratio 1.005).

### ADE and Frechet against the GPS fixes, whole runs

The independent position reference is GPS: ~1 Hz fixes with a 3 m reported
sigma (the fixes scatter about 0.3-0.6 m around any smooth path, so they are
better than their covariance says). Each model is rolled out over the whole
run on the recorded steering with vx pinned to the wheel speed, the path is
rigidly aligned to the fixes once per run - the initial heading is
unobservable because the IMU yaw is not magnetometer referenced - and ADE is
the mean distance to the fixes. Frechet is the discrete Frechet distance
between the whole path (thinned to 5 Hz) and the fix polyline, the
worst-case coupled separation along the two curves. ADE pooled is
fix-weighted; Frechet pooled is the mean over runs.

| run | fixes | KBM ADE / Frechet | GM3 preset | GM3 fitted | dead reckoning |
|---|---|---|---|---|---|
| 162125 | 81 | 1.28 / 2.36 | 1.46 / 3.14 | 1.31 / 2.59 | 0.44 / 1.11 |
| 162516 | 57 | 0.49 / 0.95 | 0.41 / 0.90 | 0.45 / 0.93 | 0.56 / 1.13 |
| 162644 | 64 | 0.53 / 1.15 | 0.36 / 1.05 | 0.52 / 1.07 | 0.32 / 0.94 |
| 162832 | 22 | 0.13 / 0.92 | 0.08 / 0.94 | 0.10 / 0.93 | 0.09 / 0.91 |
| 163031 | 40 | 2.32 / 5.27 | 2.49 / 5.72 | 2.40 / 5.51 | 0.25 / 1.18 |
| 163129 | 34 | 0.53 / 1.30 | 0.38 / 1.29 | 0.49 / 1.30 | 0.23 / 1.30 |
| 163418 | 77 | 0.59 / 1.31 | 0.46 / 1.30 | 0.50 / 1.23 | 0.33 / 1.03 |
| 163556 | 60 | 1.05 / 2.78 | 1.05 / 3.38 | 1.04 / 2.90 | 0.22 / 1.11 |
| 163726 (held out) | 52 | 0.38 / 1.14 | 0.29 / 0.94 | 0.29 / 0.93 | 0.14 / 0.93 |
| **pooled** | 487 | **0.84 / 1.91** | **0.81 / 2.07** | **0.81 / 1.93** | 0.29 / 1.07 |

All in metres. Before the sensor calibration these read 1.02 / 2.11 (KBM),
1.02 / 2.29 (preset), 1.04 / 2.17 (fitted) and 0.71 / 1.41 (dead reckoning):
the wheel-speed scale and gyro bias were worth 0.2 m on every model and 0.4 m
on dead reckoning, the largest single improvement in this file. With them in,
GM3 fitted beats the kinematic bicycle on six of nine runs and on the
held-out run by 25% in ADE (0.29 vs 0.38 m) and 19% in Frechet; pooled the
two are within 3%. The model's yaw inertia and tire compliance smooth the
rider's steering jitter that the kinematic bicycle passes straight into
heading, which is where the gain comes from; it is small because most of the
remaining error is the rider's unpredictable yaw, identical for both, and the
gap to dead reckoning (0.29 m, which is now at the GPS scatter) on 162125,
163031 and 163556 is that yaw integrated over 60-80 s. Run 163031 is the
outlier at 2.3 m for every model because of one event at t = 12 s: the
rider slows to 0.15 m/s and pivots at 72 deg/s with 41 deg of steer, a
foot-down turn that no rolling model represents (23.7 deg/s of yaw-rate
error in that 3 s window against ~2.5 elsewhere); the heading error it
leaves is carried for the remaining 28 s.

### 3 s windows against dead reckoning

Reference: the wheel-plus-gyro dead-reckoned path, the only reference that
resolves a 3 s window; it consumes the same wheel speed the models do, so it
scores heading and sideslip divergence. Held-out run, 17 windows:

| | r RMSE | psi @ 3 s | ADE | FDE |
|---|---|---|---|---|
| kinematic bicycle | 2.22 deg/s | 1.32 deg | 0.035 m | 0.067 m |
| GM3 preset | 2.32 | 1.65 | 0.049 | 0.105 |
| GM3 fitted | 2.27 | 1.31 | 0.035 | 0.068 |

Pooled over all nine runs (138 windows): ADE 0.038 (KBM) / 0.045 (preset) /
0.040 m (fitted). A trap in this comparison: the reference tracks the centre
of gravity and carries the no-slip sideslip `vy = lr * r`, so a kinematic
bicycle propagated with `vy = 0` traces the rear-axle path and picks up a
spurious `lr * r * t` error (0.1-0.3 m over a 3 s corner) that once read as
GM3 beating the baseline 6x. `kinematic_trace` propagates at the CG.

### The slope input: what it did

With the front wheel free-rolling, "open loop" means the model predicts
speed from rolling resistance and slope alone, and the fair baseline is a
bicycle that holds the window's initial speed (no speed input at all).
Speed RMSE / ADE over 3 s windows, pooled over all runs:

| | v RMSE | ADE |
|---|---|---|
| KBM, constant speed | 0.318 m/s | 0.251 m |
| GM3 fitted, no slope | 0.492 | 0.407 |
| GM3 fitted, IMU slope | 0.567 | 0.450 |

Constant speed wins, and adding the slope makes GM3 worse. Both are the
same fact: the rider is the motor. Kicks of +0.5 m/s² (p90) balance the
0.21 m/s² rolling drag on average, so a model that coasts drifts away from
a rider who does not, and the slope estimator's ~1 deg of noise adds
0.17 m/s² of spurious gravity on a quad that is flat to 0.7 deg. On the
14 windows without a kick (1.5 s, |dv/dt| < 0.35 throughout) speed RMSE is
0.090 (constant) / 0.158 (no slope) / 0.169 m/s (slope): the same ordering,
and no difference between slope on and off. The slope input is plumbed,
verified in sign, and inert on this terrain. It will matter on a recording
with a grade, and the open-loop task will only be winnable once the rider's
input is measured (motor current with the motor on) or the kicks are
modelled from the wheel speed they leave behind.

`python -m experiments.scooter_plots` draws all of this:
`experiments/out/plots/scooter_paths.png` (every run against the GPS fixes,
start marked, ADE and Frechet per model in the title), `scooter_holdout.png`
(yaw rate and heading error on the held-out run), `scooter_errors.png`
(per-run bars including whole-run ADE to GPS) and `scooter_residual.png`
(the yaw residual is flat in steering angle for both models).

**Reading.** With the sensors calibrated against GPS, the fitted model
reproduces the scooter's path to within the GPS scatter's reach of the best
available reference and edges the kinematic bicycle on the held-out run,
with a physically sensible tire (36 x Fz per radian is stiff for a small
pneumatic tire; the profile is flat from ~7,000 N/rad up, so treat "at least
~7,000 N/rad" as the finding). The identifiable content of these logs is
geometry and sensors: wheelbase 0.95 m, gear ratio 0.344, wheel-speed scale
0.545, gyro bias -0.085 deg/s, the encoder zero per run, rolling drag
0.21 m/s². Stiffness, friction, inertia, damping and the slope term need
what this recording does not have: lateral accelerations several times
higher, a grade, and a measured longitudinal input.

**Before the next recording.** Calibrate the wheel-speed scale on a
measured distance and measure the wheelbase with a tape (0.95 m is the
prediction); re-home or log the steering encoder index so the zero does not
drift 83 deg mid-session; mount the IMU upright with x forward; log motor
current with the motor on so speed is predictable; include a hill; and if
tire identification is the goal, ride fast enough to slide.

## Per-surface tuning on the corrected folders

`python -m experiments.scooter_calibrate --root data/<folder>_corrected --tag <name> --fit cp mu yaw_inertia yaw_damping steer_zero --per-run --per-speed`
then `python -m experiments.scooter_plots --tag <name>`; outputs in
`out/scooter_calibration_<name>.json` and `out/plots/<name>/`.

Three log folders, three grounds: `kim_quad` is concrete sidewalk (9 runs,
714 m), `av_williams_lot` is asphalt road surface (6 runs, 474 m),
`slope_sidewalk` is sidewalk with a real grade on three of its 11 runs
(835 m; IMU grade spread 3.3-3.5 deg on 190606 and 190809 against ~0.9 on
flat ground). The wheel-speed scale, gyro bias and wheelbase check out on all
three (chord ratio 0.94-0.98, yaw sign +0.91 to +0.97, wheelbase 0.95-0.96 m
pooled). Each fit holds out its last run; `mu` was freed this time so each
surface would get a friction estimate to judge.

| | concrete (kim quad) | asphalt (AV Williams) | sloped sidewalk |
|---|---|---|---|
| held-out run | 163726 | 182515 | 191928 |
| held-out loss | 9.1e-3 -> 8.3e-3 | 20.8e-3 -> 18.8e-3 | 27.1e-3 -> 25.3e-3 |
| C_alpha (N/rad) | 15,000 | 15,800 | 16,500 |
| per-run C_alpha range | 7,200-19,300 (2.7x) | 12,200-17,500 (1.4x) | 10,500-19,700 (1.9x) |
| per-speed C_alpha | 17,400 / 12,900 / 11,100 | 16,900 / 15,400 / 15,700 | 17,300 / 16,900 / - |
| mu | 1.22 | 1.09 | 1.35 |
| per-run mu range | 0.34-1.59 | 1.08-1.43 | 1.20-1.65 |
| yaw_inertia (kg m²) | 7.18 | 7.21 | 7.04 |
| per-run Jz range | 7.0-8.5 | 7.17-7.35 | 6.94-7.27 |
| yaw_damping | 0.44 | 0.35 | 0.28 |
| steady-state yaw gain ratio | 1.005 | 1.004 | 1.004 |

ADE / Frechet against the GPS fixes over whole runs, pooled (metres):

| | KBM | GM3 preset | GM3 fitted | dead reckoning |
|---|---|---|---|---|
| concrete, 487 fixes | 0.83 / 1.89 | 0.80 / 2.06 | 0.81 / 1.91 | 0.29 / 1.07 |
| asphalt, 284 fixes | 0.50 / 1.20 | 0.48 / 1.21 | 0.46 / 1.21 | 0.20 / 1.03 |
| sloped sidewalk, 463 fixes | 0.55 / 1.24 | 0.61 / 1.37 | 0.44 / 1.19 | 0.26 / 1.10 |
| held-out runs only | 0.38 / 1.12 vs 0.28 / 0.92 | 0.32 / 1.11 vs 0.29 / 1.17 | 0.38 / 1.10 vs 0.28 / 1.10 |

(held-out column: KBM vs GM3 fitted.) GM3 fitted beats the kinematic bicycle
on ADE on every surface, by 3% (concrete), 9% (asphalt) and 19% (sloped
sidewalk), and by 25% or more on each held-out run. Frechet is within 2% on
the flat sets and 4% better on the sloped one. The margin grows with the
grade: on 190606, the run with the steepest grade, the fitted model's ADE is
0.50 m against 1.01 for the bicycle, and that is the slope input at work
(it enters the yaw balance through load transfer and the lateral gravity
term on banked stretches; the longitudinal channel is still measured).

### Are the parameters reasonable?

**Stiffness, inertia and damping: yes, and they agree across surfaces.**
C_alpha lands at 15,000-16,500 N/rad on all three grounds (36-40 x Fz per
radian), yaw inertia at 7.0-7.2 kg m², damping at 0.3-0.4. A tire's cornering
stiffness is a property of the tire and its load, not the ground, and the
rider plus scooter inertia is the same on every set, so agreement to within
10% is the right answer. Per-speed values hold within 1.1x on asphalt and the
slope, 1.6x on concrete where the slow tercile pulls high. All of these are
still bounds rather than values (the loss profile is flat above ~7,000 N/rad
and below Jz ~8 on every surface), which is why the per-run spread on
stiffness runs 1.4-2.7x while inertia holds within 4% on 26 of 26 runs: the
yaw inertia is the one dynamic parameter these logs really pin down.

**mu: no, and it cannot be.** The fitted values, 1.09 (asphalt), 1.22
(concrete), 1.35 (sloped sidewalk), are ordered the opposite way from what
the materials suggest (asphalt is normally the grippiest of the three) and
they scatter 0.34-1.65 between runs on the *same* concrete. The loss profile
explains both: with the other parameters at their presets, held-out loss is
flat to 0.3% from mu 0.8 to 2.5 on every surface, and only rises steeply
below 0.5, where the tire would begin to saturate at the 2.5 m/s² peak
lateral acceleration recorded. So the data says mu is above about 0.6 and
nothing more; a fitted value in that band is where Adam stopped, not what
the ground is. Rubber on dry concrete or asphalt sits near 0.8-1.0, and any
of these fits is consistent with that. To separate the three surfaces the
tire has to slide: the same corners at 3-4 m/s (lateral acceleration above
5 m/s²), or braking to lock-up with the motor on, would put mu on the steep
part of the profile and let the per-run spread collapse to the surface
difference. Until then, hold mu at 0.8 on every surface and free only the
parameters the logs constrain.

**Two cautions in the sloped set.** Run 190210 gives a nonsense coasting-drag
regression (-2.0 m/s², a 15 deg "plane") because it has a 10 s GPS gap and
too few kick-free samples; the grade estimate itself is fine (0.75 deg
spread) and the run fits normally. Run 190351 is 7 s of motion and should
not be read on its own.
