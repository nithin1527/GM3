# Scooter experiments

Everything here works on the Hiboy S2 scooter logs under `data/` (CSV at
30 Hz: ODrive drive and steer state, ~1 Hz GPS, IMU) through the loader in
`gm3.dataset.scooter` and the `make_scooter_config` preset. Scripts run from the
repo root as modules; `_bootstrap.py` registers the checkout as the `gm3`
package if it is not already importable under that name.

- `scooter_correct.py` - writes corrected copies of every log folder
  (`data/<folder>_corrected/`): scale and sign of the wheel speed, gyro bias,
  IMU axes into the vehicle frame, steering zero and column angle, body speed,
  dead-reckoned pose, surface angles, GPS in local metres. Same rows and
  timestamps as the originals; each output folder carries a `README.md` with
  the column definitions and a `corrections.json` with the constants used.
- `scooter_report.py` - inventories the logs (durations, distances, speeds,
  turn sequences) and re-derives every loader assumption with `--checks`:
  wheel-speed scale, yaw sign, steering gear ratio and zero, understeer,
  steer-to-yaw lag, IMU axes, dead reckoning against GPS, grade.
- `scooter_sensors.py` - GPS-anchored calibration of the sensor constants
  (wheel-speed scale, wheelbase, per-run gyro bias and steering zero) with a
  differentiable Procrustes alignment inside a Huber loss; no tire model.
- `scooter_calibrate.py` - fits the scooter preset's tire stiffness, inertia,
  damping and steering zero on the yaw channel with four validation gates
  (held-out run, per-run, per-speed, steady-state yaw gain), against a
  kinematic-bicycle bar; reports ADE and discrete Frechet distance against the
  GPS fixes over whole runs, and an open-loop speed comparison (slope input
  on / off, coasting windows only) against a constant-speed bicycle.
  `--slope {imu,plane,none}` picks the surface-angle source, `--motor-on`
  treats the wheel speed as a drive input.
- `scooter_tire_models.py` - one calibration per tire law (`brush`, `fiala`,
  `dugoff`, `burckhardt`, `pacejka`) on all three corrected folders pooled,
  with `scooter_calibrate`'s protocol: `fit` (last run of each surface held
  out), `fit --all-data` (the reported parameters), `fit --all-data --free all`
  (friction and shape freed too), `profile` (loss along the stiffness axis) and
  `report` (parameter and accuracy tables, `out/tire_models/report.md`).
  `scooter_calibrate --tire-model` runs the single-folder workflow on any law. `--surfaces kim_quad slope_sidewalk`
  pools a subset of the folders and writes to `out/tire_models_<surfaces>/`.
- `scooter_tire_model_plots.py` - parameter tables (`parameters.md`, `parameters.csv`) and plots for whatever
  `scooter_tire_models` has fitted: fitted force curves against the recorded load range, held-out accuracy per
  law, per-run GPS error, stiffness profile, parameter comparison, held-out traces and paths. Takes the same
  `--surfaces`; `--only` redraws a subset.
- `scooter_plots.py` - renders the fitted GM3 against the kinematic bicycle on
  every run (paths vs GPS with the start marked, held-out yaw rate and heading
  error, per-run error bars, residual by steering angle) into `out/plots/`.

```bash
python -m experiments.scooter_correct                   # data/<folder>_corrected/
python -m experiments.scooter_report --checks
python -m experiments.scooter_sensors
python -m experiments.scooter_calibrate --per-run --per-speed
python -m experiments.scooter_plots                     # needs matplotlib
```

Outputs land in `out/`: `scooter_runs.csv`, `scooter_sensors.json`,
`scooter_calibration.{csv,json}` and `plots/`. See
[CALIBRATION.md](../CALIBRATION.md) for the workflow and what it found.
