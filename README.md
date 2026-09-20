# gm3 API Guide

`gm3` contains the newer GM3 implementation with a shared vehicle configuration API and two simulation backends:

- `gm3.gm3.GM3`: normal implementation for deterministic simulation.
- `gm3.diffgm3.DiffGM3`: PyTorch `nn.Module` implementation for differentiable rollout and parameter training.

Both backends use the same `VehicleConfig`, `TireConfig`, state layout, and control layout.

## State And Control Layout

State has 8 values:

```text
[x, y, psi, vx, vy, r, gamma, gamma_dot]
```

- `x`, `y`: global position.
- `psi`: heading angle.
- `vx`, `vy`: body-frame longitudinal and lateral velocity.
- `r`: yaw rate.
- `gamma`: lean/roll angle.
- `gamma_dot`: lean/roll angular rate.

Control width depends on the vehicle's `drive_mode`. With the default `"single"` it has 2 values:

```text
[omega, delta]
```

- `omega`: wheel angular velocity applied to driven tires.
- `delta`: steering angle applied to steerable tires.

Non-driven tires free-roll internally. If multiple tires are marked `driven=True`, they receive the same `omega`.

With `drive_mode="independent"` each driven tire gets its own speed, in tire order, so the control widens to `1 + driven_count`:

```text
[omega_0, ..., omega_k, delta]
```

This is what a skid-steer vehicle needs: it has nothing to steer, and generates yaw purely by driving its left and right sides at different speeds. Read the width off the config with `config.n_control`. No shipped preset uses it; build one with `drive_mode="independent"`.

```python
from gm3.shared import GM3Control

shared = GM3Control(omega=12.0, delta=0.05)              # one speed for every driven tire
skid = GM3Control(omega=(4.0, 8.0, 4.0, 8.0), delta=0.0) # one speed per driven tire
```

Tire contact velocities include the across-track term `-r * y`, so on a vehicle with laterally offset wheels the outer track runs faster than the inner one through a turn. Single-track presets are unaffected because their tires sit at `y = 0`; multi-track ones differ slightly from results recorded before this term existed.

## Quick Start With Presets

```python
from gm3.gm3 import GM3
from gm3.diffgm3 import DiffGM3
from gm3.shared import make_bicycle_config, make_cart_config, make_scooter_config

bike_cfg = make_bicycle_config()
cart_cfg = make_cart_config()
scooter_cfg = make_scooter_config()   # Hiboy S2 kick scooter, single track, casters

normal_model = GM3(bike_cfg)
diff_model = DiffGM3(bike_cfg, dt=0.05)
```

Use `GM3` when you do not need gradients. Use `DiffGM3` when you want PyTorch autograd, optimization, or differentiable training.

## Custom Vehicle Geometry

Create custom vehicles with `VehicleConfig` and `TireConfig`.

```python
from gm3.shared import TireConfig, VehicleConfig

lf = 0.6
lr = 0.5

cfg = VehicleConfig(
    mass=105.0,
    yaw_inertia=16.0,
    roll_inertia=24.0,
    lf=lf,
    lr=lr,
    width=0.38,
    cg_height=0.92,
    tires=(
        TireConfig(
            x=lf,
            y=0.0,
            radius=0.32,
            mu=1.05,
            cp=14_500.0,
            contact_length=0.035,
            steerable=True,
            driven=False,
            can_lean=True,
        ),
        TireConfig(
            x=-lr,
            y=0.0,
            radius=0.32,
            mu=0.98,
            cp=13_500.0,
            contact_length=0.032,
            steerable=False,
            driven=True,
            can_lean=True,
        ),
    ),
    can_lean=True,
    align_gain=0.22,
    yaw_damping=0.65,
    roll_damping=12.0,
    steering_mode="direct",
)
```

Customizable vehicle-level fields:

```text
mass
yaw_inertia
roll_inertia
lf, lr
width
cg_height
can_lean
align_gain
yaw_damping
roll_damping
steering_mode: "direct" or "ackermann"
drive_mode: "single" or "independent"
gravity
min_normal_load
eps
```

Customizable tire-level fields:

```text
x, y
radius
mu
cp
contact_length
steerable
driven
can_lean
```

This supports two-wheelers, carts, trikes, front/rear/all-wheel drive layouts, direct steering, Ackermann steering, leaning vehicles, and non-leaning vehicles.

## Normal GM3 Usage

```python
from gm3.gm3 import GM3
from gm3.shared import GM3Control, GM3State, make_bicycle_config

cfg = make_bicycle_config()
model = GM3(cfg)

state = GM3State(
    x=0.0,
    y=0.0,
    psi=0.0,
    vx=2.0,
    vy=0.0,
    r=0.0,
    gamma=0.0,
    gamma_dot=0.0,
)

control = GM3Control(
    omega=2.0 / cfg.tires[1].radius,
    delta=0.05,
)

next_state = model.step(state, control, dt=0.05)
```

Roll out multiple controls:

```python
controls = [control] * 100
states = model.rollout(state, controls, dt=0.05)
```

Request auxiliary debug data:

```python
next_state, aux = model.step(state, control, dt=0.05, return_aux=True)

print(aux["normal_loads"])
print(aux["steering_angles"])
print(aux["tire_forces"])
print(aux["slip"])
```

## DiffGM3 Usage

```python
import torch

from gm3.diffgm3 import DiffGM3
from gm3.shared import make_bicycle_config

cfg = make_bicycle_config()
model = DiffGM3(cfg, dt=0.05)

initial_state = torch.tensor([
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
])

controls = torch.tensor([
    [[2.0 / cfg.tires[1].radius, 0.05]]
] * 50)

states = model.rollout(initial_state, controls)
```

DiffGM3 tensor shapes:

```text
initial_state: [B, 8]
controls:      [T, B, 2]
states:        [T + 1, B, 8]
```

Single-step usage:

```python
next_state, aux = model(initial_state, controls[0], return_aux=True)
```

`return_aux=True` returns normal loads, steering angles, tire forces, body forces, slip, tire velocities, total forces, total moment, and physical parameters.

### Sloped Surfaces

`DiffGM3` accepts an optional `slope` input with body-frame surface angles `[alpha_p, alpha_r]` in radians. `alpha_p > 0` means the vehicle is climbing along body `+x`; `alpha_r > 0` means the body `+y` side is uphill. Gravity is decomposed in the surface frame: the longitudinal/lateral components enter the equations of motion and load transfer, and the normal component `g cos(alpha_p) cos(alpha_r)` scales normal loads and the lean restoring moment.

```python
import math

slope = torch.tensor([[math.atan(0.10), 0.0]])   # [B, 2] -- 10% uphill grade

next_state = model(initial_state, controls[0], slope=slope)

# constant slope for a rollout (pass [T, B, 2] for per-step slopes)
states = model.rollout(initial_state, controls, slopes=slope)
```

Omitting `slope` (or passing zeros) recovers the flat-ground dynamics exactly.

### Obstacles And The Enveloping Tire

`DiffGM3` accepts an optional `obstacle` argument naming a localized ground feature
from `gm3.shared.terrain`: `"speedbump"` (4 cm ridge at x = 8 m), `"pothole"`
(0.6 m wide, 10 cm deep, centered at (8, 0)), or `"rough"` (2.5 cm ripples over
x in [6, 14]). Each wheel is discretized into radial spring elements coupled by
interradial springs, calibrated at build time so flat ground reproduces that
wheel's static load. Over an obstacle the model applies each wheel's load
deviation from flat and its longitudinal drag.

```python
next_state, aux = model(initial_state, controls[0], obstacle="speedbump", return_aux=True)

print(aux["enveloping_delta_fz"])      # [B, n_tires] load deviation from flat, N
print(aux["enveloping_drag"])          # [B] summed longitudinal drag, N
print(aux["enveloping_drag_moment"])   # [B] yaw moment from asymmetric drag, N m

states = model.rollout(initial_state, controls, obstacle="pothole")
```

Drag acts at the wheel that meets the obstacle, so a one-sided hit (an offset
pothole, an oblique bump crossing) yaws the vehicle toward that wheel. On a
single-track vehicle, where both tires sit at `y = 0`, that moment is identically
zero.

Obstacle geometry is fixed in world coordinates, so the vehicle has to be driven
over it. Omitting `obstacle` recovers the flat-ground dynamics exactly.

## Tire Models

`DiffGM3` takes a `tire_model`. The default, `"brush"`, is the original GM3 brush tire and is
bit-for-bit what it was; the other four are the comparison laws from `gm3-models`, ported into
`gm3.diffgm3.tire_models` with their formulations and smoothing unchanged.

| name | force law | tire parameters |
|---|---|---|
| `brush` | GM3 brush with spin and aligning moment | `TireConfig` `mu`, `cp`, `contact_length`; no tire kwargs |
| `fiala` | critical-slip `Fx`, cubic `Fy` with residual lateral capacity | `cx`, `cy`, `mu` |
| `dugoff` | combined-slip Dugoff, divisor 2 | `cx`, `cy`, `mu` |
| `burckhardt` | `c1 (1 - exp(-c2 s)) - c3 s` at resultant slip, resolved along the slip direction | `road` or `coefficients_x` / `coefficients_y`; `match_framework_mu` |
| `pacejka` | B-C-D-E Magic Formula with a radial friction projection | `cx`, `cy`, `mu`, `shape_x/y`, `curvature_x/y` |

```python
from gm3.diffgm3 import DiffGM3, TIRE_MODELS

model = DiffGM3(cfg, dt=0.005, tire_model="fiala", cy=2000.0, trainable_stiffness=True)
wet = DiffGM3(cfg, tire_model="burckhardt", road="wet_asphalt")
mf = DiffGM3(cfg, tire_model="pacejka", cy=[1800.0, 2100.0], shape_y=1.4, trainable_shape=True)

print(model.cornering_stiffness())                 # -dFy/dalpha per tire at zero slip, any law
print(model.physical_parameters(detach=True))      # framework values plus "tire.*"
```

Aliases `magic_formula` and `gm3_brush` work, and a `TireModel` module can be passed instead of
a name. `cx` / `cy` default to `2 * cp * contact_length**2` per tire. Nothing model-specific is
trainable unless asked for: `trainable_stiffness` (fiala, dugoff, pacejka),
`trainable_shape` (pacejka C and E) and `trainable_coefficients` (Burckhardt, which has no
stiffness: its zero-slip slope is `(c1*c2 - c3) * Fz`). Framework parameters a law does not
read (`cp`, `contact_length`, `align_gain`, and `mu` for absolute-friction Burckhardt) are
frozen. The four comparison laws return `Mz = 0`.

Two differences from `gm3-models`: stiffness is log-parametrized between `STIFFNESS_BOUNDS`
rather than softplus (the same reason `cp` is, so old `raw_cx` / `raw_cy` checkpoints do not
load), and the comparison laws see the same `kappa` and `alpha` as the brush. Pass
`speed_epsilon=0.5` for that repo's `sqrt(vx**2 + speed_epsilon**2)` slip instead; with it the
two implementations agree to 1e-14 on the four comparison laws. The non-differentiable `GM3` backend still has
the brush tire only.

The published road coefficients make a stiff tire (about `30 * Fz` per radian on dry asphalt),
and so do realistic `cy` values: explicit Euler at the default `dt` then oscillates, see
CALIBRATION.md. Use a small `dt` or substep `derivative`.

## Training Physical Parameters

`DiffGM3` is an `nn.Module`, so train with normal PyTorch optimizers.

```python
import torch.nn.functional as F

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# target_states shape: [T + 1, B, 8]
target_states = ...

for step in range(200):
    pred_states = model.rollout(initial_state, controls)

    # Usually train against measured pose/velocity/yaw states.
    loss = F.mse_loss(pred_states[..., :6], target_states[..., :6])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

Trainable physical parameters:

```text
raw_mu
raw_cp
raw_contact_length
raw_yaw_inertia
raw_roll_inertia
raw_align_gain
raw_yaw_damping
raw_roll_damping
```

Use `physical_parameters()` to view constrained physical values:

```python
params = model.physical_parameters(detach=True)

for name, value in params.items():
    print(name, value)
```

Geometry is fixed during training:

```text
mass
lf, lr
width
cg_height
tire x/y positions
tire radius
steerable/driven masks
```

If `can_lean=False`, roll inertia and roll damping are inactive in the dynamics and will not receive useful gradients.

## Calibrating From The Scooter Logs

`gm3.dataset.scooter` reads the Hiboy S2 CSV logs (ODrive drive/steer, GPS,
IMU) into rollout tensors, and `make_scooter_config` is the matching
single-track preset. The loader applies a GPS-calibrated wheel-speed scale, the
gyro's rest bias, an upside-down IMU sign, the front-wheel speed projection,
the 3.75 / 10.9 steering gear ratio, a per-run steering-encoder zero and an
accelerometer-derived surface angle, each verified by
`experiments.scooter_report --checks`. [CALIBRATION.md](CALIBRATION.md) is the
end-to-end workflow and what it found.

```python
from gm3.dataset.scooter import load_all, windows_of

runs = load_all()                          # every moving run, recording order
windows = windows_of(runs, horizon=45)     # 1.5 s windows at 30 Hz
```

## Batching Guidance

For training, batch trajectories. Do not loop one trajectory at a time.

Good:

```python
# T timesteps, B trajectories, 2 controls
controls.shape == (T, B, 2)
states = model.rollout(initial_states, controls)
```

Avoid:

```python
for trajectory in trajectories:
    states = model.rollout(one_initial_state, one_control_sequence)
```
