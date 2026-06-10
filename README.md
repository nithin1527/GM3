# gm3 API Guide

`gm3` contains the newer GM3 implementation with a shared vehicle configuration API and two simulation backends:

- `gm3.gm3.GM3`: normal NumPy implementation for deterministic simulation.
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

Control has 2 values:

```text
[omega, delta]
```

- `omega`: wheel angular velocity applied to driven tires.
- `delta`: steering angle applied to steerable tires.

Non-driven tires free-roll internally. If multiple tires are marked `driven=True`, they receive the same `omega`.

## Quick Start With Presets

```python
from gm3.gm3 import GM3
from gm3.diffgm3 import DiffGM3
from gm3.shared import make_bicycle_config, make_cart_config

bike_cfg = make_bicycle_config()
cart_cfg = make_cart_config()

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
