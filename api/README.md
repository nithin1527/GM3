# DiffGM3 API

HTTP/WebSocket layer over `gm3.diffgm3.DiffGM3`. Lives entirely in `api/` — the
model code in `diffgm3/`, `gm3/`, and `shared/` is untouched.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r api/requirements.txt
```

## Run

```bash
.venv/bin/python api/server.py
# or
DIFFGM3_API_PORT=9000 .venv/bin/python api/server.py
```

Defaults to `http://127.0.0.1:8731`. Interactive docs at `http://127.0.0.1:8731/docs`.

`server.py` aliases this checkout as the `gm3` package at startup, so it works
no matter what the directory is named or where it is run from. With the parent
directory on `sys.path` (and named `gm3`) you can also use
`uvicorn gm3.api.server:app --reload`.

## Conventions

- State: `[x, y, psi, vx, vy, r, gamma, gamma_dot]`
- Control: `[omega, delta]`
- Slope (optional): `[alpha_p, alpha_r]` in radians — surface angles in the
  vehicle's body frame. `alpha_p > 0` = climbing along body +x, `alpha_r > 0`
  = the body +y side is uphill. Omit for flat ground.
- Every request selects a vehicle via `"model"`: either
  `{"preset": "bicycle" | "cart", "dt": 0.02}` or
  `{"config": {...VehicleConfig fields...}, "dt": 0.02}`.
  Models are cached server-side per config, so repeated requests are cheap.

## REST

```bash
curl http://localhost:8731/health
curl http://localhost:8731/presets

curl -X POST http://localhost:8731/step -H 'Content-Type: application/json' -d '{
  "model": {"preset": "bicycle", "dt": 0.02},
  "state": [0, 0, 0, 2.0, 0, 0, 0, 0],
  "control": [5.7, 0.05],
  "slope": [0.0997, 0.0],
  "return_aux": true
}'

curl -X POST http://localhost:8731/rollout -H 'Content-Type: application/json' -d '{
  "model": {"preset": "bicycle", "dt": 0.02},
  "initial_state": [0, 0, 0, 2.0, 0, 0, 0, 0],
  "controls": [[5.7, 0.05], [5.7, 0.05], [5.7, 0.05]]
}'
```

`return_aux: true` adds normal loads, steering angles, tire/body forces, slip,
tire velocities, force/moment totals, and the constrained physical parameters.

## WebSocket session (interactive frontends)

`ws://localhost:8731/ws/session` keeps per-connection state so a render loop
only sends a control each frame:

```jsonc
// client -> server
{"type": "init",  "model": {"preset": "bicycle", "dt": 0.02}, "state": [0,0,0,2,0,0,0,0]}
{"type": "step",  "control": [5.7, 0.05], "slope": [0.0997, 0.0], "return_aux": false}
{"type": "reset", "state": [0,0,0,0,0,0,0,0]}   // state optional -> zeros

// server -> client
{"type": "ready", "state": [...], "t": 0.0}
{"type": "state", "state": [...], "t": 0.02}
{"type": "error", "message": "..."}
```

Measured round-trip: ~0.4 ms/step on localhost.

## Frontend

The GMMSim three.js app (`../GMMSim`) connects through
`src/api/diffgm3/` — open it with `?backend=diffgm3` in the URL. See
`GMMSim/src/api/diffgm3/README.md`.
