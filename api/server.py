"""FastAPI server exposing DiffGM3 over REST and WebSocket.

Run directly:
    python GM3/api/server.py
or via uvicorn (with the GM3 parent directory on sys.path):
    uvicorn gm3.api.server:app --reload --port 8731

REST:
    GET  /health
    GET  /presets
    POST /step       single DiffGM3 step (stateless)
    POST /rollout    multi-step rollout (stateless)

WebSocket /ws/session (stateful, for interactive frontends):
    -> {"type": "init", "model": {"preset": "bicycle", "dt": 0.02}, "state": [..8]}
    <- {"type": "ready", "state": [..8], "t": 0.0}
    -> {"type": "step", "control": [omega, delta], "slope": [alpha_p, alpha_r], "return_aux": false}
    <- {"type": "state", "state": [..8], "t": 0.02, "aux": null}
    -> {"type": "reset", "state": [..8]}        # state optional, defaults to zeros
    <- {"type": "state", "state": [..8], "t": 0.0}
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _ensure_gm3_importable() -> None:
    """Alias this checkout as the `gm3` package when run directly.

    The checkout directory may be named `GM3`, and Python import matching is
    case-sensitive, so a plain sys.path insert is not sufficient.
    """
    try:
        import gm3  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    package_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gm3", package_dir / "__init__.py", submodule_search_locations=[str(package_dir)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gm3"] = module
    spec.loader.exec_module(module)


_ensure_gm3_importable()

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from gm3.shared.constants import CONTROL_FIELDS, STATE_FIELDS

from gm3.api.registry import PRESET_BUILDERS, ModelRegistry, to_jsonable, vehicle_config_to_model
from gm3.api.schemas import (
    N_CONTROL,
    N_SLOPE,
    N_STATE,
    HealthResponse,
    ModelSpec,
    PresetsResponse,
    RolloutRequest,
    RolloutResponse,
    StepRequest,
    StepResponse,
)

app = FastAPI(
    title="diffgm3-api",
    description="DiffGM3 vehicle dynamics — REST + WebSocket API over the trainable GM3 model",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only — tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        torch_version=torch.__version__,
        state_fields=list(STATE_FIELDS),
        control_fields=list(CONTROL_FIELDS),
    )


@app.get("/presets", response_model=PresetsResponse)
def presets() -> PresetsResponse:
    return PresetsResponse(
        presets={name: vehicle_config_to_model(builder()) for name, builder in PRESET_BUILDERS.items()}
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    try:
        model = registry.get(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    state = torch.tensor(req.state, dtype=torch.get_default_dtype())
    control = torch.tensor(req.control, dtype=torch.get_default_dtype())
    slope = None if req.slope is None else torch.tensor(req.slope, dtype=torch.get_default_dtype())

    with torch.inference_mode():
        if req.return_aux:
            next_state, aux = model(state, control, req.dt, slope=slope, return_aux=True)
            return StepResponse(state=next_state.tolist(), aux=to_jsonable(aux))
        next_state = model(state, control, req.dt, slope=slope)
        return StepResponse(state=next_state.tolist())


@app.post("/rollout", response_model=RolloutResponse)
def rollout(req: RolloutRequest) -> RolloutResponse:
    try:
        model = registry.get(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    dtype = torch.get_default_dtype()
    initial_state = torch.tensor(req.initial_state, dtype=dtype).unsqueeze(0)  # [1, 8]
    controls = torch.tensor(req.controls, dtype=dtype).unsqueeze(1)  # [T, 1, 2]
    slopes = None if req.slopes is None else torch.tensor(req.slopes, dtype=dtype).unsqueeze(1)  # [T|1, 1, 2]

    with torch.inference_mode():
        states = model.rollout(initial_state, controls, req.dt, slopes=slopes)  # [T + 1, 1, 8]

    return RolloutResponse(states=states.squeeze(1).tolist())


class _Session:
    """Per-connection simulation state for the WebSocket endpoint."""

    def __init__(self) -> None:
        self.model = None
        self.dt: float = 0.02
        self.state: torch.Tensor | None = None
        self.t: float = 0.0
        self.control: torch.Tensor | None = None
        self.max_drive_surface_accel: float = 0.8
        self.max_steer_rate: float = 1.6
        self.coast_rolling_accel: float = 0.28
        self.coast_drag_gain: float = 0.035

    def configure(self, spec: ModelSpec, state: list[float] | None) -> None:
        self.model = registry.get(spec)
        self.dt = spec.dt
        self.reset(state)

    def reset(self, state: list[float] | None) -> None:
        if state is not None:
            if len(state) != N_STATE:
                raise ValueError(f"state must have {N_STATE} values")
            self.state = torch.tensor(state, dtype=torch.get_default_dtype())
        else:
            self.state = torch.zeros(N_STATE, dtype=torch.get_default_dtype())
        self.t = 0.0
        self.control = None

    def step(
        self,
        control: list[float],
        n: int,
        return_aux: bool,
        slope: list[float] | None = None,
        coast: bool = False,
    ):
        if self.model is None or self.state is None:
            raise ValueError("session not initialized — send an 'init' message first")
        if len(control) != N_CONTROL:
            raise ValueError(f"control must have {N_CONTROL} values [omega, delta]")
        if slope is not None and len(slope) != N_SLOPE:
            raise ValueError(f"slope must have {N_SLOPE} values [alpha_p, alpha_r]")
        control_t = self._filtered_control(torch.tensor(control, dtype=torch.get_default_dtype()), coast=coast)
        slope_t = None if slope is None else torch.tensor(slope, dtype=torch.get_default_dtype())
        aux = None
        with torch.inference_mode():
            for _ in range(max(n, 1)):
                if return_aux:
                    self.state, aux = self.model(self.state, control_t, self.dt, slope=slope_t, return_aux=True)
                else:
                    self.state = self.model(self.state, control_t, self.dt, slope=slope_t)
                if coast:
                    self._apply_coast_resistance()
                self.t += self.dt
        return aux

    def _filtered_control(self, target: torch.Tensor, *, coast: bool) -> torch.Tensor:
        """Rate-limit interactive WebSocket commands to avoid actuator jumps."""
        if self.control is None:
            self.control = torch.zeros_like(target)

        drive_radius = self._drive_radius()
        omega_rate = self.max_drive_surface_accel / max(drive_radius, 1e-6)
        rates = target.new_tensor([omega_rate, self.max_steer_rate])
        max_step = rates * self.dt
        delta = torch.clamp(target - self.control, min=-max_step, max=max_step)
        self.control = self.control + delta
        if coast and self.state is not None:
            self.control = self.control.clone()
            self.control[0] = self.state[3] / max(drive_radius, 1e-6)
        return self.control

    def _apply_coast_resistance(self) -> None:
        """Apply mild rolling/aero resistance to interactive coasting."""
        if self.state is None:
            return
        vx = self.state[3]
        speed = torch.abs(vx)
        if float(speed) < 1e-5:
            return
        accel = self.coast_rolling_accel + self.coast_drag_gain * speed * speed
        dv = torch.minimum(speed, accel * self.state.new_tensor(self.dt))
        self.state = self.state.clone()
        self.state[3] = vx - torch.sign(vx) * dv

    def _drive_radius(self) -> float:
        if self.model is None:
            return 0.3
        radii = self.model.tire_radius
        driven = self.model.driven_mask
        if bool(driven.any()):
            radii = radii[driven]
        return float(radii.mean().detach().cpu())


@app.websocket("/ws/session")
async def ws_session(ws: WebSocket) -> None:
    await ws.accept()
    session = _Session()

    async def send_error(message: str) -> None:
        await ws.send_json({"type": "error", "message": message})

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            try:
                if msg_type == "init":
                    spec = ModelSpec.model_validate(msg.get("model") or {})
                    session.configure(spec, msg.get("state"))
                    await ws.send_json({"type": "ready", "state": session.state.tolist(), "t": session.t})

                elif msg_type == "step":
                    aux = session.step(
                        msg.get("control") or [],
                        int(msg.get("n", 1)),
                        bool(msg.get("return_aux", False)),
                        msg.get("slope"),
                        bool(msg.get("coast", False)),
                    )
                    payload = {"type": "state", "state": session.state.tolist(), "t": session.t}
                    if aux is not None:
                        payload["aux"] = to_jsonable(aux)
                    await ws.send_json(payload)

                elif msg_type == "reset":
                    session.reset(msg.get("state"))
                    await ws.send_json({"type": "state", "state": session.state.tolist(), "t": session.t})

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

                else:
                    await send_error(f"unknown message type: {msg_type!r}")

            except (ValueError, ValidationError) as exc:
                await send_error(str(exc))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("DIFFGM3_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("DIFFGM3_API_PORT", "8731")),
    )
