from __future__ import annotations

import time
import unittest

import numpy as np

from new_gm3.gm3 import GM3
from new_gm3.shared import GM3Control, GM3State, TireConfig, VehicleConfig, make_bicycle_config, make_cart_config

try:
    import torch

    from new_gm3.diffgm3 import DiffGM3
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    torch = None
    DiffGM3 = None


def _seconds_per_call(fn, *, repeats: int = 1) -> tuple[object, float]:
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed / repeats


def make_custom_trike_config() -> VehicleConfig:
    lf = 0.7
    lr = 0.9
    half_width = 0.42
    return VehicleConfig(
        mass=240.0,
        yaw_inertia=95.0,
        roll_inertia=45.0,
        lf=lf,
        lr=lr,
        width=2.0 * half_width,
        cg_height=0.55,
        tires=(
            TireConfig(
                x=lf,
                y=0.0,
                radius=0.26,
                mu=0.82,
                cp=9_000.0,
                contact_length=0.055,
                steerable=True,
                driven=False,
            ),
            TireConfig(
                x=-lr,
                y=half_width,
                radius=0.29,
                mu=0.88,
                cp=11_000.0,
                contact_length=0.075,
                steerable=False,
                driven=True,
            ),
            TireConfig(
                x=-lr,
                y=-half_width,
                radius=0.29,
                mu=0.88,
                cp=11_000.0,
                contact_length=0.075,
                steerable=False,
                driven=True,
            ),
        ),
        can_lean=False,
        align_gain=0.16,
        yaw_damping=0.8,
        roll_damping=8.0,
        steering_mode="ackermann",
    )


def make_custom_leaning_direct_config() -> VehicleConfig:
    lf = 0.6
    lr = 0.5
    return VehicleConfig(
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


class GM3ApiSmokePerfTests(unittest.TestCase):
    def test_normal_gm3_bicycle_rollout_api_and_timing(self) -> None:
        cfg = make_bicycle_config()
        model = GM3(cfg)
        initial = GM3State(x=0.0, y=0.0, psi=0.0, vx=2.0, vy=0.0, r=0.0)
        control = GM3Control(omega=2.0 / cfg.tires[1].radius, delta=0.04)
        controls = [control] * 200

        states, elapsed = _seconds_per_call(lambda: model.rollout(initial, controls, dt=0.02), repeats=3)
        stacked = np.stack([state.as_array() for state in states], axis=0)

        self.assertEqual(stacked.shape, (201, 8))
        self.assertTrue(np.isfinite(stacked).all())
        self.assertGreater(stacked[-1, 0], stacked[0, 0])
        print(f"\n[normal GM3 bicycle] 200-step rollout: {elapsed * 1000.0:.2f} ms")

    def test_normal_gm3_cart_aux_api_and_timing(self) -> None:
        cfg = make_cart_config()
        model = GM3(cfg)
        state = GM3State(x=0.0, y=0.0, psi=0.0, vx=3.0, vy=0.0, r=0.0)
        control = GM3Control(omega=3.0 / cfg.tires[2].radius, delta=0.03)

        result, elapsed = _seconds_per_call(lambda: model.step(state, control, dt=0.02, return_aux=True), repeats=200)
        next_state, aux = result

        self.assertEqual(next_state.as_array().shape, (8,))
        self.assertTrue(np.isfinite(next_state.as_array()).all())
        self.assertEqual(aux["normal_loads"].shape, (len(cfg.tires),))
        self.assertEqual(aux["tire_forces"].shape, (len(cfg.tires), 3))
        self.assertTrue(np.isfinite(aux["tire_forces"]).all())
        print(f"[normal GM3 cart] step + aux: {elapsed * 1000.0:.3f} ms/step")

    def test_normal_gm3_custom_configs_rollout(self) -> None:
        cases = [
            ("custom trike", make_custom_trike_config(), 2, 0.05),
            ("custom leaning direct", make_custom_leaning_direct_config(), 1, 0.04),
        ]

        for label, cfg, driven_tire_index, delta in cases:
            with self.subTest(label=label):
                model = GM3(cfg)
                initial = GM3State(x=0.0, y=0.0, psi=0.05, vx=2.4, vy=0.05, r=0.02, gamma=0.02, gamma_dot=0.01)
                control = GM3Control(omega=2.4 / cfg.tires[driven_tire_index].radius, delta=delta)

                states, elapsed = _seconds_per_call(lambda: model.rollout(initial, [control] * 100, dt=0.02), repeats=3)
                stacked = np.stack([state.as_array() for state in states], axis=0)

                self.assertEqual(stacked.shape, (101, 8))
                self.assertTrue(np.isfinite(stacked).all())
                self.assertGreater(stacked[-1, 0], stacked[0, 0])
                print(f"[normal GM3 {label}] 100-step rollout: {elapsed * 1000.0:.2f} ms")


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class DiffGM3ApiSmokePerfTests(unittest.TestCase):
    def test_diffgm3_batch_rollout_backward_api_and_timing(self) -> None:
        cfg = make_bicycle_config()
        model = DiffGM3(cfg, dt=0.02)

        batch = 8
        horizon = 64
        initial = torch.zeros(batch, 8)
        initial[:, 3] = torch.linspace(1.5, 3.0, batch)
        initial[:, 6] = 0.02
        initial[:, 7] = 0.01

        controls = torch.zeros(horizon, batch, 2)
        controls[..., 0] = initial[:, 3].view(1, batch) / cfg.tires[1].radius
        controls[..., 1] = 0.04

        states, elapsed = _seconds_per_call(lambda: model.rollout(initial, controls), repeats=3)

        self.assertEqual(tuple(states.shape), (horizon + 1, batch, 8))
        self.assertTrue(torch.isfinite(states).all().item())
        print(f"[DiffGM3 bicycle] batch={batch}, horizon={horizon} rollout: {elapsed * 1000.0:.2f} ms")

        next_state, aux = model(initial, controls[0], return_aux=True)
        loss = states[..., :6].square().mean()
        loss = loss + next_state[..., :6].square().mean()
        loss = loss + 1e-6 * aux["tire_forces"].square().mean()

        model.zero_grad(set_to_none=True)
        loss.backward()

        grads = {name: param.grad for name, param in model.named_parameters()}
        self.assertTrue(grads)
        for name, grad in grads.items():
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all().item(), name)

    def test_diffgm3_and_normal_gm3_one_step_parity(self) -> None:
        cfg = make_bicycle_config()
        normal = GM3(cfg)
        diff = DiffGM3(cfg, dt=0.02).double()

        state = GM3State(x=0.0, y=0.0, psi=0.02, vx=2.0, vy=0.1, r=0.05, gamma=0.02, gamma_dot=0.01)
        control = GM3Control(omega=2.0 / cfg.tires[1].radius, delta=0.03)

        normal_next = normal.step(state, control, dt=0.02).as_array()
        diff_next = diff(
            torch.tensor(state.as_array(), dtype=torch.float64),
            torch.tensor(control.as_array(), dtype=torch.float64),
        )
        diff_next_np = diff_next.detach().numpy()

        self.assertTrue(np.isfinite(diff_next_np).all())
        self.assertLess(float(np.linalg.norm(normal_next[:6] - diff_next_np[:6])), 2e-2)

    def test_diffgm3_custom_configs_rollout_backward(self) -> None:
        cases = [
            ("custom trike", make_custom_trike_config(), 2, 0.05),
            ("custom leaning direct", make_custom_leaning_direct_config(), 1, 0.04),
        ]

        for label, cfg, driven_tire_index, delta in cases:
            with self.subTest(label=label):
                model = DiffGM3(cfg, dt=0.02)
                batch = 4
                horizon = 32
                initial = torch.zeros(batch, 8)
                initial[:, 2] = 0.05
                initial[:, 3] = torch.linspace(1.8, 2.6, batch)
                initial[:, 4] = 0.05
                initial[:, 5] = 0.02
                initial[:, 6] = 0.02
                initial[:, 7] = 0.01

                controls = torch.zeros(horizon, batch, 2)
                controls[..., 0] = initial[:, 3].view(1, batch) / cfg.tires[driven_tire_index].radius
                controls[..., 1] = delta

                states, elapsed = _seconds_per_call(lambda: model.rollout(initial, controls), repeats=3)
                loss = states[..., :6].square().mean()

                model.zero_grad(set_to_none=True)
                loss.backward()

                self.assertEqual(tuple(states.shape), (horizon + 1, batch, 8))
                self.assertTrue(torch.isfinite(states).all().item())
                for name, param in model.named_parameters():
                    if not cfg.can_lean and name in {"raw_roll_inertia", "raw_roll_damping"}:
                        self.assertIsNone(param.grad, name)
                        continue
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all().item(), name)
                print(f"[DiffGM3 {label}] batch={batch}, horizon={horizon} rollout: {elapsed * 1000.0:.2f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
