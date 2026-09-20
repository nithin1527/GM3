"""Interchangeable differentiable tire laws for DiffGM3.

Ported from ``gm3-models`` (``src/gm3_models/tires/models.py``): the force
laws, their smoothing and their conventions are unchanged. Two things differ,
both because these laws are meant to be calibrated here rather than compared
at fixed coefficients:

* ``cx`` / ``cy`` are log-parametrized between ``STIFFNESS_BOUNDS`` instead of
  softplus. A cornering stiffness runs from tens to tens of thousands of N/rad,
  and a softplus raw coordinate is linear up there, so Adam at lr = 0.01 moved
  a 4,500 N/rad tire by about 1 N/rad over a whole fit. This is the same
  change ``raw_cp`` needed (see ``torch_utils.raw_log_bounded``).
* The Burckhardt curve coefficients and the Pacejka shape and curvature
  factors can be freed (``trainable_coefficients`` / ``trainable_shape``).
  Both default to frozen, which reproduces ``gm3-models`` exactly.

Conventions
-----------
* ``kappa > 0`` produces positive longitudinal force.
* ``alpha > 0`` produces negative lateral force.
* Forces are road-on-tire forces in the tire frame.
* ``Mz`` is zero for the four replacement models; aligning moment is outside
  the scope of these minimal formulations.

Published formulations
----------------------
* Pacejka: Bakker, Pacejka, and Lidner, SAE 890087 (1989),
  doi:10.4271/890087. Equation cross-check: Cabrera et al., Sensors 18(3),
  896 (2018), Eq. (1), doi:10.3390/s18030896.
* Dugoff: Dugoff, Fancher, and Segel, SAE 700377 (1970),
  doi:10.4271/700377.
* Fiala handling forces: Li, Wu, Zhou, and Yao, "Study on Roll Instability
  Mechanism and Stability Index of Articulated Steering Vehicles,"
  Mathematical Problems in Engineering (2016), article 7816503, Section 3,
  doi:10.1155/2016/7816503. Constant friction is used here.
* Burckhardt: Burckhardt, Fahrwerktechnik: Radschlupf-Regelsysteme,
  Vogel (1993), ISBN 978-3-8023-0477-4. The combined-slip vector form follows
  Floren, Khajepour, and Hashemi, Proc. 2021 ACC, pp. 436-441, Eqs. (5)-(8),
  doi:10.23919/ACC50511.2021.9483421. Example road coefficients: He and Zhang,
  Mathematical Problems in Engineering (2020), Table 1,
  doi:10.1155/2020/4251027.

Numerical regularizations and closures are implementation choices, not claims
about the cited papers: regularized norms, smooth saturation, Fiala residual
lateral capacity, Pacejka radial projection, and optional Burckhardt peak
normalization.

Operating domain: finite kappa > -1, abs(alpha) < pi/2, nonnegative normal
loads and positive mu. Output forces are multiplied by Fz/sqrt(Fz^2+eps^2),
giving exactly zero at zero load. Input tensors have shape (..., n_tires).
Run ``python -m diffgm3.tire_models`` from the repo root for the self-tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from gm3.diffgm3.tire import brush_forces
from gm3.diffgm3.torch_utils import bounded, log_bounded, raw_bounded, raw_log_bounded, raw_positive
from gm3.shared.constants import STIFFNESS_BOUNDS


@dataclass
class TireInputs:
    kappa: torch.Tensor
    alpha: torch.Tensor
    normal_loads: torch.Tensor
    gamma: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor
    steering_angles: torch.Tensor
    tire_radius: torch.Tensor
    wheelbase: torch.Tensor
    eps: torch.Tensor
    min_normal_load: torch.Tensor
    lean_mask: torch.Tensor
    has_leaning_tires: bool
    longitudinal_speed: torch.Tensor | None = None


class TireModel(nn.Module):
    name = "custom"

    def forward(
        self,
        inputs: TireInputs,
        parameters: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BrushTire(TireModel):
    """Unchanged differentiable GM3 brush-tire baseline."""

    name = "brush"

    def forward(self, inputs, parameters):
        kwargs = vars(inputs).copy()
        kwargs.pop("kappa")
        kwargs.pop("longitudinal_speed")
        return brush_forces(**kwargs, **parameters)


def smooth_abs(x: torch.Tensor, epsilon: torch.Tensor | float) -> torch.Tensor:
    """Differentiable ``abs(x)`` used only at singular denominators."""

    return torch.sqrt(x.square() + torch.as_tensor(epsilon, dtype=x.dtype, device=x.device).square())


def smooth_cap_one(x: torch.Tensor, width: torch.Tensor | float = 1e-4) -> torch.Tensor:
    """Smooth approximation of ``min(x, 1)`` for nonnegative ``x``."""

    width = torch.as_tensor(width, dtype=x.dtype, device=x.device)
    smooth_max = 0.5 * (x + 1.0 + torch.sqrt((x - 1.0).square() + width.square()))
    return x / smooth_max


def smooth_lower_bound(
    x: torch.Tensor,
    lower: torch.Tensor | float,
    width: torch.Tensor | float,
) -> torch.Tensor:
    """Smooth approximation of ``max(x, lower)``."""

    lower = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    width = torch.as_tensor(width, dtype=x.dtype, device=x.device)
    delta = x - lower
    return lower + 0.5 * (delta + torch.sqrt(delta.square() + width.square()))


def friction_circle(
    fx: torch.Tensor,
    fy: torch.Tensor,
    capacity: torch.Tensor,
    width: torch.Tensor | float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth radial projection onto ``sqrt(Fx^2 + Fy^2) <= capacity``."""

    radius_squared = (fx / capacity).square() + (fy / capacity).square()
    width = torch.as_tensor(width, dtype=fx.dtype, device=fx.device)
    smooth_max = 0.5 * (
        1.0
        + radius_squared
        + torch.sqrt((radius_squared - 1.0).square() + width.square())
    )
    scale = torch.rsqrt(smooth_max)
    return fx * scale, fy * scale


def magic_formula(
    slip: torch.Tensor,
    stiffness_factor: torch.Tensor,
    shape_factor: torch.Tensor,
    peak_factor: torch.Tensor,
    curvature_factor: torch.Tensor,
) -> torch.Tensor:
    """Basic four-parameter Pacejka Magic Formula."""

    bx = stiffness_factor * slip
    return peak_factor * torch.sin(
        shape_factor
        * torch.atan(bx - curvature_factor * (bx - torch.atan(bx)))
    )


def burckhardt_mu(
    slip: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
) -> torch.Tensor:
    """Burckhardt friction curve ``c1*(1-exp(-c2*s)) - c3*s``."""

    return c1 * (-torch.expm1(-c2 * slip)) - c3 * slip


BURCKHARDT_ROADS = {
    "dry_asphalt": (1.280, 23.99, 0.520),
    "dry_cobblestone": (1.371, 6.46, 0.670),
    "dry_cement": (1.197, 25.17, 0.540),
    "wet_asphalt": (0.857, 33.82, 0.350),
    "wet_cobblestone": (0.400, 33.71, 0.120),
    "snow": (0.195, 94.13, 0.0646),
    "ice": (0.050, 306.39, 0.0010),
}


class StiffnessTire(TireModel):
    """Common storage for longitudinal and cornering stiffness."""

    def __init__(
        self,
        config,
        *,
        cx=None,
        cy=None,
        trainable_stiffness: bool = False,
        transition_width: float = 1e-4,
        force_epsilon: float = 1e-6,
        slip_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.n_tires = len(config.tires)
        base = [2.0 * tire.cp * tire.contact_length**2 for tire in config.tires]

        for name, value in (("cx", base if cx is None else cx), ("cy", base if cy is None else cy)):
            values = self._values(value, name, positive=True)
            raw = torch.stack([raw_log_bounded(float(v), *STIFFNESS_BOUNDS) for v in values])
            setattr(self, f"raw_{name}", nn.Parameter(raw, requires_grad=trainable_stiffness))

        for name, value in (
            ("transition_width", transition_width),
            ("force_epsilon", force_epsilon),
            ("slip_epsilon", slip_epsilon),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            self.register_buffer(name, torch.tensor(float(value)))

    def _values(self, value, name, *, positive=False):
        values = torch.as_tensor(value, dtype=torch.get_default_dtype())
        values = values.expand(self.n_tires).clone()
        if not torch.isfinite(values).all() or (positive and (values <= 0.0).any()):
            raise ValueError(f"{name} must contain finite positive values")
        return values

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "cx": log_bounded(self.raw_cx, *STIFFNESS_BOUNDS),
            "cy": log_bounded(self.raw_cy, *STIFFNESS_BOUNDS),
            "transition_width": self.transition_width,
            "force_epsilon": self.force_epsilon,
            "slip_epsilon": self.slip_epsilon,
        }

    def normal_load(self, inputs: TireInputs) -> torch.Tensor:
        return smooth_abs(inputs.normal_loads, self.force_epsilon)

    def capacity(self, inputs: TireInputs, parameters) -> torch.Tensor:
        return parameters["mu"] * self.normal_load(inputs)

    def finish(self, fx: torch.Tensor, fy: torch.Tensor, inputs: TireInputs):
        contact = inputs.normal_loads / self.normal_load(inputs)
        return fx * contact, fy * contact, torch.zeros_like(fx)


class FialaTire(StiffnessTire):
    """Minimal longitudinal-and-lateral Fiala handling model.

    Longitudinal force uses the critical-slip expression. Lateral force uses
    the standard cubic Fiala expression. Residual lateral capacity is an
    explicit GM3 combined-slip closure, not attributed to Li et al.
    Algebraic smooth-limit forms below avoid evaluating singular inactive
    branches, and are smooth for positive capacity.
    """

    name = "fiala"

    def forward(self, i: TireInputs, p: dict[str, torch.Tensor]):
        tire = self.physical_parameters()
        cx, cy = tire["cx"], tire["cy"]
        epsilon = self.force_epsilon
        capacity = self.capacity(i, p)

        # Equivalent to Cx*kappa below |Cx*kappa|=D/2, and to
        # sign(kappa)*(D-D^2/(4*Cx*|kappa|)) above it, as widths -> 0.
        qx = cx * i.kappa
        magnitude_x = smooth_abs(qx, epsilon)
        m = smooth_lower_bound(magnitude_x, capacity / 2.0,
                               self.transition_width * capacity)
        direction_x = qx / m
        a = capacity / (4.0 * m)
        fx = capacity * direction_x * (1.0 - a)

        # Stable form of 1-(Fx/D)^2. No subtraction of two large forces.
        residual = (1.0 - direction_x.square()) + direction_x.square() * a * (2.0 - a)
        lateral_capacity = capacity * torch.sqrt(residual)

        # Fiala cubic q - q*|q|/(3D) + q^3/(27D^2), then saturation.
        qy = -cy * torch.tan(i.alpha)
        magnitude_y = smooth_abs(qy, epsilon)
        t = smooth_cap_one(magnitude_y / (3.0 * lateral_capacity), self.transition_width)
        fy = lateral_capacity * (qy / magnitude_y) * t * (3.0 - 3.0*t + t.square())
        return self.finish(fx, fy, i)


class DugoffTire(StiffnessTire):
    """Standard steady-state combined-slip Dugoff model."""

    name = "dugoff"

    def forward(self, i: TireInputs, p: dict[str, torch.Tensor]):
        tire = self.physical_parameters()
        cx, cy = tire["cx"], tire["cy"]
        epsilon = self.force_epsilon
        capacity = self.capacity(i, p)

        # 1+kappa is positive for the normal slip-ratio domain kappa > -1.
        denominator = smooth_lower_bound(1.0 + i.kappa, self.slip_epsilon, self.slip_epsilon)
        qx = cx * i.kappa
        qy = -cy * torch.tan(i.alpha)
        demand = torch.sqrt(qx.square() + qy.square() + epsilon.square())

        lam = capacity * denominator / (2.0 * demand)
        limited_lam = smooth_cap_one(lam, self.transition_width)
        saturation = limited_lam * (2.0 - limited_lam)

        fx = qx / denominator * saturation
        fy = qy / denominator * saturation
        return self.finish(fx, fy, i)


class BurckhardtTire(StiffnessTire):
    """Combined-slip Burckhardt model.

    The scalar Burckhardt curve is evaluated at resultant slip, then resolved
    along the longitudinal/lateral slip direction. By default, each curve is
    evaluated with its absolute friction level. Optional peak normalization is
    a comparison-design choice, not part of the published model. In the default
    mode p['mu'] is unused; road coefficients set friction. Set
    match_framework_mu=True explicitly for a matched-friction experiment.
    The default road table is a longitudinal curve example, NOT a calibrated
    lateral curve or the fitted coefficients used by Floren et al.

    The zero-slip slope of an axis is ``(c1*c2 - c3) * Fz`` (times
    ``mu / peak`` when matched), which is this law's cornering stiffness: it
    has no ``cx`` / ``cy``. ``trainable_coefficients=True`` frees both triples
    in log space so that slope can be calibrated.
    """

    name = "burckhardt"

    def __init__(
        self,
        config,
        *,
        road: str = "dry_asphalt",
        coefficients_x=None,
        coefficients_y=None,
        match_framework_mu: bool = False,
        trainable_coefficients: bool = False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        if road not in BURCKHARDT_ROADS:
            raise ValueError(f"unknown road {road!r}; choose {tuple(BURCKHARDT_ROADS)}")

        default = BURCKHARDT_ROADS[road]
        coefficients_x = default if coefficients_x is None else coefficients_x
        coefficients_y = default if coefficients_y is None else coefficients_y
        for name, value in (("coefficients_x", coefficients_x), ("coefficients_y", coefficients_y)):
            values = self._check_coefficients(value, name)
            setattr(self, f"log_{name}", nn.Parameter(values.log(), requires_grad=trainable_coefficients))
        self.match_framework_mu = bool(match_framework_mu)
        # The radial model has no independent Cx/Cy parameters.
        self.raw_cx.requires_grad_(False)
        self.raw_cy.requires_grad_(False)

    @property
    def coefficients_x(self) -> torch.Tensor:
        return self.log_coefficients_x.exp()

    @property
    def coefficients_y(self) -> torch.Tensor:
        return self.log_coefficients_y.exp()

    @staticmethod
    def _check_coefficients(coefficients, name):
        values = torch.as_tensor(coefficients, dtype=torch.get_default_dtype())
        if values.shape != (3,) or not torch.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError(f"{name} must be three finite positive values")
        c1, c2, c3 = values
        if c3 >= c1 * (-torch.expm1(-c2)):
            raise ValueError(f"{name} must produce positive friction through slip=1")
        return values

    @staticmethod
    def _peak(coefficients: torch.Tensor) -> torch.Tensor:
        c1, c2, c3 = coefficients.unbind()
        peak_slip = torch.clamp(torch.log(c1 * c2 / c3) / c2, 0.0, 1.0)
        return burckhardt_mu(peak_slip, c1, c2, c3)

    def physical_parameters(self):
        return {
            **super().physical_parameters(),
            "coefficients_x": self.coefficients_x,
            "coefficients_y": self.coefficients_y,
            "match_framework_mu": self.match_framework_mu,
        }

    def forward(self, i: TireInputs, p: dict[str, torch.Tensor]):
        epsilon = self.slip_epsilon
        load = self.normal_load(i)

        # Floren et al. combined-slip direction; minus alpha matches GM3 Fy.
        sx = i.kappa
        sy = -i.alpha
        resultant = torch.sqrt(sx.square() + sy.square() + epsilon.square())
        curve_slip = smooth_cap_one(resultant, self.transition_width)

        coefficients_x, coefficients_y = self.coefficients_x, self.coefficients_y
        c1x, c2x, c3x = coefficients_x.unbind()
        c1y, c2y, c3y = coefficients_y.unbind()
        mu_x = burckhardt_mu(curve_slip, c1x, c2x, c3x)
        mu_y = burckhardt_mu(curve_slip, c1y, c2y, c3y)

        if self.match_framework_mu:
            mu_x = mu_x * p["mu"] / self._peak(coefficients_x)
            mu_y = mu_y * p["mu"] / self._peak(coefficients_y)

        fx = load * (sx / resultant) * mu_x
        fy = load * (sy / resultant) * mu_y
        return self.finish(fx, fy, i)


class MagicFormulaTire(StiffnessTire):
    """Basic B-C-D-E Pacejka Magic Formula with a shared friction circle.

    ``trainable_shape=True`` frees C (kept in ``(0, 2)``) and E (kept below 1)
    on both axes; B follows from the stiffness and D from ``mu * Fz``.
    """

    name = "pacejka"

    def __init__(
        self,
        config,
        *,
        shape_x=1.65,
        shape_y=1.30,
        curvature_x=0.50,
        curvature_y=-0.20,
        trainable_shape: bool = False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        for axis, shape, curvature in (("x", shape_x, curvature_x), ("y", shape_y, curvature_y)):
            shape = self._values(shape, f"shape_{axis}")
            curvature = self._values(curvature, f"curvature_{axis}")
            if ((shape <= 0.) | (shape > 2.)).any() or (curvature > 1.).any():
                raise ValueError("This force model requires 0 < C <= 2 and E <= 1")
            raw_shape = torch.stack([raw_bounded(float(v), 0.0, 2.0) for v in shape])
            raw_curvature = torch.stack([raw_positive(1.0 - float(v)) for v in curvature])
            setattr(self, f"raw_shape_{axis}", nn.Parameter(raw_shape, requires_grad=trainable_shape))
            setattr(self, f"raw_curvature_{axis}", nn.Parameter(raw_curvature, requires_grad=trainable_shape))

    def physical_parameters(self):
        return {
            **super().physical_parameters(),
            "shape_x": bounded(self.raw_shape_x, 0.0, 2.0),
            "shape_y": bounded(self.raw_shape_y, 0.0, 2.0),
            "curvature_x": 1.0 - F.softplus(self.raw_curvature_x),
            "curvature_y": 1.0 - F.softplus(self.raw_curvature_y),
        }

    def forward(self, i: TireInputs, p: dict[str, torch.Tensor]):
        tire = self.physical_parameters()
        capacity = self.capacity(i, p)

        # B is chosen so the zero-slip slopes equal Cx and Cy.
        bx = tire["cx"] / (tire["shape_x"] * capacity)
        by = tire["cy"] / (tire["shape_y"] * capacity)

        fx = magic_formula(
            i.kappa,
            bx,
            tire["shape_x"],
            capacity,
            tire["curvature_x"],
        )
        fy = -magic_formula(
            i.alpha,
            by,
            tire["shape_y"],
            capacity,
            tire["curvature_y"],
        )
        fx, fy = friction_circle(fx, fy, capacity, self.transition_width)
        return self.finish(fx, fy, i)


TIRE_MODELS = {
    cls.name: cls
    for cls in (BrushTire, FialaTire, DugoffTire, BurckhardtTire, MagicFormulaTire)
}


def make_tire(name: str, config, **kwargs) -> TireModel:
    aliases = {"magic_formula": "pacejka", "gm3_brush": "brush"}
    name = aliases.get(name.lower(), name.lower())
    if name not in TIRE_MODELS:
        raise ValueError(f"unknown tire model {name!r}; choose {tuple(TIRE_MODELS)}")
    if name == "brush":
        if kwargs:
            raise ValueError("brush coefficients are supplied through TireConfig")
        return BrushTire()
    return TIRE_MODELS[name](config, **kwargs)


def _self_test():
    """Local checks of the four comparison laws, not a calibration."""
    from types import SimpleNamespace

    config = SimpleNamespace(tires=[SimpleNamespace(cp=1e5, contact_length=.05)] * 2)
    for model_name in ("fiala", "dugoff", "burckhardt", "pacejka"):
        options = {"match_framework_mu": True} if model_name == "burckhardt" else {}
        model = make_tire(model_name, config, cx=1000., cy=1200., **options).double()

        def evaluate(kappa, alpha, loads, mu):
            # Comparison laws only consume these fields of TireInputs.
            inputs = SimpleNamespace(kappa=kappa, alpha=alpha, normal_loads=loads)
            return model(inputs, {"mu": mu})[:2]

        k = torch.tensor([[.025, -.08], [.2, .4]], dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([[.03, -.04], [.1, -.2]], dtype=torch.float64, requires_grad=True)
        loads = torch.full_like(k, 100., requires_grad=True)
        mu = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(evaluate, (k, alpha, loads, mu))
        assert torch.autograd.gradgradcheck(evaluate, (k, alpha, loads, mu))
        fx, fy = evaluate(k, alpha, loads, mu)
        assert torch.all(torch.sqrt(fx.square()+fy.square()) <= mu*loads + 1e-7)
        assert torch.all(fx*k >= 0.) and torch.all(fy*alpha <= 0.)
        z = torch.zeros_like(k, requires_grad=True)
        fx0, fy0 = evaluate(z, z, loads, mu)
        assert torch.equal(fx0, torch.zeros_like(fx0))
        assert torch.equal(fy0, torch.zeros_like(fy0))
        for force in (fx0, fy0):
            gradient, = torch.autograd.grad(force.sum(), z, retain_graph=True)
            assert torch.isfinite(gradient).all()
        unloaded = evaluate(k, alpha, torch.zeros_like(loads), mu)
        assert all(torch.equal(force, torch.zeros_like(force)) for force in unloaded)
        print(f"{model_name}: gradcheck, gradgradcheck, signs, bound, zero slip/load PASS")


if __name__ == "__main__":
    _self_test()
