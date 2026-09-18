from __future__ import annotations

import math

import torch


def raw_positive(value: float, minimum: float = 0.0) -> torch.Tensor:
    shifted = max(float(value) - minimum, 1e-8)
    if shifted > 20.0:
        return torch.tensor(shifted, dtype=torch.get_default_dtype())
    return torch.log(torch.expm1(torch.tensor(shifted, dtype=torch.get_default_dtype())))


def raw_bounded(value: float, lower: float, upper: float) -> torch.Tensor:
    if upper <= lower:
        raise ValueError("upper bound must be greater than lower bound")
    scaled = (float(value) - lower) / (upper - lower)
    scaled = min(max(scaled, 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(scaled, dtype=torch.get_default_dtype()))


def bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def smooth_abs(value: torch.Tensor, eps: float | torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.square() + eps * eps)



def raw_log_bounded(value: float, lower: float, upper: float) -> torch.Tensor:
    """Logit of the position of ``log(value)`` within ``[log(lower), log(upper)]``.

    For a parameter that spans decades (brush stiffness ``cp`` runs 1e4-1e7
    N/m^2 across tire sizes) a linear sigmoid over the range puts almost the
    whole raw axis on the top decade and gives an optimizer steps of a
    thousand at the bottom of the range and a million at the top. In log
    space every step is a fixed percentage.
    """
    if lower <= 0.0:
        raise ValueError("log bounds need a positive lower bound")
    return raw_bounded(math.log(float(value)), math.log(lower), math.log(upper))


def log_bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.exp(bounded(raw, math.log(lower), math.log(upper)))
