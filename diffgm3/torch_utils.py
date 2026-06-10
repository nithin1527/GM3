from __future__ import annotations

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

