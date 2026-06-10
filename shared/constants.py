from __future__ import annotations

STATE_FIELDS = ("x", "y", "psi", "vx", "vy", "r", "gamma", "gamma_dot")
CONTROL_FIELDS = ("omega", "delta")

DEFAULT_GRAVITY = 9.81
DEFAULT_EPS = 1e-6
DEFAULT_MIN_NORMAL_LOAD = 1e-3

MU_BOUNDS = (0.05, 3.0)
CP_BOUNDS = (1.0, 100_000.0)
CONTACT_LENGTH_BOUNDS = (1e-4, 1.0)
ALIGN_GAIN_BOUNDS = (0.0, 2.0)

SLIP_EPS = 1e-3
MAX_KAPPA = 5.0
MIN_KAPPA = -0.95
MAX_ALPHA = 1.2
MAX_SIGMA = 10.0

