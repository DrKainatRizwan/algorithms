"""Problem container for mixed-integer linear programs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MIP:
    """min c.x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  lb <= x <= ub,  x_j integer for j in `integer`."""

    c: np.ndarray
    A_ub: np.ndarray = field(default=None)  # type: ignore[assignment]
    b_ub: np.ndarray = field(default=None)  # type: ignore[assignment]
    A_eq: np.ndarray = field(default=None)  # type: ignore[assignment]
    b_eq: np.ndarray = field(default=None)  # type: ignore[assignment]
    lb: np.ndarray = field(default=None)  # type: ignore[assignment]
    ub: np.ndarray = field(default=None)  # type: ignore[assignment]
    integer: np.ndarray = field(default=None)  # type: ignore[assignment]
    name: str = "mip"

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        n = self.c.size
        self.A_ub = np.zeros((0, n)) if self.A_ub is None else np.atleast_2d(np.asarray(self.A_ub, float))
        self.b_ub = np.zeros(0) if self.b_ub is None else np.asarray(self.b_ub, float).ravel()
        self.A_eq = np.zeros((0, n)) if self.A_eq is None else np.atleast_2d(np.asarray(self.A_eq, float))
        self.b_eq = np.zeros(0) if self.b_eq is None else np.asarray(self.b_eq, float).ravel()
        self.lb = np.zeros(n) if self.lb is None else np.asarray(self.lb, float)
        self.ub = np.full(n, np.inf) if self.ub is None else np.asarray(self.ub, float)
        self.integer = np.ones(n, bool) if self.integer is None else np.asarray(self.integer, bool)
        if self.A_ub.shape != (self.b_ub.size, n) or self.A_eq.shape != (self.b_eq.size, n):
            raise ValueError("constraint dimensions do not match c")
        if not np.all(np.isfinite(self.lb)):
            raise ValueError("lower bounds must be finite")

    @property
    def n(self) -> int:
        return self.c.size

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Check all constraints, bounds and integrality of a candidate point."""
        x = np.asarray(x, float)
        if np.any(x < self.lb - tol) or np.any(x > self.ub + tol):
            return False
        if self.b_ub.size and np.any(self.A_ub @ x > self.b_ub + tol):
            return False
        if self.b_eq.size and np.any(np.abs(self.A_eq @ x - self.b_eq) > tol):
            return False
        frac = np.abs(x[self.integer] - np.round(x[self.integer]))
        return bool(np.all(frac <= tol))

    def objective(self, x: np.ndarray) -> float:
        return float(self.c @ np.asarray(x, float))


def brute_force(mip: MIP, limit: int = 2_000_000) -> tuple[float, np.ndarray | None]:
    """Exhaustive search for pure-integer problems with finite bounds (testing oracle)."""
    if not np.all(mip.integer):
        raise ValueError("brute force supports pure integer problems only")
    if not np.all(np.isfinite(mip.ub)):
        raise ValueError("brute force needs finite upper bounds")
    ranges = [range(int(np.ceil(l - 1e-9)), int(np.floor(u + 1e-9)) + 1) for l, u in zip(mip.lb, mip.ub)]
    size = 1
    for r in ranges:
        size *= max(len(r), 1)
    if size > limit:
        raise ValueError("search space too large")
    best, arg = np.inf, None
    for point in itertools.product(*ranges):
        x = np.array(point, float)
        if mip.is_feasible(x):
            v = mip.objective(x)
            if v < best - 1e-12:
                best, arg = v, x
    return best, arg
