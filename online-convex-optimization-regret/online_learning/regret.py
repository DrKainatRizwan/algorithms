"""Regret measurement against the best fixed point in hindsight, and theory bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .domains import Domain
from .learners import Learner
from .losses import LinearLoss, Loss, aggregate


@dataclass
class OnlineRun:
    """Trace of one learner on one loss sequence."""

    predictions: np.ndarray   # (T, d)
    losses: np.ndarray        # (T,) loss incurred each round

    @property
    def total_loss(self) -> float:
        return float(self.losses.sum())


def run_online(learner: Learner, losses: Sequence[Loss]) -> OnlineRun:
    """Play the full-information protocol for ``len(losses)`` rounds."""
    preds, incurred = [], []
    for loss in losses:
        x = learner.predict()
        preds.append(x)
        incurred.append(loss.value(x))
        learner.update(loss)
    return OnlineRun(np.array(preds), np.array(incurred))


def minimise_over_domain(loss: Loss, domain: Domain, max_iter: int = 3000,
                         tol: float = 1e-10) -> np.ndarray:
    """Projected FISTA with backtracking for a smooth convex ``loss`` on ``domain``."""
    if isinstance(loss, LinearLoss):
        return domain.argmin_linear(loss.g)
    x = domain.center()
    y, tk = x.copy(), 1.0
    L = getattr(loss, "lipschitz_gradient", lambda: 1.0)()
    L = max(L, 1e-8)
    for _ in range(max_iter):
        gy, fy = loss.grad(y), loss.value(y)
        while True:
            x_new = domain.project(y - gy / L)
            diff = x_new - y
            if loss.value(x_new) <= fy + gy @ diff + 0.5 * L * diff @ diff + 1e-14:
                break
            L *= 2.0
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        y = x_new + ((tk - 1.0) / t_new) * (x_new - x)
        if np.linalg.norm(x_new - x) <= tol * (1.0 + np.linalg.norm(x)):
            x = x_new
            break
        x, tk = x_new, t_new
    return x


def best_in_hindsight(losses: Sequence[Loss], domain: Domain) -> tuple[np.ndarray, float]:
    """Return ``(x*, sum_t f_t(x*))`` for the best fixed decision."""
    total = aggregate(losses)
    x = minimise_over_domain(total, domain)
    return x, total.value(x)


def regret(run: OnlineRun, losses: Sequence[Loss], domain: Domain) -> float:
    """``sum_t f_t(x_t) - min_{u in K} sum_t f_t(u)``."""
    return run.total_loss - best_in_hindsight(losses, domain)[1]


def regret_curve_linear(run: OnlineRun, gradients: np.ndarray, domain: Domain) -> np.ndarray:
    """Regret after every prefix for linear losses (exact, O(T d))."""
    cum_alg = np.cumsum(run.losses)
    gsum = np.zeros(domain.dim)
    best = np.empty(len(gradients))
    for t, g in enumerate(gradients):
        gsum = gsum + g
        best[t] = gsum @ domain.argmin_linear(gsum)
    return cum_alg - best


def growth_exponent(horizons: Sequence[int], regrets: Sequence[float]) -> float:
    """Least-squares slope of ``log regret`` vs ``log T``: regret ~ T^slope."""
    r = np.maximum(np.asarray(regrets, dtype=float), 1e-9)
    return float(np.polyfit(np.log(np.asarray(horizons, dtype=float)), np.log(r), 1)[0])


def ogd_regret_bound(diameter: float, grad_bound: float, T: int) -> float:
    """``D G sqrt(T)``: regret of OGD with ``eta = D / (G sqrt T)`` (constant step)."""
    return diameter * grad_bound * np.sqrt(T)


def strongly_convex_regret_bound(grad_bound: float, H: float, T: int) -> float:
    """``G^2 (1 + ln T) / (2H)`` for OGD with ``eta_t = 1/(H t)``."""
    return grad_bound ** 2 * (1.0 + np.log(T)) / (2.0 * H)


def hedge_regret_bound(n: int, T: int) -> float:
    """``sqrt(T ln N / 2)``: Hedge with ``eta = sqrt(8 ln N / T)``, losses in [0, 1]."""
    return float(np.sqrt(T * np.log(n) / 2.0))


def hedge_eta(n: int, T: int) -> float:
    """Optimal fixed-horizon Hedge learning rate for losses in [0, 1]."""
    return float(np.sqrt(8.0 * np.log(n) / T))
