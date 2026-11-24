"""Online learning algorithms.

Every learner exposes ``predict() -> x_t`` and ``update(loss)``; the loss is only
revealed after the prediction is made (full-information feedback).
"""
from __future__ import annotations

import numpy as np

from .domains import Ball, Box, Domain, Simplex
from .losses import Loss


class Learner:
    """Base class holding the domain and the round counter."""

    def __init__(self, domain: Domain) -> None:
        self.domain = domain
        self.t = 0
        self.x = domain.center()

    def predict(self) -> np.ndarray:
        return self.x.copy()

    def update(self, loss: Loss) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class OnlineGradientDescent(Learner):
    """Projected online gradient descent (Zinkevich 2003).

    ``schedule``: ``"const"`` (eta), ``"sqrt"`` (eta/sqrt t) or ``"strong"``
    (1/(H t), giving O(log T) regret for H-strongly convex losses).
    """

    def __init__(self, domain: Domain, eta: float = 0.1, schedule: str = "sqrt",
                 strong_convexity: float = 0.0) -> None:
        super().__init__(domain)
        if schedule not in ("const", "sqrt", "strong"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.eta, self.schedule, self.H = eta, schedule, strong_convexity

    def step_size(self, loss: Loss) -> float:
        if self.schedule == "const":
            return self.eta
        if self.schedule == "sqrt":
            return self.eta / np.sqrt(self.t)
        H = max(self.H, loss.strong_convexity)
        if H <= 0:
            raise ValueError("'strong' schedule needs a strongly convex loss")
        return 1.0 / (H * self.t)

    def update(self, loss: Loss) -> None:
        self.t += 1
        g = loss.grad(self.x)
        self.x = self.domain.project(self.x - self.step_size(loss) * g)


class FollowTheLeader(Learner):
    """FTL: play the minimiser of the cumulative *linearised* loss so far.

    Has no regret guarantee for linear losses (it can be forced to Omega(T)).
    """

    def __init__(self, domain: Domain) -> None:
        super().__init__(domain)
        self.gsum = np.zeros(domain.dim)

    def update(self, loss: Loss) -> None:
        self.t += 1
        self.gsum = self.gsum + loss.grad(self.x)
        self.x = self.domain.argmin_linear(self.gsum)


class FTRL(Learner):
    """Follow-the-regularised-leader with a quadratic regulariser.

    ``x_{t+1} = argmin_K <sum g, x> + ||x||^2 / (2 eta)  =  Pi_K(-eta sum g)``
    (lazy projection / dual averaging).
    """

    def __init__(self, domain: Domain, eta: float = 0.1) -> None:
        super().__init__(domain)
        self.eta = eta
        self.gsum = np.zeros(domain.dim)

    def update(self, loss: Loss) -> None:
        self.t += 1
        self.gsum = self.gsum + loss.grad(self.x)
        self.x = self.domain.project(-self.eta * self.gsum)


class Hedge(Learner):
    """Multiplicative weights over N experts (simplex domain), lazy form."""

    def __init__(self, n: int, eta: float) -> None:
        super().__init__(Simplex(n))
        self.eta = eta
        self.gsum = np.zeros(n)

    def update(self, loss: Loss) -> None:
        self.t += 1
        self.gsum = self.gsum + loss.grad(self.x)
        z = -self.eta * (self.gsum - self.gsum.min())
        w = np.exp(z)
        self.x = w / w.sum()


class ExponentiatedGradient(Learner):
    """Entropic mirror descent on the simplex (Kivinen & Warmuth 1997).

    For linear losses with a constant step this is identical to :class:`Hedge`.
    """

    def __init__(self, n: int, eta: float) -> None:
        super().__init__(Simplex(n))
        self.eta = eta

    def update(self, loss: Loss) -> None:
        self.t += 1
        g = loss.grad(self.x)
        w = self.x * np.exp(-self.eta * (g - g.min()))
        self.x = w / w.sum()


class FollowThePerturbedLeader(Learner):
    """FPL over experts: pick the best expert after exponential perturbation.

    Plays a vertex of the simplex, so the *expected* loss is what is bounded; the
    returned point is the realised one-hot choice.
    """

    def __init__(self, n: int, eta: float, seed: int = 0) -> None:
        super().__init__(Simplex(n))
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        self.gsum = np.zeros(n)
        self._draw()

    def _draw(self) -> None:
        noise = self.rng.exponential(1.0 / self.eta, size=self.domain.dim)
        x = np.zeros(self.domain.dim)
        x[int(np.argmin(self.gsum - noise))] = 1.0
        self.x = x

    def update(self, loss: Loss) -> None:
        self.t += 1
        self.gsum = self.gsum + loss.grad(self.x)
        self._draw()


class AdaGrad(Learner):
    """Diagonal AdaGrad (Duchi, Hazan & Singer 2011) on a box domain.

    The diagonal-metric projection onto a box is coordinate-wise clipping, so the
    update is exact for :class:`Box`.
    """

    def __init__(self, domain: Box, eta: float = 0.5, eps: float = 1e-12) -> None:
        if not isinstance(domain, Box):
            raise TypeError("AdaGrad here is implemented for Box domains")
        super().__init__(domain)
        self.eta, self.eps = eta, eps
        self.G = np.zeros(domain.dim)

    def update(self, loss: Loss) -> None:
        self.t += 1
        g = loss.grad(self.x)
        self.G = self.G + g * g
        self.x = self.domain.project(self.x - self.eta * g / (np.sqrt(self.G) + self.eps))


class OnlineNewtonStep(Learner):
    """Online Newton Step (Hazan, Agarwal & Kale 2007) on a Euclidean ball.

    Achieves O((d/alpha) log T) regret for alpha-exp-concave losses.  The update is
    ``x <- Pi_A(x - (1/gamma) A^{-1} g)`` with ``A <- A + g g^T``; ``A^{-1}`` is kept
    with Sherman-Morrison rank-one updates.
    """

    def __init__(self, domain: Ball, gamma: float, eps: float = 1.0) -> None:
        if not isinstance(domain, Ball):
            raise TypeError("ONS here is implemented for Ball domains")
        super().__init__(domain)
        self.gamma = gamma
        d = domain.dim
        self.A = eps * np.eye(d)
        self.Ainv = np.eye(d) / eps

    @staticmethod
    def default_gamma(alpha: float, grad_bound: float, diameter: float) -> float:
        """``gamma = 0.5 min(1/(4 G D), alpha)`` from the original analysis."""
        return 0.5 * min(1.0 / (4.0 * grad_bound * diameter), alpha)

    def update(self, loss: Loss) -> None:
        self.t += 1
        g = loss.grad(self.x)
        self.A = self.A + np.outer(g, g)
        Ag = self.Ainv @ g
        self.Ainv = self.Ainv - np.outer(Ag, Ag) / (1.0 + g @ Ag)
        y = self.x - self.Ainv @ g / self.gamma
        self.x = self.domain.project_norm(y, self.A)  # type: ignore[attr-defined]
