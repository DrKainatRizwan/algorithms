"""Bootstrap particle filter with selectable resampling."""
from __future__ import annotations
from typing import Callable
import numpy as np


def effective_sample_size(w: np.ndarray) -> float:
    """ESS = 1 / sum(w^2) for normalised weights."""
    return float(1.0 / np.sum(np.asarray(w) ** 2))


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: one uniform draw, N evenly spaced pointers."""
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    cs = np.cumsum(w)
    cs[-1] = 1.0
    return np.searchsorted(cs, pos)


def multinomial_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multinomial resampling (higher variance than systematic)."""
    cs = np.cumsum(w)
    cs[-1] = 1.0
    return np.searchsorted(cs, rng.random(len(w)))


class ParticleFilter:
    """Sequential importance resampling filter.

    Parameters
    ----------
    transition : (particles (N,n), rng) -> particles; samples the dynamics incl. process noise.
    log_likelihood : (z, particles) -> (N,) log p(z | particle).
    init_particles : (N, n) initial cloud.
    resample_threshold : resample when ESS < threshold * N.
    """

    def __init__(self, transition: Callable, log_likelihood: Callable, init_particles: np.ndarray,
                 seed: int = 0, resample_threshold: float = 0.5, method: str = "systematic"):
        self.transition = transition
        self.log_likelihood = log_likelihood
        self.particles = np.asarray(init_particles, float).copy()
        self.n = len(self.particles)
        self.w = np.full(self.n, 1.0 / self.n)
        self.rng = np.random.default_rng(seed)
        self.threshold = resample_threshold
        self.resampler = {"systematic": systematic_resample,
                          "multinomial": multinomial_resample}[method]
        self.resample_count = 0

    def predict(self):
        self.particles = self.transition(self.particles, self.rng)

    def update(self, z):
        logw = np.log(self.w + 1e-300) + self.log_likelihood(z, self.particles)
        logw -= logw.max()
        w = np.exp(logw)
        self.w = w / w.sum()
        if effective_sample_size(self.w) < self.threshold * self.n:
            idx = self.resampler(self.w, self.rng)
            self.particles = self.particles[idx]
            self.w = np.full(self.n, 1.0 / self.n)
            self.resample_count += 1

    def estimate(self) -> np.ndarray:
        """Weighted posterior mean."""
        return self.w @ self.particles

    def covariance(self) -> np.ndarray:
        d = self.particles - self.estimate()
        return (self.w[:, None] * d).T @ d

    def run(self, zs):
        est = []
        for z in zs:
            self.predict(); self.update(z)
            est.append(self.estimate())
        return np.array(est)
