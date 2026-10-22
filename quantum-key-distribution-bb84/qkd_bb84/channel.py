"""Lossy, noisy quantum channel and an intercept-resend eavesdropper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .photons import measure


@dataclass
class InterceptResend:
    """Eve measures a fraction of the photons in random bases and resends.

    Attributes:
        fraction: probability that a given photon is attacked.
    """

    fraction: float = 1.0

    def attack(
        self, angles: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(new_angles, attacked_mask, eve_bits)``.

        ``eve_bits`` are Eve's measurement results (meaningful only where
        ``attacked_mask`` is True) and ``eve_bases`` is hidden inside the
        returned angles.
        """
        n = angles.shape[0]
        mask = rng.random(n) < self.fraction
        eve_bases = rng.integers(0, 2, n)
        bits, post = measure(angles, eve_bases, rng)
        out = np.where(mask, post, angles)
        self.last_bases = eve_bases
        return out, mask, bits


@dataclass
class Channel:
    """Fibre-like channel.

    Attributes:
        loss: probability a photon is lost.
        misalignment: polarisation rotation (radians) applied to every photon.
        dark_count: probability that Bob's detector fires randomly on a lost
            photon slot (yielding a uniformly random bit).
        eve: optional eavesdropper.
    """

    loss: float = 0.0
    misalignment: float = 0.0
    dark_count: float = 0.0
    eve: InterceptResend | None = None

    def transmit(
        self, angles: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Send photons; return ``(angles_at_bob, detected_mask, eve_mask)``.

        Photons lost in the channel may still register as a dark count; those
        are returned as detected with a random polarisation.
        """
        n = angles.shape[0]
        eve_mask = np.zeros(n, dtype=bool)
        cur = angles
        if self.eve is not None:
            cur, eve_mask, _ = self.eve.attack(cur, rng)
        cur = cur + self.misalignment
        survived = rng.random(n) >= self.loss
        dark = (~survived) & (rng.random(n) < self.dark_count)
        # A dark click carries no information: give it a random angle so the
        # subsequent measurement yields a uniformly random bit.
        random_angles = rng.integers(0, 4, n) * (np.pi / 4)
        cur = np.where(dark, random_angles, cur)
        return cur, survived | dark, eve_mask
