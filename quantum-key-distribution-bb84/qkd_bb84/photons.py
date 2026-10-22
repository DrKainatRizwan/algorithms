"""Polarisation-qubit model.

A photon is described by a real polarisation angle ``alpha``.  The
rectilinear basis (index 0) has axes at 0 and pi/2, the diagonal basis
(index 1) has axes at pi/4 and 3pi/4.  Measuring in a basis whose first axis
is at ``phi`` yields outcome 0 with probability ``cos^2(alpha - phi)`` and
collapses the photon onto the axis that was observed (Malus' law).
"""

from __future__ import annotations

import numpy as np

RECTILINEAR = 0
DIAGONAL = 1


def bits_to_angles(bits: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Polarisation angle for each (bit, basis) pair."""
    bits = np.asarray(bits, dtype=np.int64)
    bases = np.asarray(bases, dtype=np.int64)
    return bases * (np.pi / 4) + bits * (np.pi / 2)


def encode(bits: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Alias of :func:`bits_to_angles` used by the sender."""
    return bits_to_angles(bits, bases)


def measure(
    angles: np.ndarray, bases: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Projective measurement.

    Returns ``(outcomes, post_angles)`` where ``post_angles`` are the
    collapsed polarisation angles.
    """
    angles = np.asarray(angles, dtype=float)
    axis = np.asarray(bases, dtype=np.int64) * (np.pi / 4)
    p0 = np.cos(angles - axis) ** 2
    outcome = (rng.random(angles.shape) >= p0).astype(np.int64)
    return outcome, axis + outcome * (np.pi / 2)
