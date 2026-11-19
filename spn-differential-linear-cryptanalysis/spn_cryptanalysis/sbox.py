"""4-bit S-box analysis: difference distribution and linear approximation tables.

All tables are indexed ``[input][output]`` and are computed directly from the
definitions (no shortcuts) so they can be cross-checked against brute force.
"""
from __future__ import annotations

import numpy as np

#: S-box from Heys' tutorial on linear and differential cryptanalysis.
SBOX_HEYS: tuple[int, ...] = (0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
                              0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7)
#: The S-box of the PRESENT lightweight cipher.
SBOX_PRESENT: tuple[int, ...] = (0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                                 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2)
#: A deliberately weak, affine S-box (Gray-code map plus a constant).
SBOX_AFFINE: tuple[int, ...] = tuple(
    (x ^ (x >> 1)) ^ 0x6 for x in range(16))

SIZE = 16


def parity(x: int) -> int:
    """Return the parity (XOR of all bits) of a non-negative integer."""
    return bin(x).count("1") & 1


def is_bijective(sbox: tuple[int, ...] | list[int]) -> bool:
    """True when ``sbox`` is a permutation of ``0..15``."""
    return sorted(sbox) == list(range(SIZE))


def invert(sbox: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return the inverse permutation of a bijective S-box."""
    if not is_bijective(sbox):
        raise ValueError("S-box is not a permutation")
    inv = [0] * SIZE
    for x, y in enumerate(sbox):
        inv[y] = x
    return tuple(inv)


def random_sbox(rng: np.random.Generator) -> tuple[int, ...]:
    """Draw a uniformly random bijective S-box."""
    return tuple(int(v) for v in rng.permutation(SIZE))


def ddt(sbox: tuple[int, ...] | list[int]) -> np.ndarray:
    """Difference distribution table.

    ``ddt[a][b] = #{x : S(x) ^ S(x ^ a) = b}``; every row sums to 16.
    """
    t = np.zeros((SIZE, SIZE), dtype=np.int64)
    for a in range(SIZE):
        for x in range(SIZE):
            t[a, sbox[x] ^ sbox[x ^ a]] += 1
    return t


def lat(sbox: tuple[int, ...] | list[int]) -> np.ndarray:
    """Linear approximation table (centred counts).

    ``lat[a][b] = #{x : a.x = b.S(x)} - 8``; the bias is ``lat / 16``.
    """
    t = np.zeros((SIZE, SIZE), dtype=np.int64)
    for a in range(SIZE):
        for b in range(SIZE):
            hits = sum(parity(a & x) == parity(b & sbox[x]) for x in range(SIZE))
            t[a, b] = hits - SIZE // 2
    return t


def walsh_spectrum(sbox: tuple[int, ...] | list[int]) -> np.ndarray:
    """Walsh-Hadamard spectrum ``W[a][b] = sum_x (-1)^(a.x + b.S(x))``.

    Computed with a fast Walsh transform of each component function; equals
    ``2 * lat`` and is used to cross-check :func:`lat`.
    """
    w = np.zeros((SIZE, SIZE), dtype=np.int64)
    for b in range(SIZE):
        f = np.array([1 - 2 * parity(b & sbox[x]) for x in range(SIZE)])
        h = 1
        while h < SIZE:
            for i in range(0, SIZE, 2 * h):
                for j in range(i, i + h):
                    u, v = f[j], f[j + h]
                    f[j], f[j + h] = u + v, u - v
            h *= 2
        w[:, b] = f
    return w


def differential_uniformity(sbox: tuple[int, ...] | list[int]) -> int:
    """Largest non-trivial DDT entry (smaller is better; optimum is 4)."""
    return int(ddt(sbox)[1:, :].max())


def linearity(sbox: tuple[int, ...] | list[int]) -> int:
    """Largest absolute non-trivial LAT entry (smaller is better; optimum is 4)."""
    return int(np.abs(lat(sbox)[1:, 1:]).max())


def nonlinearity(sbox: tuple[int, ...] | list[int]) -> int:
    """Minimum distance of any component function to the affine functions."""
    w = np.abs(walsh_spectrum(sbox))[:, 1:]
    return int(SIZE // 2 - w.max() // 2)


def max_diff_prob(sbox: tuple[int, ...] | list[int]) -> float:
    """Best single-S-box differential probability over non-zero inputs."""
    return differential_uniformity(sbox) / SIZE


def max_bias(sbox: tuple[int, ...] | list[int]) -> float:
    """Best single-S-box linear bias over non-trivial masks."""
    return linearity(sbox) / SIZE
