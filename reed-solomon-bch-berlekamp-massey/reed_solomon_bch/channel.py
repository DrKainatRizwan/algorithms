"""Noise models and interleaving for exercising the decoders."""

from __future__ import annotations

import random
from collections.abc import Sequence


def add_symbol_errors(word: Sequence[int], count: int, rng: random.Random,
                      field_size: int = 256) -> tuple[list[int], list[int]]:
    """Corrupt ``count`` distinct symbols with non-zero error values.

    Returns ``(noisy_word, positions)``.
    """
    if not 0 <= count <= len(word):
        raise ValueError("invalid error count")
    positions = sorted(rng.sample(range(len(word)), count))
    noisy = list(word)
    for p in positions:
        noisy[p] ^= rng.randrange(1, field_size)
    return noisy, positions


def add_bit_errors(word: Sequence[int], count: int,
                   rng: random.Random) -> tuple[list[int], list[int]]:
    """Flip ``count`` distinct bits of a binary word."""
    positions = sorted(rng.sample(range(len(word)), count))
    noisy = list(word)
    for p in positions:
        noisy[p] ^= 1
    return noisy, positions


def add_burst(word: Sequence[int], start: int, length: int, rng: random.Random,
              field_size: int = 256) -> list[int]:
    """Corrupt ``length`` consecutive symbols beginning at ``start``."""
    if start < 0 or start + length > len(word):
        raise ValueError("burst outside word")
    noisy = list(word)
    for p in range(start, start + length):
        noisy[p] ^= rng.randrange(1, field_size)
    return noisy


class BlockInterleaver:
    """Write ``depth`` codewords row-wise, transmit column-wise.

    A burst of length ``b`` on the channel is spread so that each codeword
    sees at most ``ceil(b / depth)`` consecutive errors.
    """

    def __init__(self, depth: int, width: int) -> None:
        if depth < 1 or width < 1:
            raise ValueError("depth and width must be positive")
        self.depth, self.width = depth, width

    def interleave(self, data: Sequence[int]) -> list[int]:
        if len(data) != self.depth * self.width:
            raise ValueError("data length must equal depth * width")
        return [data[r * self.width + c]
                for c in range(self.width) for r in range(self.depth)]

    def deinterleave(self, data: Sequence[int]) -> list[int]:
        if len(data) != self.depth * self.width:
            raise ValueError("data length must equal depth * width")
        out = [0] * len(data)
        i = 0
        for c in range(self.width):
            for r in range(self.depth):
                out[r * self.width + c] = data[i]
                i += 1
        return out
