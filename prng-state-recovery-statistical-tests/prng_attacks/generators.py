"""Reference implementations of several non-cryptographic generators."""

from __future__ import annotations

import numpy as np

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


class MT19937:
    """Mersenne Twister (32-bit), with explicit access to the 624-word state."""

    N, M = 624, 397
    MATRIX_A = 0x9908B0DF
    UPPER, LOWER = 0x80000000, 0x7FFFFFFF

    def __init__(self, seed: int = 5489) -> None:
        self.state = [0] * self.N
        self.state[0] = seed & MASK32
        for i in range(1, self.N):
            prev = self.state[i - 1]
            self.state[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & MASK32
        self.index = self.N

    @classmethod
    def from_state(cls, state: list[int], index: int = 624) -> "MT19937":
        """Build a generator directly from a raw state vector."""
        if len(state) != cls.N:
            raise ValueError("state must have 624 words")
        g = cls(0)
        g.state = [int(w) & MASK32 for w in state]
        g.index = index
        return g

    def _twist(self) -> None:
        mt = self.state
        for i in range(self.N):
            y = (mt[i] & self.UPPER) | (mt[(i + 1) % self.N] & self.LOWER)
            v = mt[(i + self.M) % self.N] ^ (y >> 1)
            if y & 1:
                v ^= self.MATRIX_A
            mt[i] = v
        self.index = 0

    def next_u32(self) -> int:
        if self.index >= self.N:
            self._twist()
        y = self.state[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & MASK32

    def outputs(self, n: int) -> list[int]:
        return [self.next_u32() for _ in range(n)]


class Xorshift128Plus:
    """xorshift128+ (shift triple 23, 18, 5) with a 128-bit state."""

    def __init__(self, s0: int, s1: int) -> None:
        if (s0 | s1) == 0:
            raise ValueError("state must be non-zero")
        self.s = [s0 & MASK64, s1 & MASK64]

    def next_u64(self) -> int:
        s1, s0 = self.s[0], self.s[1]
        self.s[0] = s0
        s1 ^= (s1 << 23) & MASK64
        self.s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5)
        return (self.s[1] + s0) & MASK64

    def outputs(self, n: int) -> list[int]:
        return [self.next_u64() for _ in range(n)]


class LCG:
    """Linear congruential generator ``x -> (a x + c) mod m``.

    The public output is ``(x >> shift) & out_mask`` (defaults to the raw state).
    """

    def __init__(self, seed: int, a: int, c: int, m: int, shift: int = 0,
                 out_bits: int | None = None) -> None:
        self.x = seed % m
        self.a, self.c, self.m = a, c, m
        self.shift = shift
        bits = m.bit_length() - shift if out_bits is None else out_bits
        self.out_mask = (1 << bits) - 1

    def step(self) -> int:
        self.x = (self.a * self.x + self.c) % self.m
        return self.x

    def next(self) -> int:
        return (self.step() >> self.shift) & self.out_mask

    def outputs(self, n: int) -> list[int]:
        return [self.next() for _ in range(n)]


class JavaRandom(LCG):
    """``java.util.Random``: 48-bit LCG, ``next(32)`` returns the top 32 bits (signed in Java)."""

    A, C, M = 0x5DEECE66D, 0xB, 1 << 48

    def __init__(self, seed: int) -> None:
        super().__init__((seed ^ self.A) & (self.M - 1), self.A, self.C, self.M,
                         shift=16, out_bits=32)


def lsb_stream(words: list[int] | np.ndarray, bit: int = 0) -> np.ndarray:
    """Extract one bit position from every word as a uint8 array."""
    arr = np.asarray(words, dtype=np.uint64)
    return ((arr >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)


def bit_matrix(words: list[int] | np.ndarray, width: int) -> np.ndarray:
    """Concatenate all ``width`` bits (MSB first) of every word into one bit array."""
    arr = np.asarray(words, dtype=np.uint64)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((arr[:, None] >> shifts) & np.uint64(1)).astype(np.uint8).reshape(-1)
