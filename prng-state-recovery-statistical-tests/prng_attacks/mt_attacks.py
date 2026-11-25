"""Mersenne Twister state recovery: untempering, cloning, rewinding, seed recovery."""

from __future__ import annotations

from .generators import MT19937, MASK32


def temper(y: int) -> int:
    """MT19937 output tempering."""
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & MASK32


def _undo_right(y: int, shift: int) -> int:
    """Invert ``y ^= y >> shift`` on 32-bit words."""
    x = y
    for _ in range(32 // shift + 1):
        x = y ^ (x >> shift)
    return x & MASK32


def _undo_left(y: int, shift: int, mask: int) -> int:
    """Invert ``y ^= (y << shift) & mask`` on 32-bit words."""
    x = y
    for _ in range(32 // shift + 1):
        x = y ^ ((x << shift) & mask)
    return x & MASK32


def untemper(z: int) -> int:
    """Exact inverse of :func:`temper` (the tempering map is a bijection)."""
    z = _undo_right(z, 18)
    z = _undo_left(z, 15, 0xEFC60000)
    z = _undo_left(z, 7, 0x9D2C5680)
    z = _undo_right(z, 11)
    return z & MASK32


def clone_mt19937(outputs: list[int]) -> MT19937:
    """Clone a generator from 624 consecutive outputs aligned to a block boundary.

    The returned generator continues with the victim's *next* output.
    """
    if len(outputs) < MT19937.N:
        raise ValueError("need 624 consecutive outputs")
    state = [untemper(o) for o in outputs[-MT19937.N:]]
    return MT19937.from_state(state, index=MT19937.N)


def rewind_twist(state: list[int]) -> list[int]:
    """Invert one twist, returning the previous 624-word block.

    Bits 0..30 of word 0 do not influence the next block, so they are
    unrecoverable and returned as zero.
    """
    n, m = MT19937.N, MT19937.M
    new = list(state)
    old = [0] * n
    upper_of = [0] * n   # bit 31 of old[i]
    lower_of = [0] * n   # bits 0..30 of old[i]
    for i in range(n - 1, -1, -1):
        partner = new[i - (n - m)] if i >= n - m else old[i + m]
        t = new[i] ^ partner
        if t & 0x80000000:
            y = (((t ^ MT19937.MATRIX_A) << 1) | 1) & MASK32
        else:
            y = (t << 1) & MASK32
        upper_of[i] = y & MT19937.UPPER
        if i + 1 < n:
            lower_of[i + 1] = y & MT19937.LOWER
        # old[i] needs its lower bits from step i-1, which runs later; fill after.
        old[i] = upper_of[i]
        if i + 1 < n:
            old[i + 1] |= lower_of[i + 1]
    return old


def recover_seed(first_block: list[int]) -> int | None:
    """Recover the 32-bit ``init_genrand`` seed from a generator's first 624 outputs."""
    n = MT19937.N
    if len(first_block) < n:
        raise ValueError("need the first 624 outputs")
    initial = rewind_twist([untemper(o) for o in first_block[:n]])
    inv = pow(1812433253, -1, 1 << 32)
    # state[1] = f * (s ^ s>>30) + 1, with s = state[0]
    u = ((initial[1] - 1) * inv) & MASK32
    seed = _undo_right(u, 30)
    cand = MT19937(seed)
    if all(cand.state[i] == initial[i] for i in range(1, n)):
        return seed
    return None
