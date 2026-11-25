"""Linear-algebra state recovery for xorshift128+ from least-significant output bits."""

from __future__ import annotations

from .generators import Xorshift128Plus
from .gf2 import solve_gf2

W = 64


def _shl(w: list[int], k: int) -> list[int]:
    return [0] * k + w[: W - k]


def _shr(w: list[int], k: int) -> list[int]:
    return w[k:] + [0] * k


def _xor(*ws: list[int]) -> list[int]:
    return [_fold(bits) for bits in zip(*ws)]


def _fold(bits: tuple[int, ...]) -> int:
    v = 0
    for b in bits:
        v ^= b
    return v


def lsb_equations(n: int) -> list[int]:
    """Masks over the 128 initial state bits giving the LSB of each of ``n`` outputs.

    Variable j < 64 is bit j of ``s[0]``; variable 64 + j is bit j of ``s[1]``.
    The LSB of a sum has no carry-in, so it is the XOR of the operand LSBs.
    """
    s = [[1 << j for j in range(W)], [1 << (W + j) for j in range(W)]]
    eqs = []
    for _ in range(n):
        s1, s0 = s[0], s[1]
        s[0] = s0
        s1 = _xor(s1, _shl(s1, 23))
        s[1] = _xor(s1, s0, _shr(s1, 18), _shr(s0, 5))
        eqs.append(s[1][0] ^ s0[0])
    return eqs


def recover_xorshift128plus(lsbs: list[int]) -> Xorshift128Plus | None:
    """Recover the generator from the LSBs of consecutive outputs.

    Returns a generator positioned *after* the observed outputs (so it predicts
    the next full 64-bit value), or ``None`` if the equations are inconsistent
    or underdetermined.  About 128 bits are needed; a few more guard against degeneracy.
    """
    n = len(lsbs)
    if n < 128:
        raise ValueError("need at least 128 output bits")
    eqs = lsb_equations(n)
    sol = solve_gf2(eqs, [int(b) for b in lsbs], 2 * W)
    if sol is None:
        return None
    s0 = sum(b << j for j, b in enumerate(sol[:W]))
    s1 = sum(b << j for j, b in enumerate(sol[W:]))
    if (s0 | s1) == 0:
        return None
    gen = Xorshift128Plus(s0, s1)
    replay = [gen.next_u64() & 1 for _ in range(n)]
    if replay != [int(b) for b in lsbs]:
        return None
    return gen
