"""Exact privacy audits and communication models for the PIR schemes."""
from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Sequence


def view_distribution(scheme, n: int, i: int, coalition: Sequence[int]) -> Counter:
    """Exact distribution of the queries seen by ``coalition`` when retrieving ``i``.

    Enumerates the full random-coin space, so only usable for tiny ``n``.
    Every coin vector is equally likely, hence counts are the distribution.
    """
    ranges = scheme.coin_ranges(n)
    dist: Counter = Counter()
    for coins in itertools.product(*[range(r) for r in ranges]):
        qs = scheme.queries_from_coins(n, i, list(coins))
        dist[tuple(qs[s] for s in coalition)] += 1
    return dist


def is_private(scheme, n: int, coalition: Sequence[int]) -> bool:
    """True iff the coalition's view distribution is identical for every index."""
    ref = view_distribution(scheme, n, 0, coalition)
    return all(view_distribution(scheme, n, i, coalition) == ref for i in range(1, n))


def total_variation(p: Counter, q: Counter) -> float:
    """Total variation distance between two empirical distributions."""
    sp, sq = sum(p.values()), sum(q.values())
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p[k] / sp - q[k] / sq) for k in keys)


def xor_cost(n: int, record_len: int, k: int = 2) -> tuple[int, int]:
    """(upload bits, download bits) of ``k``-server XOR PIR."""
    return k * n, k * 8 * record_len


def cube_cost(n: int, record_len: int, d: int) -> tuple[int, int]:
    """(upload bits, download bits) of ``d``-dimensional cube PIR."""
    m = 1
    while m ** d < n:
        m += 1
    return 2 ** d * d * m, 2 ** d * 8 * record_len


def trivial_cost(n: int, record_len: int) -> tuple[int, int]:
    """(upload bits, download bits) of downloading the database."""
    return 0, 8 * n * record_len


def paillier_cost(n: int, record_len: int, key_bits: int, rows: int = 0) -> tuple[int, int]:
    """(upload bits, download bits) of Paillier PIR for the given layout."""
    r = rows or (math.isqrt(n - 1) + 1 if n > 1 else 1)
    r = min(r, n)
    c = math.ceil(n / r)
    chunks = math.ceil(record_len / ((key_bits - 1) // 8))
    ct = 2 * key_bits
    return c * ct, r * chunks * ct
