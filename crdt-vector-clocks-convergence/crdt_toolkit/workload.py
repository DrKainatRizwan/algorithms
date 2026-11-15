"""Random workload generators used by tests, demos and benchmarks."""
from __future__ import annotations

import random
from typing import Callable

from .crdts import CRDT, GCounter, LWWMap, ORSet, PNCounter, TwoPSet
from .sequence import RGA

Op = Callable[[CRDT, random.Random], None]


def counter_op(r: GCounter, rng: random.Random) -> None:
    r.increment(rng.randint(1, 5))


def pncounter_op(r: PNCounter, rng: random.Random) -> None:
    (r.increment if rng.random() < 0.6 else r.decrement)(rng.randint(1, 5))


def orset_op(r: ORSet, rng: random.Random) -> None:
    x = rng.randint(0, 9)
    if rng.random() < 0.65:
        r.add(x)
    else:
        r.remove(x)


def twopset_op(r: TwoPSet, rng: random.Random) -> None:
    x = rng.randint(0, 9)
    if rng.random() < 0.7:
        r.add(x)
    else:
        r.remove(x)


def lwwmap_op(r: LWWMap, rng: random.Random) -> None:
    k = rng.choice("abcde")
    t = rng.randint(1, 1000)
    if rng.random() < 0.8:
        r.put(k, rng.randint(0, 99), t)
    else:
        r.delete(k, t)


def rga_op(r: RGA, rng: random.Random) -> None:
    n = len(r)
    if n > 0 and rng.random() < 0.3:
        r.delete(rng.randrange(n))
    else:
        r.insert(rng.randint(0, n), rng.choice("abcdefghij"))


def apply_random_ops(replicas: dict[str, CRDT], op: Op, count: int, rng: random.Random) -> None:
    """Apply ``count`` random local operations at random replicas."""
    ids = sorted(replicas)
    for _ in range(count):
        op(replicas[rng.choice(ids)], rng)
