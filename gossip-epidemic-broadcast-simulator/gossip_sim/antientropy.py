"""Anti-entropy reconciliation of replicated last-writer-wins stores."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

Entry = tuple[int, int, str]  # (timestamp, origin replica id, value)


@dataclass
class Replica:
    """A replica holding key -> (timestamp, origin, value); highest tuple wins."""

    rid: int
    data: dict[str, Entry] = field(default_factory=dict)

    def put(self, key: str, value: str, ts: int) -> None:
        self.merge_entry(key, (ts, self.rid, value))

    def merge_entry(self, key: str, entry: Entry) -> bool:
        cur = self.data.get(key)
        if cur is None or entry > cur:
            self.data[key] = entry
            return True
        return False

    def digest(self) -> str:
        """Order-independent content hash used to skip identical replicas."""
        h = hashlib.sha256()
        for k in sorted(self.data):
            ts, rid, val = self.data[k]
            h.update(f"{k}|{ts}|{rid}|{val};".encode())
        return h.hexdigest()


def reconcile(a: Replica, b: Replica, mode: str = "push-pull") -> int:
    """Exchange state between two replicas; returns number of entries transferred."""
    if a.digest() == b.digest():
        return 0
    moved = 0
    if mode in ("push", "push-pull"):
        for k, e in list(a.data.items()):
            moved += b.merge_entry(k, e)
    if mode in ("pull", "push-pull"):
        for k, e in list(b.data.items()):
            moved += a.merge_entry(k, e)
    return moved


def anti_entropy_round(replicas: list[Replica], rng: np.random.Generator,
                       mode: str = "push-pull") -> int:
    """One round: each replica reconciles with one uniformly random peer."""
    n = len(replicas)
    moved = 0
    for i in range(n):
        j = int(rng.integers(n - 1))
        j += j >= i
        if mode == "push":
            moved += reconcile(replicas[i], replicas[j], "push")
        elif mode == "pull":
            moved += reconcile(replicas[i], replicas[j], "pull")
        else:
            moved += reconcile(replicas[i], replicas[j], "push-pull")
    return moved


def is_converged(replicas: list[Replica]) -> bool:
    d = replicas[0].digest()
    return all(r.digest() == d for r in replicas[1:])


def run_anti_entropy(replicas: list[Replica], rng: np.random.Generator,
                     mode: str = "push-pull", max_rounds: int = 200) -> int:
    """Run rounds until all replicas agree; returns rounds used (-1 if not converged)."""
    for r in range(max_rounds + 1):
        if is_converged(replicas):
            return r
        anti_entropy_round(replicas, rng, mode)
    return -1
