"""Union-Find (disjoint set) variants with operation counters.

Variants differ in the union rule (naive / size / rank) and the compression
rule (none / full path compression / path halving / path splitting).
"""
from __future__ import annotations

UNION_RULES = ("naive", "size", "rank")
COMPRESSION_RULES = ("none", "full", "halving", "splitting")


class UnionFind:
    """Disjoint-set forest with configurable heuristics.

    With union by rank/size and any compression the amortised cost per
    operation is O(alpha(n)); without heuristics a find can cost O(n).
    ``steps`` counts parent-pointer traversals, which makes variants
    comparable independent of interpreter speed.
    """

    def __init__(self, n: int, union: str = "rank", compression: str = "full") -> None:
        if union not in UNION_RULES:
            raise ValueError(f"unknown union rule {union!r}")
        if compression not in COMPRESSION_RULES:
            raise ValueError(f"unknown compression rule {compression!r}")
        self.parent = list(range(n))
        self.size = [1] * n
        self.rank = [0] * n
        self.union_rule = union
        self.compression = compression
        self.components = n
        self.steps = 0

    def find(self, x: int) -> int:
        p = self.parent
        c = self.compression
        if c == "none":
            while p[x] != x:
                x = p[x]
                self.steps += 1
            return x
        if c == "full":
            root = x
            while p[root] != root:
                root = p[root]
                self.steps += 1
            while p[x] != root:
                p[x], x = root, p[x]
            return root
        if c == "halving":
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
                self.steps += 1
            return x
        while p[x] != x:  # splitting
            nxt = p[x]
            p[x] = p[nxt]
            x = nxt
            self.steps += 1
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of ``a`` and ``b``; return False if already joined."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.union_rule == "size":
            if self.size[ra] < self.size[rb]:
                ra, rb = rb, ra
        elif self.union_rule == "rank":
            if self.rank[ra] < self.rank[rb]:
                ra, rb = rb, ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def set_size(self, x: int) -> int:
        return self.size[self.find(x)]

    def max_depth(self) -> int:
        """Height of the deepest node (no compression is applied here)."""
        best = 0
        for x in range(len(self.parent)):
            d = 0
            while self.parent[x] != x:
                x = self.parent[x]
                d += 1
            best = max(best, d)
        return best


class RollbackUnionFind:
    """Union by size without compression, supporting undo of the last unions."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.history: list[int] = []
        self.components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            self.history.append(-1)
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.history.append(rb)
        self.components -= 1
        return True

    def rollback(self) -> None:
        """Undo the most recent ``union`` call."""
        rb = self.history.pop()
        if rb == -1:
            return
        ra = self.parent[rb]
        self.size[ra] -= self.size[rb]
        self.parent[rb] = rb
        self.components += 1
