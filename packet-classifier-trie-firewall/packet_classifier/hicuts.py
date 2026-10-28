"""HiCuts-style decision tree: recursively cut the 5-D header space."""
from __future__ import annotations

from dataclasses import dataclass, field

from .rules import Packet, Rule

_FULL = ((0, (1 << 32) - 1), (0, (1 << 32) - 1), (0, 65535), (0, 65535), (0, 255))


@dataclass
class _Tree:
    dim: int = -1
    lo: int = 0
    step: int = 1
    children: list["_Tree"] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)  # leaf rules, priority sorted


class HiCutsClassifier:
    """Decision tree built by equal-sized cuts on the most discriminating field.

    ``binth`` is the leaf capacity, ``cuts`` the fan-out per internal node and
    ``max_depth`` bounds recursion. Rules spanning a cut are replicated in every
    child they intersect (the classic memory/speed trade-off).
    """

    def __init__(self, rules: list[Rule], binth: int = 8, cuts: int = 8,
                 max_depth: int = 12) -> None:
        self.binth, self.cuts, self.max_depth = binth, cuts, max_depth
        self.n_nodes = 0
        self.n_stored = 0
        ordered = sorted(rules, key=lambda r: r.priority)
        self.root = self._build(ordered, _FULL, 0)

    def _build(self, rules: list[Rule], box, depth: int) -> _Tree:
        self.n_nodes += 1
        if len(rules) <= self.binth or depth >= self.max_depth:
            self.n_stored += len(rules)
            return _Tree(rules=rules)
        dim = self._pick_dim(rules, box)
        if dim < 0:
            self.n_stored += len(rules)
            return _Tree(rules=rules)
        lo, hi = box[dim]
        span = hi - lo + 1
        n = min(self.cuts, span)
        if n < 2:
            self.n_stored += len(rules)
            return _Tree(rules=rules)
        step = -(-span // n)
        node = _Tree(dim=dim, lo=lo, step=step)
        progress = False
        parts = []
        for i in range(n):
            c_lo = lo + i * step
            c_hi = min(hi, c_lo + step - 1)
            if c_lo > hi:
                break
            sub = [r for r in rules
                   if r.ranges()[dim][0] <= c_hi and r.ranges()[dim][1] >= c_lo]
            if len(sub) < len(rules):
                progress = True
            parts.append((sub, c_lo, c_hi))
        if not progress:  # cutting does not separate anything: stop
            self.n_stored += len(rules)
            return _Tree(rules=rules)
        for sub, c_lo, c_hi in parts:
            child_box = box[:dim] + ((c_lo, c_hi),) + box[dim + 1:]
            node.children.append(self._build(sub, child_box, depth + 1))
        return node

    @staticmethod
    def _pick_dim(rules: list[Rule], box) -> int:
        """Field with the most distinct clipped rule projections."""
        best, best_score = -1, 0
        for d in range(5):
            lo, hi = box[d]
            if lo == hi:
                continue
            proj = {(max(r.ranges()[d][0], lo), min(r.ranges()[d][1], hi)) for r in rules}
            if len(proj) > best_score:
                best, best_score = d, len(proj)
        return best

    def lookup(self, p: Packet) -> Rule | None:
        node = self.root
        f = p.fields()
        while node.dim >= 0:
            idx = (f[node.dim] - node.lo) // node.step
            node = node.children[min(idx, len(node.children) - 1)]
        for r in node.rules:
            if r.matches(p):
                return r
        return None

    def classify(self, p: Packet, default: str = "deny") -> str:
        r = self.lookup(p)
        return r.action if r else default

    def depth(self) -> int:
        def d(n: _Tree) -> int:
            return 1 + max((d(c) for c in n.children), default=0)
        return d(self.root)

    def replication(self) -> float:
        """Stored rule copies divided by distinct rules (>= 1)."""
        distinct = {r.priority for r in self._leaf_rules(self.root)}
        return self.n_stored / max(1, len(distinct))

    def _leaf_rules(self, n: _Tree):
        if n.dim < 0:
            yield from n.rules
        for c in n.children:
            yield from self._leaf_rules(c)
