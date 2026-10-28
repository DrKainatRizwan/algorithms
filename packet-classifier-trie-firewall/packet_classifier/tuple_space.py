"""Tuple space search: one hash table per (src_len, dst_len) prefix tuple."""
from __future__ import annotations

from .rules import Packet, Rule, prefix_mask


class TupleSpaceClassifier:
    """Rules are grouped by prefix lengths; each group is a hash table keyed by
    the masked addresses. Lookup probes each non-empty tuple and prunes tuples
    whose best (minimum) priority cannot beat the current best."""

    def __init__(self, rules: list[Rule]) -> None:
        tables: dict[tuple[int, int], dict[tuple[int, int], list[Rule]]] = {}
        for r in sorted(rules, key=lambda r: r.priority):
            t = tables.setdefault((r.src[1], r.dst[1]), {})
            t.setdefault((r.src[0], r.dst[0]), []).append(r)
        # probe tuples holding the highest-priority rule first for early pruning
        self._order = sorted(
            tables,
            key=lambda k: min(b[0].priority for b in tables[k].values()))
        self.tables = tables
        self._best = {k: min(b[0].priority for b in tables[k].values()) for k in tables}
        self._masks = {k: (prefix_mask(k[0]), prefix_mask(k[1])) for k in tables}
        self.probes = 0

    @property
    def n_tuples(self) -> int:
        return len(self.tables)

    def lookup(self, p: Packet) -> Rule | None:
        best: Rule | None = None
        for key in self._order:
            if best is not None and self._best[key] >= best.priority:
                break  # order is sorted by best priority, nothing later can win
            ms, md = self._masks[key]
            self.probes += 1
            bucket = self.tables[key].get((p.src & ms, p.dst & md))
            if bucket is None:
                continue
            for r in bucket:
                if best is not None and r.priority >= best.priority:
                    break
                if r.match_rest(p):
                    best = r
                    break
        return best

    def classify(self, p: Packet, default: str = "deny") -> str:
        r = self.lookup(p)
        return r.action if r else default
