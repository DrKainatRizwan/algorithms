"""Static rule-set analysis: shadowing, redundancy and conflicts."""
from __future__ import annotations

from .rules import Rule


def find_shadowed(rules: list[Rule]) -> list[tuple[Rule, Rule]]:
    """Pairs ``(shadowed, shadower)``: a lower-priority rule fully covered by a
    single higher-priority rule can never fire."""
    ordered = sorted(rules, key=lambda r: r.priority)
    out = []
    for j, r in enumerate(ordered):
        for q in ordered[:j]:
            if q.covers(r):
                out.append((r, q))
                break
    return out


def find_conflicts(rules: list[Rule]) -> list[tuple[Rule, Rule]]:
    """Overlapping rule pairs with different actions where neither covers the
    other (true ambiguity that depends on rule order)."""
    ordered = sorted(rules, key=lambda r: r.priority)
    out = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a.action != b.action and a.overlaps(b) and not a.covers(b) and not b.covers(a):
                out.append((a, b))
    return out
