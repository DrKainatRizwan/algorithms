"""Reference first-match linear classifier (ground truth for the tests)."""
from __future__ import annotations

from .rules import Packet, Rule


class LinearClassifier:
    """O(N) scan over rules sorted by priority."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = sorted(rules, key=lambda r: r.priority)

    def lookup(self, p: Packet) -> Rule | None:
        for r in self.rules:
            if r.matches(p):
                return r
        return None

    def match_all(self, p: Packet) -> list[Rule]:
        """Every matching rule in priority order."""
        return [r for r in self.rules if r.matches(p)]

    def classify(self, p: Packet, default: str = "deny") -> str:
        r = self.lookup(p)
        return r.action if r else default
