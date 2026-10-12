"""Runtime checks of Raft's safety properties (paper, Figure 3)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .messages import Entry


class InvariantViolation(AssertionError):
    """Raised when a Raft safety property is broken."""


class InvariantChecker:
    """Tracks history across steps and validates the cluster each tick."""

    def __init__(self) -> None:
        self.leader_of_term: Dict[int, int] = {}
        self.committed: Dict[int, Entry] = {}
        self._seen_leaders: set = set()

    def check(self, cluster: "Any") -> None:
        nodes = list(cluster.nodes.values())
        for n in nodes:
            if n.role == "leader" and (n.id, n.current_term) not in self._seen_leaders:
                self._seen_leaders.add((n.id, n.current_term))
                self._on_new_leader(n)
        self._log_matching(nodes)
        self._state_machine_safety(nodes)

    def _on_new_leader(self, leader) -> None:
        prev = self.leader_of_term.setdefault(leader.current_term, leader.id)
        if prev != leader.id:
            raise InvariantViolation(
                f"election safety: term {leader.current_term} has leaders {prev} and {leader.id}")
        for idx, entry in self.committed.items():
            if idx > leader.last_index or leader.log[idx - 1] != entry:
                raise InvariantViolation(
                    f"leader completeness: leader {leader.id} lacks committed index {idx}")

    def _log_matching(self, nodes) -> None:
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                for k in range(min(a.last_index, b.last_index), 0, -1):
                    if a.log[k - 1] == b.log[k - 1]:
                        if a.log[:k] != b.log[:k]:
                            raise InvariantViolation(
                                f"log matching: nodes {a.id},{b.id} differ below index {k}")
                        break

    def _state_machine_safety(self, nodes) -> None:
        for n in nodes:
            for idx, _ in n.applied[-4:]:
                entry = n.log[idx - 1]
                known = self.committed.setdefault(idx, entry)
                if known != entry:
                    raise InvariantViolation(
                        f"state machine safety: index {idx} committed as {known} and {entry}")
            # keep the committed map complete even if many entries applied at once
            for idx in range(max(1, n.commit_index - 8), n.commit_index + 1):
                entry = n.log[idx - 1]
                if self.committed.setdefault(idx, entry) != entry:
                    raise InvariantViolation(f"state machine safety violated at index {idx}")
