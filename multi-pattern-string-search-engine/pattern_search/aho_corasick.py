"""Aho-Corasick automaton with full goto/failure/output links."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True, order=True)
class Match:
    """A pattern occurrence: ``text[start:end] == pattern``."""
    start: int
    end: int
    pattern_id: int
    pattern: str


class AhoCorasick:
    """Multi-pattern matcher.

    Build is O(sum |p| * sigma_used); scanning is O(n + z) with z outputs.
    Transitions are stored as per-node dicts; failure links are resolved lazily
    during scan, with a dictionary link ``dict_link`` that skips non-terminal
    suffix states so reporting costs O(z).
    """

    def __init__(self, patterns: Iterable[str] = ()) -> None:
        self.next: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.depth: list[int] = [0]
        self.out: list[list[int]] = [[]]       # pattern ids ending exactly at node
        self.dict_link: list[int] = [-1]
        self.patterns: list[str] = []
        self._built = False
        for p in patterns:
            self.add(p)
        self.build()

    def add(self, pattern: str) -> int:
        """Insert a pattern (non-empty); returns its id. Call ``build`` afterwards."""
        if not pattern:
            raise ValueError("empty pattern not allowed")
        node = 0
        for ch in pattern:
            nxt = self.next[node].get(ch)
            if nxt is None:
                nxt = len(self.next)
                self.next.append({})
                self.fail.append(0)
                self.depth.append(self.depth[node] + 1)
                self.out.append([])
                self.dict_link.append(-1)
                self.next[node][ch] = nxt
            node = nxt
        pid = len(self.patterns)
        self.patterns.append(pattern)
        self.out[node].append(pid)
        self._built = False
        return pid

    def build(self) -> None:
        """Compute failure and dictionary links by BFS."""
        queue: deque[int] = deque()
        for child in self.next[0].values():
            self.fail[child] = 0
            self.dict_link[child] = -1
            queue.append(child)
        while queue:
            u = queue.popleft()
            for ch, v in self.next[u].items():
                f = self.fail[u]
                while f and ch not in self.next[f]:
                    f = self.fail[f]
                f = self.next[f].get(ch, 0)
                if f == v:
                    f = 0
                self.fail[v] = f
                self.dict_link[v] = f if self.out[f] else self.dict_link[f]
                queue.append(v)
        self._built = True

    def step(self, state: int, ch: str) -> int:
        """Automaton transition with failure fallback."""
        while state and ch not in self.next[state]:
            state = self.fail[state]
        return self.next[state].get(ch, 0)

    def iter_matches(self, text: str) -> Iterator[Match]:
        """Yield all occurrences ordered by end position (then longer first)."""
        if not self._built:
            self.build()
        state = 0
        for i, ch in enumerate(text):
            state = self.step(state, ch)
            node = state if self.out[state] else self.dict_link[state]
            while node > 0:
                for pid in self.out[node]:
                    p = self.patterns[pid]
                    yield Match(i + 1 - len(p), i + 1, pid, p)
                node = self.dict_link[node]

    def find_all(self, text: str) -> list[Match]:
        """All occurrences, sorted by (start, end, id)."""
        return sorted(self.iter_matches(text))

    def count(self, text: str) -> dict[str, int]:
        """Occurrence counts per distinct pattern."""
        counts: dict[str, int] = {p: 0 for p in self.patterns}
        for m in self.iter_matches(text):
            counts[m.pattern] += 1
        return counts

    def contains_any(self, text: str) -> bool:
        """Early-exit membership test."""
        return next(self.iter_matches(text), None) is not None

    def num_states(self) -> int:
        return len(self.next)
