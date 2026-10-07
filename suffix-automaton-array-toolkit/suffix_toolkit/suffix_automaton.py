"""Suffix automaton (DAWG) with online construction in O(n) amortised."""
from __future__ import annotations


class SuffixAutomaton:
    """Minimal DFA recognising all suffixes (hence all substrings) of a text."""

    def __init__(self, text: str = "") -> None:
        self.next: list[dict[str, int]] = [{}]
        self.link: list[int] = [-1]
        self.length: list[int] = [0]
        self.is_clone: list[bool] = [False]
        self.first_end: list[int] = [-1]  # end position of first occurrence
        self.last = 0
        self._occ: list[int] | None = None
        self.n = 0
        for c in text:
            self.extend(c)

    def __len__(self) -> int:
        return len(self.next)

    def _new_state(self, length: int, link: int, nxt: dict[str, int], clone: bool, fe: int) -> int:
        self.next.append(nxt)
        self.link.append(link)
        self.length.append(length)
        self.is_clone.append(clone)
        self.first_end.append(fe)
        return len(self.next) - 1

    def extend(self, c: str) -> None:
        """Append one character to the indexed text."""
        self._occ = None
        cur = self._new_state(self.length[self.last] + 1, 0, {}, False, self.n)
        self.n += 1
        p = self.last
        while p != -1 and c not in self.next[p]:
            self.next[p][c] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = self._new_state(self.length[p] + 1, self.link[q],
                                        dict(self.next[q]), True, self.first_end[q])
                while p != -1 and self.next[p].get(c) == q:
                    self.next[p][c] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur

    def _walk(self, pattern: str) -> int:
        """Return the state reached by ``pattern`` or -1."""
        v = 0
        for c in pattern:
            v = self.next[v].get(c, -1)
            if v == -1:
                return -1
        return v

    def contains(self, pattern: str) -> bool:
        """True if ``pattern`` is a substring of the text."""
        return self._walk(pattern) != -1

    def _occurrences(self) -> list[int]:
        if self._occ is None:
            occ = [0 if self.is_clone[v] else 1 for v in range(len(self))]
            occ[0] = 0
            for v in sorted(range(1, len(self)), key=lambda v: -self.length[v]):
                occ[self.link[v]] += occ[v]
            self._occ = occ
        return self._occ

    def count(self, pattern: str) -> int:
        """Number of (possibly overlapping) occurrences of ``pattern``."""
        if pattern == "":
            return self.n + 1
        v = self._walk(pattern)
        return 0 if v == -1 else self._occurrences()[v]

    def first_occurrence(self, pattern: str) -> int:
        """Start index of the leftmost occurrence, or -1."""
        if pattern == "":
            return 0
        v = self._walk(pattern)
        return -1 if v == -1 else self.first_end[v] - len(pattern) + 1

    def distinct_substrings(self) -> int:
        """Number of distinct non-empty substrings."""
        return sum(self.length[v] - self.length[self.link[v]] for v in range(1, len(self)))

    def longest_common_substring(self, other: str) -> str:
        """Longest substring shared with ``other`` (leftmost in ``other``)."""
        v, l, best, best_end = 0, 0, 0, 0
        for i, c in enumerate(other):
            while v != 0 and c not in self.next[v]:
                v = self.link[v]
                l = self.length[v]
            if c in self.next[v]:
                v = self.next[v][c]
                l += 1
            else:
                v, l = 0, 0
            if l > best:
                best, best_end = l, i + 1
        return other[best_end - best:best_end]
