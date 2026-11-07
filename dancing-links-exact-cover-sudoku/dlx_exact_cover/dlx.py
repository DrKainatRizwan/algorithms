"""Knuth's Dancing Links with Algorithm X.

The constraint matrix is stored as a toroidal doubly linked list held in flat
integer arrays.  ``cover``/``uncover`` remove and restore a column and all
rows that intersect it in O(1) per node, which makes backtracking cheap.

Columns are either *primary* (must be covered exactly once) or *secondary*
(covered at most once).  Secondary columns are not linked into the header
row, so the search never selects them, yet they still prune conflicting
options.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass
class SearchStats:
    """Counters collected during a search."""
    nodes: int = 0          # recursive calls to the search routine
    updates: int = 0        # link removals performed by cover operations
    solutions: int = 0
    max_depth: int = 0


class DLX:
    """Exact cover solver using dancing links.

    Parameters
    ----------
    n_primary:
        Number of primary columns, indexed ``0 .. n_primary-1``.
    n_secondary:
        Number of secondary columns, indexed ``n_primary .. n_primary+n_secondary-1``.
    heuristic:
        ``"mrv"`` picks the primary column with fewest remaining rows
        (Knuth's S heuristic); ``"first"`` picks the leftmost column.
    """

    def __init__(self, n_primary: int, n_secondary: int = 0,
                 heuristic: str = "mrv") -> None:
        if n_primary < 0 or n_secondary < 0:
            raise ValueError("column counts must be non-negative")
        if heuristic not in ("mrv", "first"):
            raise ValueError("heuristic must be 'mrv' or 'first'")
        self.n_primary = n_primary
        self.n_cols = n_primary + n_secondary
        self.heuristic = heuristic
        n = self.n_cols
        self.root = n
        size = n + 1
        self.L = list(range(size))
        self.R = list(range(size))
        self.U = list(range(size))
        self.D = list(range(size))
        self.C = list(range(size))
        self.S = [0] * size            # column sizes (valid for header nodes)
        self.row_of = [-1] * size      # option id of each node
        # link primary headers (and root) into a circular row
        chain = list(range(n_primary)) + [self.root]
        for a, b in zip(chain, chain[1:] + chain[:1]):
            self.R[a] = b
            self.L[b] = a
        self.options: list[tuple[int, ...]] = []
        self.stats = SearchStats()

    # ------------------------------------------------------------------ build
    def add_option(self, columns: Sequence[int]) -> int:
        """Add an option (matrix row) covering ``columns``; returns its id."""
        cols = tuple(columns)
        if len(set(cols)) != len(cols):
            raise ValueError("duplicate column in option")
        if not cols:
            raise ValueError("option must cover at least one column")
        for c in cols:
            if not 0 <= c < self.n_cols:
                raise ValueError(f"column {c} out of range")
        rid = len(self.options)
        self.options.append(cols)
        first = -1
        for c in cols:
            node = len(self.L)
            self.L.append(node); self.R.append(node)
            self.U.append(self.U[c]); self.D.append(c)
            self.C.append(c); self.S.append(0); self.row_of.append(rid)
            self.D[self.U[c]] = node
            self.U[c] = node
            self.S[c] += 1
            if first < 0:
                first = node
            else:  # splice into the row's circular list, left of ``first``
                last = self.L[first]
                self.R[last] = node; self.L[node] = last
                self.R[node] = first; self.L[first] = node
        return rid

    # ------------------------------------------------------- cover / uncover
    def _cover(self, c: int) -> None:
        L, R, U, D, C, S = self.L, self.R, self.U, self.D, self.C, self.S
        if c < self.n_primary:
            R[L[c]] = R[c]
            L[R[c]] = L[c]
        i = D[c]
        while i != c:
            j = R[i]
            while j != i:
                U[D[j]] = U[j]
                D[U[j]] = D[j]
                S[C[j]] -= 1
                self.stats.updates += 1
                j = R[j]
            i = D[i]

    def _uncover(self, c: int) -> None:
        L, R, U, D, C, S = self.L, self.R, self.U, self.D, self.C, self.S
        i = U[c]
        while i != c:
            j = L[i]
            while j != i:
                S[C[j]] += 1
                U[D[j]] = j
                D[U[j]] = j
                j = L[j]
            i = U[i]
        if c < self.n_primary:
            R[L[c]] = c
            L[R[c]] = c

    def _choose(self) -> int:
        R, S = self.R, self.S
        c = R[self.root]
        if self.heuristic == "first":
            return c
        best, best_size = c, S[c]
        while c != self.root:
            if S[c] < best_size:
                best, best_size = c, S[c]
                if best_size <= 1:
                    break
            c = R[c]
        return best

    def _select(self, r: int) -> None:
        j = self.R[r]
        while j != r:
            self._cover(self.C[j])
            j = self.R[j]

    def _unselect(self, r: int) -> None:
        j = self.L[r]
        while j != r:
            self._uncover(self.C[j])
            j = self.L[j]

    # ----------------------------------------------------------------- search
    def solve(self, limit: int | None = None) -> Iterator[list[int]]:
        """Yield solutions as sorted lists of option ids.

        ``limit`` stops the enumeration after that many solutions.
        """
        self.stats = SearchStats()
        if limit is not None and limit <= 0:
            return
        partial: list[int] = []
        yield from self._search(partial, limit)

    def _search(self, partial: list[int], limit: int | None) -> Iterator[list[int]]:
        st = self.stats
        st.nodes += 1
        st.max_depth = max(st.max_depth, len(partial))
        if self.R[self.root] == self.root:
            st.solutions += 1
            yield sorted(self.row_of[r] for r in partial)
            return
        c = self._choose()
        if self.S[c] == 0:
            return
        self._cover(c)
        r = self.D[c]
        while r != c:
            partial.append(r)
            self._select(r)
            yield from self._search(partial, limit)
            self._unselect(r)
            partial.pop()
            if limit is not None and st.solutions >= limit:
                break
            r = self.D[r]
        self._uncover(c)

    def first_solution(self) -> list[int] | None:
        """Return one solution or ``None`` if the problem is infeasible."""
        for sol in self.solve(limit=1):
            return sol
        return None

    def count(self, limit: int | None = None) -> int:
        """Count solutions (optionally capped at ``limit``)."""
        return sum(1 for _ in self.solve(limit))

    def is_consistent(self) -> bool:
        """Check structural invariants of the linked structure (for tests)."""
        for c in range(self.n_cols):
            n, i = 0, self.D[c]
            while i != c:
                if self.U[self.D[i]] != i or self.C[i] != c:
                    return False
                n += 1
                i = self.D[i]
            if n != self.S[c]:
                return False
        return True
