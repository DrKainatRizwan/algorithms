"""Sudoku as an exact cover problem, with solver and puzzle generator.

Constraint columns (4 * n^2 * ... for the classic 9x9 grid, 324 in total):
cell (r, c) holds a digit, row r holds digit d, column c holds digit d and
box b holds digit d.  Each candidate placement (r, c, d) is an option that
covers exactly four columns.  Supports any box-size ``b`` (grid ``b*b``).
"""
from __future__ import annotations

import random
from typing import Sequence

from .dlx import DLX, SearchStats

Grid = list[list[int]]


def parse_grid(text: str, box: int = 3) -> Grid:
    """Parse a string of digits ('.' or '0' for blanks) into a grid."""
    n = box * box
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) != n * n:
        raise ValueError(f"expected {n * n} cells, got {len(chars)}")
    vals = [0 if ch in ".0" else int(ch) for ch in chars]
    if any(v < 0 or v > n for v in vals):
        raise ValueError("digit out of range")
    return [vals[r * n:(r + 1) * n] for r in range(n)]


def format_grid(grid: Sequence[Sequence[int]], box: int = 3) -> str:
    """Render a grid with box separators."""
    n = box * box
    lines = []
    for r, row in enumerate(grid):
        if r and r % box == 0:
            lines.append("+".join(["-" * (2 * box + 1)] * box))
        parts = []
        for b in range(box):
            parts.append(" " + " ".join(str(v) if v else "."
                                        for v in row[b * box:(b + 1) * box]) + " ")
        lines.append("|".join(parts))
    return "\n".join(lines)


def is_valid_solution(grid: Sequence[Sequence[int]], box: int = 3) -> bool:
    """True if every row, column and box contains each digit exactly once."""
    n = box * box
    full = set(range(1, n + 1))
    for i in range(n):
        if set(grid[i]) != full or {grid[r][i] for r in range(n)} != full:
            return False
    for br in range(0, n, box):
        for bc in range(0, n, box):
            if {grid[br + i][bc + j] for i in range(box) for j in range(box)} != full:
                return False
    return True


class SudokuSolver:
    """Solve and analyse Sudoku puzzles via DLX."""

    def __init__(self, box: int = 3, heuristic: str = "mrv") -> None:
        self.box = box
        self.n = box * box
        self.heuristic = heuristic
        self.last_stats = SearchStats()

    def _columns(self, r: int, c: int, d: int) -> list[int]:
        n, b = self.n, self.box
        sq = (r // b) * b + c // b
        nn = n * n
        return [r * n + c, nn + r * n + d, 2 * nn + c * n + d, 3 * nn + sq * n + d]

    def _build(self, grid: Sequence[Sequence[int]],
               rng: random.Random | None = None):
        n = self.n
        if len(grid) != n or any(len(row) != n for row in grid):
            raise ValueError("grid has wrong shape")
        dlx = DLX(4 * n * n, heuristic=self.heuristic)
        meta: list[tuple[int, int, int]] = []
        triples = []
        for r in range(n):
            for c in range(n):
                v = grid[r][c]
                if v:
                    triples.append((r, c, v - 1))
                else:
                    triples.extend((r, c, d) for d in range(n))
        if rng is not None:
            rng.shuffle(triples)
        for r, c, d in triples:
            dlx.add_option(self._columns(r, c, d))
            meta.append((r, c, d))
        return dlx, meta

    def _to_grid(self, sol: list[int], meta) -> Grid:
        out = [[0] * self.n for _ in range(self.n)]
        for oid in sol:
            r, c, d = meta[oid]
            out[r][c] = d + 1
        return out

    def solve(self, grid: Sequence[Sequence[int]]) -> Grid | None:
        """Return a solution or ``None`` if the puzzle is inconsistent."""
        try:
            dlx, meta = self._build(grid)
        except ValueError:
            raise
        sol = dlx.first_solution()
        self.last_stats = dlx.stats
        return None if sol is None else self._to_grid(sol, meta)

    def count_solutions(self, grid: Sequence[Sequence[int]],
                        limit: int | None = None) -> int:
        """Count completions of ``grid`` (capped at ``limit``)."""
        dlx, _ = self._build(grid)
        k = dlx.count(limit)
        self.last_stats = dlx.stats
        return k

    def has_unique_solution(self, grid: Sequence[Sequence[int]]) -> bool:
        return self.count_solutions(grid, limit=2) == 1


def _conflicting_clues(grid: Sequence[Sequence[int]], box: int) -> bool:
    n = box * box
    seen = set()
    for r in range(n):
        for c in range(n):
            v = grid[r][c]
            if not v:
                continue
            keys = [("r", r, v), ("c", c, v), ("b", (r // box) * box + c // box, v)]
            if any(k in seen for k in keys):
                return True
            seen.update(keys)
    return False


def generate_puzzle(seed: int, clues: int = 30, box: int = 3) -> tuple[Grid, Grid]:
    """Generate ``(puzzle, solution)`` with a unique solution.

    A random complete grid is built by solving an empty board with shuffled
    options; clues are then removed in random order whenever uniqueness
    survives, stopping at ``clues`` givens or when none can be removed.
    """
    rng = random.Random(seed)
    solver = SudokuSolver(box)
    n = box * box
    empty = [[0] * n for _ in range(n)]
    dlx, meta = solver._build(empty, rng)
    sol = dlx.first_solution()
    assert sol is not None
    solution = solver._to_grid(sol, meta)
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)
    remaining = n * n
    for r, c in cells:
        if remaining <= clues:
            break
        keep = puzzle[r][c]
        puzzle[r][c] = 0
        if solver.has_unique_solution(puzzle):
            remaining -= 1
        else:
            puzzle[r][c] = keep
    return puzzle, solution
