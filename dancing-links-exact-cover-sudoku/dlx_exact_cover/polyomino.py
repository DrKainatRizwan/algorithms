"""Polyomino tilings and N-queens expressed as exact cover problems."""
from __future__ import annotations

from typing import Iterable

from .dlx import DLX

Cell = tuple[int, int]

# The 12 free pentominoes (standard letter names) as cell lists.
PIECES: dict[str, tuple[Cell, ...]] = {
    "F": ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
    "I": ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    "L": ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    "N": ((0, 1), (1, 1), (2, 0), (2, 1), (3, 0)),
    "P": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    "T": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "U": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    "V": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    "W": ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    "X": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "Y": ((0, 1), (1, 0), (1, 1), (2, 1), (3, 1)),
    "Z": ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
}


def _normalise(cells: Iterable[Cell]) -> tuple[Cell, ...]:
    cells = list(cells)
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def orientations(cells: Iterable[Cell]) -> list[tuple[Cell, ...]]:
    """All distinct rotations/reflections of a shape, normalised to the origin."""
    cur = list(cells)
    seen: set[tuple[Cell, ...]] = set()
    for _ in range(2):                       # reflect
        for _ in range(4):                   # rotate
            seen.add(_normalise(cur))
            cur = [(c, -r) for r, c in cur]
        cur = [(r, -c) for r, c in cur]
    return sorted(seen)


def rectangle(rows: int, cols: int) -> set[Cell]:
    """Cells of a full rectangular region."""
    return {(r, c) for r in range(rows) for c in range(cols)}


def _build_tiling(region: set[Cell], pieces: dict[str, tuple[Cell, ...]],
                  each_once: bool):
    cells = sorted(region)
    cidx = {cell: i for i, cell in enumerate(cells)}
    names = sorted(pieces)
    n_cells = len(cells)
    # With each_once the piece columns are primary; otherwise pieces are free.
    n_piece_cols = len(names) if each_once else 0
    # Cells first (primary), piece-usage columns after (primary too).
    dlx = DLX(n_cells + n_piece_cols)
    meta: list[tuple[str, tuple[Cell, ...]]] = []
    for pi, name in enumerate(names):
        for shape in orientations(pieces[name]):
            for (ar, ac) in cells:
                # anchor: first cell of the shape lands on (ar, ac)
                dr, dc = ar - shape[0][0], ac - shape[0][1]
                placed = tuple((r + dr, c + dc) for r, c in shape)
                if all(p in cidx for p in placed):
                    cols = [cidx[p] for p in placed]
                    if each_once:
                        cols.append(n_cells + pi)
                    dlx.add_option(cols)
                    meta.append((name, placed))
    return dlx, meta


def tile_region(region: set[Cell], pieces: dict[str, tuple[Cell, ...]] | None = None,
                each_once: bool = True, limit: int | None = 1):
    """Tile ``region``; returns a list of solutions.

    Each solution is a list of ``(piece_name, cells)`` placements.  When
    ``each_once`` is true every piece must be used exactly once, otherwise
    pieces may be reused any number of times.
    """
    pieces = PIECES if pieces is None else pieces
    dlx, meta = _build_tiling(region, pieces, each_once)
    return [[meta[i] for i in sol] for sol in dlx.solve(limit)]


def count_tilings(region: set[Cell], pieces: dict[str, tuple[Cell, ...]] | None = None,
                  each_once: bool = True, limit: int | None = None) -> int:
    """Number of distinct tilings (all rotations/reflections counted)."""
    pieces = PIECES if pieces is None else pieces
    dlx, _ = _build_tiling(region, pieces, each_once)
    return dlx.count(limit)


def _queens_dlx(n: int) -> DLX:
    # primary: rows 0..n-1 and columns n..2n-1; secondary: both diagonals
    dlx = DLX(2 * n, 2 * (2 * n - 1))
    for r in range(n):
        for c in range(n):
            dlx.add_option([r, n + c, 2 * n + (r + c), 2 * n + (2 * n - 1) + (r - c + n - 1)])
    return dlx


def queens_solutions(n: int, limit: int | None = None) -> list[list[int]]:
    """Solutions of N-queens as lists ``cols[row]``."""
    dlx = _queens_dlx(n)
    out = []
    for sol in dlx.solve(limit):
        cols = [0] * n
        for oid in sol:
            r, c = divmod(oid, n)
            cols[r] = c
        out.append(cols)
    return out


def count_queens(n: int) -> int:
    """Number of N-queens solutions."""
    return _queens_dlx(n).count()
