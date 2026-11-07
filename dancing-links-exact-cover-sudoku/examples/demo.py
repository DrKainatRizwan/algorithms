"""Demo: Sudoku, pentomino tiling, N-queens and a DLX vs dict benchmark."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dlx_exact_cover import (SudokuSolver, generate_puzzle, format_grid, PIECES,
                             rectangle, tile_region, count_tilings, count_queens,
                             solve_dict, DLX)
from dlx_exact_cover.polyomino import _build_tiling, _queens_dlx


def timed(f):
    t = time.perf_counter(); v = f(); return v, time.perf_counter() - t


print("== Sudoku ==")
puzzle, sol = generate_puzzle(seed=7, clues=26)
print(format_grid(puzzle)); print()
s = SudokuSolver()
res, dt = timed(lambda: s.solve(puzzle))
print(format_grid(res))
print(f"solved == generator solution: {res == sol}; {dt*1000:.1f} ms; "
      f"nodes={s.last_stats.nodes} updates={s.last_stats.updates}")

print("\n== Hard 17-clue puzzle ==")
hard = ("000000010400000000020000000000050407008000300001090000300400200050100000000806000")
from dlx_exact_cover import parse_grid
g = parse_grid(hard)
res, dt = timed(lambda: s.solve(g))
print(f"solved in {dt*1000:.1f} ms, nodes={s.last_stats.nodes}, unique={s.has_unique_solution(g)}")

print("\n== Pentominoes: 3x20 rectangle ==")
n3, dt = timed(lambda: count_tilings(rectangle(3, 20)))
print(f"tilings (with rotations/reflections): {n3}  [{dt:.2f}s]")
sol = tile_region(rectangle(6, 10))[0]
board = [["."] * 10 for _ in range(6)]
for name, cells in sol:
    for r, c in cells:
        board[r][c] = name
print("one 6x10 tiling:"); print("\n".join("".join(r) for r in board))

print("\n== N-queens ==")
for n in (6, 8, 9):
    k, dt = timed(lambda: count_queens(n))
    print(f"n={n}: {k} solutions [{dt*1000:.0f} ms]")

print("\n== Benchmark: DLX vs dict Algorithm X (count all solutions) ==")
for n in (7, 8):
    d = _queens_dlx(n)
    opts = d.options
    a, ta = timed(lambda: d.count())
    b, tb = timed(lambda: sum(1 for _ in solve_dict(2 * n, 2 * (2 * n - 1), opts)))
    print(f"queens n={n}: DLX {ta*1000:7.1f} ms | dict {tb*1000:7.1f} ms | "
          f"speedup {tb/ta:4.1f}x | agree={a == b}")
d, _ = _build_tiling(rectangle(3, 10), {k: PIECES[k] for k in "LNPTUV"}, False)
opts = d.options
a, ta = timed(lambda: d.count())
b, tb = timed(lambda: sum(1 for _ in solve_dict(d.n_primary, 0, opts)))
print(f"six-piece 3x10 ({a} tilings): DLX {ta*1000:7.1f} ms | dict {tb*1000:7.1f} ms | "
      f"speedup {tb/ta:4.1f}x | agree={a == b}")

print("\n== Column heuristic: search nodes to enumerate all solutions ==")
for name, mk in (("queens n=8", lambda h: _queens_dlx(8)),
                 ("pentomino 3x10 (6 pieces)", lambda h: _build_tiling(rectangle(3, 10), {k: PIECES[k] for k in "LNPTUV"}, False)[0])):
    row = []
    for h in ("first", "mrv"):
        d = mk(h); d.heuristic = h; d.count(); row.append(d.stats.nodes)
    print(f"{name}: first-column={row[0]} nodes | MRV={row[1]} nodes | reduction {row[0]/row[1]:.1f}x")
