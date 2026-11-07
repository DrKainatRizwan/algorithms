# Dancing Links Exact Cover Solver for Sudoku, Pentominoes and N-Queens

## Overview
A from-scratch implementation of Knuth's Algorithm X with the Dancing Links (DLX) data structure, supporting primary and secondary columns. Three problem families are encoded as exact cover: Sudoku (with a uniqueness checker and puzzle generator), polyomino tilings (the 12 free pentominoes, dominoes, trominoes) and N-queens. A dictionary-of-sets Algorithm X is included as an independent reference implementation for cross-checking and benchmarking.

## How it works
Exact cover: given a 0/1 matrix, choose rows so that every (primary) column has exactly one 1. Algorithm X picks a column, tries each row that covers it, removes all conflicting rows and columns, and recurses. DLX stores the matrix as a toroidal doubly linked list in flat arrays; removing a node via `L[R[x]] = L[x]` is undone exactly by reversing the operation, so backtracking costs only the links touched ("dancing").

- **Heuristic:** choose the primary column with the fewest remaining rows (MRV); `first` is available for comparison.
- **Secondary columns** (at-most-once) are left out of the header row. N-queens diagonals use them.
- **Sudoku:** 324 constraint columns (cell, row-digit, column-digit, box-digit) and up to 729 options of four ones each. Clues restrict the options for that cell.
- **Tilings:** one primary column per region cell plus one per piece (when each piece is used once); each placed orientation is an option.
- **Generator:** a random complete grid comes from solving an empty board with shuffled options; clues are removed while the solution remains unique (checked by counting up to two solutions).

## Complexity
- Worst case is exponential (exact cover is NP-complete); MRV pruning makes typical instances tractable.
- Each cover/uncover costs O(number of nodes in the column's rows); a search node does work proportional to the options of the selected column.
- Space: O(number of ones in the matrix) for the link arrays.

## Usage
```python
from dlx_exact_cover import SudokuSolver, parse_grid, count_queens, count_tilings, rectangle, DLX

grid = parse_grid("000000010400000000020000000000050407008000300001090000300400200050100000000806000")
solution = SudokuSolver().solve(grid)

print(count_queens(8))                    # 92
print(count_tilings(rectangle(3, 20)))    # 8

d = DLX(3)
for opt in ([0, 1], [2], [1, 2], [0]):
    d.add_option(opt)
print(list(d.solve()))                    # [[0, 1], [1, 3]]
```

## Results
Measured with `python examples/demo.py` (CPython 3.12, single core):

| Task | Result |
|------|--------|
| 17-clue Sudoku (unique) | solved in ~2 ms, 82 search nodes |
| Generated 26-clue puzzle | solved in ~2 ms, matches generator solution |
| 3x20 pentomino rectangle | 8 tilings (2 up to symmetry), ~5 s to enumerate |
| N-queens n=8 / n=9 | 92 / 352 solutions in ~14 ms / ~56 ms |

Column heuristic, nodes to enumerate all solutions:

| Problem | first column | MRV | reduction |
|---------|--------------|-----|-----------|
| 8-queens | 2057 | 1215 | 1.7x |
| six pentominoes (LNPTUV) in 3x10, 976 tilings | 79107 | 4079 | 19.4x |

DLX against the dictionary-based Algorithm X (same heuristic, same answers):

| Problem | DLX | dict |
|---------|-----|------|
| 8-queens | 14.4 ms | 14.3 ms |
| six-piece 3x10 | 272 ms | 226 ms |

In pure Python the two are roughly on par; the interpreter overhead of indexing flat lists offsets the lower allocation cost of DLX. The main speed-up comes from the column-selection heuristic, not the data structure. DLX's advantage is the constant-memory, allocation-free backtracking that compiled implementations exploit.

## Tests
`python -m pytest -q` runs 41 tests: Knuth's worked example, structure restoration after search, secondary-column semantics, random cross-checks against the dictionary baseline, known counts (4x4 Sudoku grids = 288, queens, 3x20 pentominoes, domino tilings), and generator determinism and uniqueness.

## References
- D. E. Knuth, "Dancing Links", in *Millennial Perspectives in Computer Science*, 2000 (arXiv:cs/0011047).
- D. E. Knuth, *The Art of Computer Programming*, Vol. 4B, Section 7.2.2.1.
- R. Ao, "Algorithm X in 30 lines" (dictionary formulation).

---

**Author:** Dr. Kainat Rizwan
