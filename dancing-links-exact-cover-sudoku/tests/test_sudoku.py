import pytest
from dlx_exact_cover import (SudokuSolver, parse_grid, format_grid,
                             generate_puzzle, is_valid_solution)

EASY = ("530070000600195000098000060800060003400803001700020006"
        "060000280000419005000080079")
HARD17 = ("000000010400000000020000000000050407008000300001090000"
          "300400200050100000000806000")


def test_solves_classic_puzzle():
    g = parse_grid(EASY)
    sol = SudokuSolver().solve(g)
    assert is_valid_solution(sol)
    assert all(g[r][c] in (0, sol[r][c]) for r in range(9) for c in range(9))
    assert sol[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]


def test_solves_17_clue_puzzle_uniquely():
    s = SudokuSolver()
    g = parse_grid(HARD17)
    assert is_valid_solution(s.solve(g))
    assert s.has_unique_solution(g)


def test_contradictory_puzzle_unsolvable():
    g = parse_grid(EASY)
    g[0][2] = 5                      # duplicate 5 in row 0
    assert SudokuSolver().solve(g) is None


def test_empty_grid_has_many_solutions():
    g = [[0] * 9 for _ in range(9)]
    s = SudokuSolver()
    assert s.count_solutions(g, limit=5) == 5
    assert not s.has_unique_solution(g)


def test_4x4_sudoku_total_count():
    g = [[0] * 4 for _ in range(4)]
    assert SudokuSolver(box=2).count_solutions(g) == 288


def test_parse_format_roundtrip():
    g = parse_grid(EASY)
    text = format_grid(g)
    digits = "".join(ch if ch != "." else "0" for ch in text if ch in ".123456789")
    assert parse_grid(digits) == g


def test_parse_errors():
    with pytest.raises(ValueError):
        parse_grid("123")


def test_generator_deterministic_and_unique():
    p1, s1 = generate_puzzle(seed=3, clues=30)
    p2, s2 = generate_puzzle(seed=3, clues=30)
    assert (p1, s1) == (p2, s2)
    assert is_valid_solution(s1)
    solver = SudokuSolver()
    assert solver.has_unique_solution(p1)
    assert solver.solve(p1) == s1
    assert sum(v != 0 for row in p1 for v in row) <= 36


def test_is_valid_solution_rejects_bad_grid():
    sol = SudokuSolver().solve(parse_grid(EASY))
    sol[0][0], sol[0][1] = sol[0][1], sol[0][0]
    assert not is_valid_solution(sol)
