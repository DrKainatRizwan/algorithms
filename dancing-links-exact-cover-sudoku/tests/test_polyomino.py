import pytest
from dlx_exact_cover import (PIECES, orientations, rectangle, tile_region,
                             count_tilings, count_queens, queens_solutions)


def test_orientation_counts():
    expected = {"F": 8, "I": 2, "L": 8, "N": 8, "P": 8, "T": 4,
                "U": 4, "V": 4, "W": 4, "X": 1, "Y": 8, "Z": 4}
    for k, v in expected.items():
        assert len(orientations(PIECES[k])) == v, k


def test_all_pieces_have_five_cells():
    assert all(len(set(c)) == 5 for c in PIECES.values())


@pytest.mark.parametrize("n,expected", [(1, 1), (4, 2), (5, 10), (6, 4), (8, 92)])
def test_queens_counts(n, expected):
    assert count_queens(n) == expected


def test_queens_solutions_are_valid():
    sols = queens_solutions(6)
    assert len(sols) == 4
    for cols in sols:
        assert sorted(cols) == list(range(6))
        assert all(abs(cols[i] - cols[j]) != j - i
                   for i in range(6) for j in range(i + 1, 6))


def test_pentomino_3x20_has_eight_tilings():
    assert count_tilings(rectangle(3, 20)) == 8


def test_tiling_covers_region_exactly_once():
    sol = tile_region(rectangle(6, 10))[0]
    cells = [c for _, cs in sol for c in cs]
    assert sorted(cells) == sorted(rectangle(6, 10))
    assert sorted(n for n, _ in sol) == sorted(PIECES)


def test_domino_tilings_of_chessboard_fibonacci():
    domino = {"D": ((0, 0), (0, 1))}
    # 2xn dominoes: Fibonacci numbers; 2x6 -> 13
    assert count_tilings(rectangle(2, 6), domino, each_once=False) == 13
    # 8x8 board domino tilings
    assert count_tilings(rectangle(4, 4), domino, each_once=False) == 36


def test_untileable_region_has_no_solution():
    domino = {"D": ((0, 0), (0, 1))}
    assert tile_region(rectangle(3, 3), domino, each_once=False) == []


def test_l_tromino_tilings_of_2x3():
    tromino = {"L": ((0, 0), (1, 0), (1, 1))}
    assert count_tilings(rectangle(2, 3), tromino, each_once=False) == 2
