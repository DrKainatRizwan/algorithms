"""Dancing Links (DLX) exact cover toolkit.

Modules
-------
dlx        : array-based Dancing Links implementation of Knuth's Algorithm X
baseline   : dictionary-of-sets Algorithm X used as a reference/benchmark
sudoku     : Sudoku encoding, solver, uniqueness check and puzzle generator
polyomino  : pentomino/tetromino tilings and N-queens as exact cover problems
"""
from .dlx import DLX, SearchStats
from .baseline import solve_dict, count_dict
from .sudoku import (SudokuSolver, parse_grid, format_grid, generate_puzzle,
                     is_valid_solution)
from .polyomino import (PIECES, orientations, tile_region, rectangle,
                        count_tilings, queens_solutions, count_queens)

__all__ = [
    "DLX", "SearchStats", "solve_dict", "count_dict", "SudokuSolver",
    "parse_grid", "format_grid", "generate_puzzle", "is_valid_solution",
    "PIECES", "orientations", "tile_region", "rectangle", "count_tilings",
    "queens_solutions", "count_queens",
]
