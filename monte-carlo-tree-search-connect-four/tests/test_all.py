import random

import pytest

from mcts_connect4 import AlphaBetaPlayer, MCTS, MCTSPlayer, RandomPlayer, Board, match, play_game


def test_vertical_win():
    b = Board.from_moves([0, 1, 0, 1, 0, 1, 0])
    assert b.winner() == 0 and b.result() == 1.0


def test_horizontal_win_player1():
    b = Board.from_moves([0, 1, 0, 2, 6, 3, 6, 4])
    assert b.winner() == 1 and b.result() == 0.0


def test_diagonal_win():
    b = Board.from_moves([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
    assert b.winner() == 0


def test_no_wrap_across_columns():
    # discs at top of col 0 and bottom of col 1.. must not form a line
    b = Board.from_moves([0, 6, 0, 6, 0, 6, 1, 5, 1, 5])
    assert b.winner() is None


def test_column_fills_and_illegal():
    b = Board()
    for i in range(6):
        b.play(2)
    assert not b.can_play(2)
    with pytest.raises(ValueError):
        b.play(2)
    assert 2 not in b.legal_moves()


def test_undo_restores_state():
    b = Board.from_moves([3, 3, 4])
    key = b.key()
    b.play(4)
    b.undo(4)
    assert b.key() == key and b.moves == 3


def test_draw_game_full_board():
    # random fills until terminal; verify result consistent
    rng = random.Random(3)
    for _ in range(50):
        b = Board()
        while not b.is_terminal():
            b.play(rng.choice(b.legal_moves()))
        assert b.result() in (0.0, 0.5, 1.0)
        if b.winner() is None:
            assert b.is_full() and b.moves == 42


def test_wins_by_playing():
    b = Board.from_moves([0, 6, 1, 6, 2, 5])
    assert b.wins_by_playing(3, 0)
    assert not b.wins_by_playing(4, 0)


def test_mcts_takes_immediate_win():
    b = Board.from_moves([0, 6, 1, 6, 2, 5])
    assert MCTS(seed=1).search(b, 200) == 3


def test_mcts_blocks_immediate_loss():
    b = Board.from_moves([0, 0, 1, 1, 2])  # X threatens col 3; O to move
    assert b.wins_by_playing(3, 0)
    assert MCTS(seed=2).search(b, 800) == 3


def test_mcts_deterministic_with_seed():
    b = Board.from_moves([3, 3, 2])
    m1 = MCTS(seed=7).search(b, 300)
    m2 = MCTS(seed=7).search(b, 300)
    assert m1 == m2


def test_visits_sum_and_policy():
    e = MCTS(seed=4, solver=False)
    b = Board()
    e.search(b, 500)
    assert e.root.visits == 500
    assert sum(ch.visits for ch in e.root.children) == 500
    assert abs(sum(e.policy()) - 1.0) < 1e-9


def test_solver_proves_forced_win():
    # X has open three on bottom row (cols 1,2,3 with 0 and 4 empty): O cannot stop both
    b = Board.from_moves([1, 6, 2, 6, 3])
    e = MCTS(seed=0, solver=True)
    e.search(b, 3000)
    assert e.root.proven == 1.0  # X (the root's previous mover) wins by force


def test_terminal_search_raises():
    b = Board.from_moves([0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        MCTS().search(b, 10)


def test_alphabeta_finds_win_and_block():
    ab = AlphaBetaPlayer(depth=3)
    assert ab.choose(Board.from_moves([0, 6, 1, 6, 2, 5])) == 3
    assert ab.choose(Board.from_moves([0, 0, 1, 1, 2])) == 3


def test_mcts_beats_random():
    r = match(MCTSPlayer(300, seed=5), RandomPlayer(seed=5), games=6)
    assert r["wins"] >= 5


def test_game_runs_to_completion():
    res = play_game(MCTSPlayer(50, seed=1), RandomPlayer(seed=1))
    assert res in (0.0, 0.5, 1.0)


def test_bad_rollout_rejected():
    with pytest.raises(ValueError):
        MCTS(rollout="nope")
