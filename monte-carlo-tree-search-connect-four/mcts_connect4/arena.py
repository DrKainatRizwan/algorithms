"""Game and match drivers."""
from __future__ import annotations

from .board import Board
from .players import Player


def play_game(p0: Player, p1: Player, opening: list[int] | None = None) -> float:
    """Play one game; return the result from player 0's perspective."""
    board = Board.from_moves(opening or [])
    players = (p0, p1)
    while not board.is_terminal():
        col = players[board.to_move].choose(board)
        board.play(col)
    return board.result()


def match(a: Player, b: Player, games: int = 10) -> dict[str, float]:
    """Alternate colours; return score of ``a`` over ``games`` games."""
    score = 0.0
    wins = draws = losses = 0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(a, b)
        else:
            r = 1.0 - play_game(b, a)
        score += r
        wins += r == 1.0
        draws += r == 0.5
        losses += r == 0.0
    return {"score": score, "wins": wins, "draws": draws, "losses": losses,
            "games": games, "rate": score / games}
