"""Player implementations: random, alpha-beta baseline and MCTS."""
from __future__ import annotations

import random
from typing import Protocol

from .board import Board, COLS
from .mcts import MCTS

_ORDER = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))  # centre first


class Player(Protocol):
    name: str

    def choose(self, board: Board) -> int: ...


class RandomPlayer:
    def __init__(self, seed: int = 0) -> None:
        self.name = "random"
        self.rng = random.Random(seed)

    def choose(self, board: Board) -> int:
        return self.rng.choice(board.legal_moves())


class MCTSPlayer:
    def __init__(self, iterations: int = 1000, c: float = 1.4, rollout: str = "heuristic",
                 solver: bool = True, seed: int = 0, name: str | None = None) -> None:
        self.iterations = iterations
        self.name = name or f"mcts-{iterations}"
        self.engine = MCTS(c=c, rollout=rollout, solver=solver, seed=seed)

    def choose(self, board: Board) -> int:
        return self.engine.search(board, self.iterations)


def _evaluate(board: Board, me: int) -> int:
    """Cheap static evaluation: centre control plus immediate threats."""
    from .board import H
    centre = 0
    for r in range(6):
        bit = 1 << (3 * H + r)
        centre += bool(board.bits[me] & bit) - bool(board.bits[1 - me] & bit)
    return 3 * centre + 5 * (board.threats(me) - board.threats(1 - me))


class AlphaBetaPlayer:
    """Fixed-depth negamax with alpha-beta pruning and centre-first ordering."""

    def __init__(self, depth: int = 4, seed: int = 0) -> None:
        self.depth = depth
        self.name = f"alphabeta-{depth}"
        self.nodes = 0

    def choose(self, board: Board) -> int:
        self.nodes = 0
        best, best_val = None, -10**9
        alpha = -10**9
        for c in _ORDER:
            if not board.can_play(c):
                continue
            board.play(c)
            val = -self._negamax(board, self.depth - 1, -10**9, -alpha)
            board.undo(c)
            if val > best_val:
                best, best_val = c, val
            alpha = max(alpha, val)
        assert best is not None
        return best

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        # value from the view of the player to move; previous mover may have won
        w = board.winner()
        if w is not None:
            return -(1000 + depth)
        if board.is_full():
            return 0
        if depth == 0:
            return _evaluate(board, board.to_move)
        best = -10**9
        for c in _ORDER:
            if not board.can_play(c):
                continue
            board.play(c)
            val = -self._negamax(board, depth - 1, -beta, -alpha)
            board.undo(c)
            best = max(best, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best
