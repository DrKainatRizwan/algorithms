"""UCT Monte Carlo Tree Search.

Selection uses UCB1, expansion adds one child per iteration, and simulation is a
random playout optionally biased by tactical heuristics (win now / block now).
Terminal wins and losses can be propagated as proven values (MCTS-Solver).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .board import Board, COLS


@dataclass
class Node:
    """Search tree node. ``wins`` is from the view of the player who made ``move``."""

    move: int | None
    parent: "Node | None"
    mover: int  # player who played `move` into this node
    untried: list[int] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0
    proven: float | None = None  # +1 mover wins, -1 mover loses, 0 draw

    def ucb(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return math.inf
        return self.wins / self.visits + c * math.sqrt(math.log(parent_visits) / self.visits)

    def depth(self) -> int:
        return 0 if not self.children else 1 + max(ch.depth() for ch in self.children)

    def size(self) -> int:
        return 1 + sum(ch.size() for ch in self.children)


def heuristic_move(board: Board, rng: random.Random) -> int:
    """Play an immediate win, else block an immediate loss, else random."""
    legal = board.legal_moves()
    me = board.to_move
    for c in legal:
        if board.wins_by_playing(c, me):
            return c
    for c in legal:
        if board.wins_by_playing(c, 1 - me):
            return c
    return rng.choice(legal)


class MCTS:
    """UCT searcher.

    Parameters
    ----------
    c: exploration constant.
    rollout: ``"random"`` or ``"heuristic"``.
    solver: propagate proven terminal results up the tree.
    seed: RNG seed for reproducibility.
    """

    def __init__(self, c: float = 1.4, rollout: str = "heuristic",
                 solver: bool = True, seed: int | None = 0) -> None:
        if rollout not in ("random", "heuristic"):
            raise ValueError("rollout must be 'random' or 'heuristic'")
        self.c = c
        self.rollout = rollout
        self.solver = solver
        self.rng = random.Random(seed)
        self.root: Node | None = None

    # -- core loop -----------------------------------------------------
    def search(self, board: Board, iterations: int = 1000) -> int:
        """Run ``iterations`` simulations and return the most visited move."""
        if board.is_terminal():
            raise ValueError("cannot search a terminal position")
        root = Node(None, None, 1 - board.to_move, untried=board.legal_moves())
        self.root = root
        for _ in range(iterations):
            if root.proven is not None:
                break
            self._iterate(root, board)
        return self.best_move(root)

    def _iterate(self, root: Node, board: Board) -> None:
        b = board.copy()
        node = root
        # selection
        while not node.untried and node.children and node.proven is None:
            node = self._select(node)
            b.play(node.move)
        # expansion
        if node.untried and node.proven is None and not b.is_terminal():
            mv = node.untried.pop(self.rng.randrange(len(node.untried)))
            mover = b.to_move
            b.play(mv)
            child = Node(mv, node, mover,
                         untried=[] if b.is_terminal() else b.legal_moves())
            node.children.append(child)
            node = child
        # simulation
        res = b.result()
        if res is not None and self.solver and node.proven is None:
            w = b.winner()
            node.proven = 1.0 if w == node.mover else 0.0 if w is None else -1.0
        if res is None:
            res = self._playout(b)
        self._backprop(node, res)

    def _select(self, node: Node) -> Node:
        best, best_score = None, -math.inf
        for ch in node.children:
            if ch.proven is not None and ch.proven > 0:
                score = math.inf  # this move wins outright for the player choosing it
            elif ch.proven is not None and ch.proven < 0:
                score = -math.inf
            else:
                score = ch.ucb(node.visits, self.c)
            if score > best_score:
                best, best_score = ch, score
        return best if best is not None else node.children[0]

    def _playout(self, b: Board) -> float:
        b = b.copy()
        while True:
            r = b.result()
            if r is not None:
                return r
            if self.rollout == "heuristic":
                b.play(heuristic_move(b, self.rng))
            else:
                b.play(self.rng.choice(b.legal_moves()))

    def _backprop(self, node: Node | None, result0: float) -> None:
        while node is not None:
            node.visits += 1
            if node.move is not None:
                score = result0 if node.mover == 0 else 1.0 - result0
                node.wins += score
            if self.solver:
                self._update_proof(node)
            node = node.parent

    def _update_proof(self, node: Node) -> None:
        """Proof propagation: ``node.proven`` is from the view of its mover."""
        if node.proven is not None or node.untried or not node.children:
            return
        vals = [ch.proven for ch in node.children]
        # children's movers are the opponent of node.mover
        if any(v is not None and v > 0 for v in vals):
            # the opponent has a forced win -> node.mover loses
            node.proven = -1.0
        elif all(v is not None for v in vals):
            node.proven = 1.0 if all(v < 0 for v in vals) else 0.0

    # -- results -------------------------------------------------------
    def best_move(self, root: Node | None = None) -> int:
        root = root or self.root
        if root is None or not root.children:
            raise ValueError("no search performed")
        # a child proven winning for its mover (the player to move at the root)
        wins = [ch for ch in root.children if ch.proven is not None and ch.proven > 0]
        if wins:
            return wins[0].move
        pool = [ch for ch in root.children if not (ch.proven is not None and ch.proven < 0)]
        pool = pool or root.children
        return max(pool, key=lambda ch: (ch.visits, ch.wins)).move

    def move_stats(self) -> dict[int, tuple[int, float]]:
        """Map move -> (visits, mean win-rate for the player to move at the root)."""
        if self.root is None:
            return {}
        return {ch.move: (ch.visits, ch.wins / max(ch.visits, 1)) for ch in self.root.children}

    def policy(self) -> list[float]:
        """Visit-count distribution over columns."""
        total = sum(ch.visits for ch in self.root.children) or 1
        p = [0.0] * COLS
        for ch in self.root.children:
            p[ch.move] = ch.visits / total
        return p
