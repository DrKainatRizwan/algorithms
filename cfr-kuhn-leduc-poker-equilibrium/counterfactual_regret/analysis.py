"""Analytic reference points and diagnostic helpers for Kuhn poker."""
from __future__ import annotations

import numpy as np

from .tree import GameTree

KUHN_VALUE = -1.0 / 18.0   # value of Kuhn poker for player 0 (the first mover)


def _set(tree: GameTree, sigma: np.ndarray, key: str, player: int, p_bet: float) -> None:
    i = tree.key_index[f"{player}#{key}"]
    o = int(tree.info_offset[i])
    sigma[o], sigma[o + 1] = 1.0 - p_bet, p_bet


def kuhn_equilibrium(tree: GameTree, alpha: float = 0.0) -> np.ndarray:
    """Member ``alpha`` in [0, 1/3] of the known one-parameter Nash family.

    Player 0 bets the jack with probability ``alpha`` and the king with
    ``3 * alpha``, always checks the queen first and calls a bet after
    check-bet with the queen with probability ``alpha + 1/3``.  Player 1 follows
    the unique equilibrium best reply.
    """
    if not 0.0 <= alpha <= 1.0 / 3.0 + 1e-12:
        raise ValueError("alpha must lie in [0, 1/3]")
    s = tree.uniform_strategy()
    _set(tree, s, "J:", 0, alpha)
    _set(tree, s, "Q:", 0, 0.0)
    _set(tree, s, "K:", 0, 3 * alpha)
    _set(tree, s, "J:pb", 0, 0.0)
    _set(tree, s, "Q:pb", 0, alpha + 1.0 / 3.0)
    _set(tree, s, "K:pb", 0, 1.0)
    _set(tree, s, "J:p", 1, 1.0 / 3.0)
    _set(tree, s, "Q:p", 1, 0.0)
    _set(tree, s, "K:p", 1, 1.0)
    _set(tree, s, "J:b", 1, 0.0)
    _set(tree, s, "Q:b", 1, 1.0 / 3.0)
    _set(tree, s, "K:b", 1, 1.0)
    return s


def kuhn_alpha(tree: GameTree, sigma: np.ndarray) -> float:
    """Probability with which player 0 bets the jack at the root."""
    i = tree.key_index["0#J:"]
    return float(sigma[tree.info_offset[i] + 1])


def regret_bound(num_actions: int, max_utility: float, num_infosets: int,
                 iterations: int) -> float:
    """Zinkevich et al. bound on average overall regret: ``Delta |I| sqrt(A) / sqrt(T)``."""
    return max_utility * num_infosets * np.sqrt(num_actions) / np.sqrt(iterations)


def format_strategy(tree: GameTree, sigma: np.ndarray, player: int,
                    limit: int | None = None) -> str:
    """Pretty-print a player's strategy as one line per infoset."""
    lines = []
    for i, key in enumerate(tree.info_keys):
        if tree.info_player[i] != player:
            continue
        sl = tree.infoset_slice(i)
        probs = " ".join(f"{a}={p:.3f}" for a, p in zip(tree.info_actions[i], sigma[sl]))
        lines.append(f"  {key:<14s} {probs}")
    return "\n".join(lines[:limit] if limit else lines)
