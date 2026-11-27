"""Exact strategy evaluation: expected values, best responses, exploitability."""
from __future__ import annotations

import numpy as np

from .cfr import edge_probabilities, node_values, reach_probabilities
from .tree import GameTree


def combine(tree: GameTree, sigma0: np.ndarray, sigma1: np.ndarray) -> np.ndarray:
    """Profile that plays ``sigma0`` at player-0 infosets and ``sigma1`` at player-1's."""
    return np.where(tree.info_player[tree.slot_info] == 0, sigma0, sigma1)


def expected_value(tree: GameTree, sigma: np.ndarray) -> float:
    """Expected utility of player 0 when everybody follows ``sigma``."""
    return float(node_values(tree, edge_probabilities(tree, sigma))[0])


def best_response(tree: GameTree, sigma: np.ndarray, player: int) -> tuple[float, np.ndarray]:
    """Exact best response of ``player`` to the other players' part of ``sigma``.

    Returns ``(value, br)`` where ``value`` is the best-response expected
    utility for ``player`` and ``br`` is a flat pure strategy (one-hot per
    infoset; infosets unreachable by the opponent default to the first action).

    Perfect recall makes infoset *levels* (number of the player's own earlier
    decisions) a valid processing order: deeper levels are resolved first.
    """
    probs_fixed = edge_probabilities(tree, sigma)
    own_edge = tree.edge_player == player
    opp_probs = np.where(own_edge, 1.0, probs_fixed)
    reach = np.ones(tree.n_nodes)
    for e in tree.layers:
        reach[tree.edge_child[e]] = reach[tree.edge_parent[e]] * opp_probs[e]
    sgn = 1.0 if player == 0 else -1.0
    util = np.where(tree.node_kind == 2, sgn * tree.node_util, 0.0)
    br = np.zeros(tree.n_slots)
    own_idx = np.nonzero(own_edge)[0]
    own_slot = tree.edge_slot[own_idx]
    own_par, own_ch = tree.edge_parent[own_idx], tree.edge_child[own_idx]

    def backward(br_vec: np.ndarray) -> np.ndarray:
        probs = probs_fixed.copy()
        probs[own_idx] = br_vec[own_slot]
        v = util.copy()
        for e in reversed(tree.layers):
            np.add.at(v, tree.edge_parent[e], probs[e] * v[tree.edge_child[e]])
        return v

    mine = np.nonzero(tree.info_player == player)[0]
    max_level = int(tree.info_level[mine].max()) if len(mine) else -1
    for level in range(max_level, -1, -1):
        v = backward(br)
        cfv = np.bincount(own_slot, weights=reach[own_par] * v[own_ch],
                          minlength=tree.n_slots)
        for i in mine[tree.info_level[mine] == level]:
            sl = tree.infoset_slice(i)
            br[sl] = 0.0
            br[sl.start + int(np.argmax(cfv[sl]))] = 1.0
    # unreachable / untouched infosets: make strategy valid (first action)
    for i in mine:
        sl = tree.infoset_slice(i)
        if br[sl].sum() == 0:
            br[sl.start] = 1.0
    value = float(backward(br)[0])
    return value, br


def nash_conv(tree: GameTree, sigma: np.ndarray) -> float:
    """Sum of both players' best-response gains over the game value (``br0 + br1``)."""
    return best_response(tree, sigma, 0)[0] + best_response(tree, sigma, 1)[0]


def exploitability(tree: GameTree, sigma: np.ndarray) -> float:
    """Average best-response gain, ``(br0 + br1) / 2``; zero exactly at equilibrium."""
    return 0.5 * nash_conv(tree, sigma)


def infoset_reach(tree: GameTree, sigma: np.ndarray) -> np.ndarray:
    """Total reach probability (all players incl. chance) of every infoset."""
    reach = reach_probabilities(tree, edge_probabilities(tree, sigma))
    total = reach.prod(axis=0)
    out = np.zeros(tree.n_infosets)
    dec = tree.node_infoset >= 0
    np.add.at(out, tree.node_infoset[dec], total[dec])
    return out
