"""Compile a :class:`~counterfactual_regret.games.Game` into flat arrays.

The compiled :class:`GameTree` stores nodes in pre-order together with edge
arrays grouped into *layers* (all edges whose parent has the same depth).  The
vectorised solvers walk the layers top-down to propagate reach probabilities
and bottom-up to propagate values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .games import CHANCE, TERMINAL, Game


@dataclass
class GameTree:
    """Flat, array-based representation of an extensive-form game."""

    name: str
    node_kind: np.ndarray            # 0/1 decision, CHANCE, TERMINAL
    node_depth: np.ndarray
    node_util: np.ndarray            # player-0 utility at terminals
    node_infoset: np.ndarray         # infoset id or -1
    children: list[list[int]]
    # edges
    edge_parent: np.ndarray
    edge_child: np.ndarray
    edge_slot: np.ndarray            # strategy slot for decision edges, else -1
    edge_chance: np.ndarray          # chance probability (1.0 for decisions)
    edge_player: np.ndarray          # acting player or CHANCE
    layers: list[np.ndarray]         # edge index arrays, by parent depth
    # information sets
    info_player: np.ndarray
    info_offset: np.ndarray
    info_nact: np.ndarray
    info_level: np.ndarray           # number of own decisions before the infoset
    info_keys: list[str]
    info_actions: list[list[str]]
    slot_info: np.ndarray            # infoset id of each strategy slot
    n_slots: int = 0
    key_index: dict[str, int] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.node_kind)

    @property
    def n_infosets(self) -> int:
        return len(self.info_keys)

    def uniform_strategy(self) -> np.ndarray:
        """Flat strategy vector assigning equal probability to all actions."""
        return 1.0 / self.info_nact[self.slot_info]

    def infoset_slice(self, i: int) -> slice:
        o = int(self.info_offset[i])
        return slice(o, o + int(self.info_nact[i]))

    def strategy_dict(self, sigma: np.ndarray) -> dict[str, dict[str, float]]:
        """Human-readable ``{infoset key: {action: probability}}`` mapping."""
        out: dict[str, dict[str, float]] = {}
        for i, key in enumerate(self.info_keys):
            sl = self.infoset_slice(i)
            out[key] = dict(zip(self.info_actions[i], map(float, sigma[sl])))
        return out


def build_tree(game: Game) -> GameTree:
    """Enumerate ``game`` depth-first and return its compiled tree."""
    kinds: list[int] = []
    depths: list[int] = []
    utils: list[float] = []
    node_info: list[int] = []
    children: list[list[int]] = []
    key_index: dict[str, int] = {}
    info_player: list[int] = []
    info_actions: list[list[str]] = []
    info_keys: list[str] = []
    info_level: list[int] = []
    pending_edges: list[tuple[int, int, int, float, int]] = []

    def visit(state, depth: int, own: list[int]) -> int:
        idx = len(kinds)
        kind = game.node_kind(state)
        kinds.append(kind)
        depths.append(depth)
        utils.append(0.0)
        node_info.append(-1)
        children.append([])
        if kind == TERMINAL:
            utils[idx] = float(game.utility(state))
            return idx
        acts = list(game.actions(state))
        if kind == CHANCE:
            probs = list(game.chance_probs(state))
            for a, pr in zip(acts, probs):
                child = visit(game.apply(state, a), depth + 1, own)
                children[idx].append(child)
                pending_edges.append((idx, child, -1, pr, CHANCE))
            return idx
        key = f"{kind}#{game.infoset_key(state)}"
        if key not in key_index:
            key_index[key] = len(info_keys)
            info_keys.append(game.infoset_key(state))
            info_player.append(kind)
            info_actions.append(acts)
            info_level.append(own[kind])
        iid = key_index[key]
        if info_actions[iid] != acts:
            raise ValueError(f"inconsistent actions in infoset {key}")
        if info_level[iid] != own[kind]:
            raise ValueError(f"imperfect recall detected at {key}")
        node_info[idx] = iid
        own[kind] += 1
        for k, a in enumerate(acts):
            child = visit(game.apply(state, a), depth + 1, own)
            children[idx].append(child)
            pending_edges.append((idx, child, k, 1.0, kind))
        own[kind] -= 1
        return idx

    visit(game.initial(), 0, [0, 0])

    n_info = len(info_keys)
    nact = np.array([len(a) for a in info_actions], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(nact)[:-1]]).astype(np.int64)
    n_slots = int(nact.sum())
    slot_info = np.repeat(np.arange(n_info), nact)

    ep = np.array([e[0] for e in pending_edges], dtype=np.int64)
    ec = np.array([e[1] for e in pending_edges], dtype=np.int64)
    eslot = np.full(len(pending_edges), -1, dtype=np.int64)
    echance = np.array([e[3] for e in pending_edges], dtype=float)
    eplayer = np.array([e[4] for e in pending_edges], dtype=np.int64)
    nd = np.array(node_info, dtype=np.int64)
    for j, (par, _, k, _, kind) in enumerate(pending_edges):
        if kind != CHANCE:
            eslot[j] = offset[nd[par]] + k
    depth_arr = np.array(depths, dtype=np.int64)
    pdepth = depth_arr[ep]
    layers = [np.nonzero(pdepth == d)[0] for d in range(int(pdepth.max()) + 1)]

    return GameTree(
        name=game.name,
        node_kind=np.array(kinds, dtype=np.int64),
        node_depth=depth_arr,
        node_util=np.array(utils),
        node_infoset=nd,
        children=children,
        edge_parent=ep, edge_child=ec, edge_slot=eslot,
        edge_chance=echance, edge_player=eplayer, layers=layers,
        info_player=np.array(info_player, dtype=np.int64),
        info_offset=offset, info_nact=nact,
        info_level=np.array(info_level, dtype=np.int64),
        info_keys=info_keys, info_actions=info_actions,
        slot_info=slot_info, n_slots=n_slots, key_index=key_index,
    )
