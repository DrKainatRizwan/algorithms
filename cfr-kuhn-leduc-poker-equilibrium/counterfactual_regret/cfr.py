"""Full-tree counterfactual regret minimisation (vanilla CFR, CFR+, LCFR, DCFR).

All variants share one vectorised pass over the compiled game tree.  A pass
propagates reach probabilities top-down, values bottom-up, and then converts the
per-edge counterfactual regrets to per-slot regrets with ``np.bincount``.

Variants are expressed through :class:`CFRConfig`:

* **vanilla** - simultaneous updates, unrestricted regrets, uniform averaging
  (Zinkevich et al., 2007).
* **plus** - alternating updates, regrets floored at zero, linear averaging
  (Tammelin et al., 2015).
* **linear** - simultaneous, regrets and averages weighted linearly in ``t``
  (Brown & Sandholm, 2019).
* **dcfr** - discounted CFR with exponents ``(alpha, beta, gamma)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import GameTree


@dataclass(frozen=True)
class CFRConfig:
    """Hyper-parameters selecting a CFR variant."""

    alternating: bool = False
    floor: bool = False
    alpha: float | None = None   # discount exponent for positive regrets
    beta: float | None = None    # discount exponent for negative regrets
    gamma: float | None = None   # discount exponent for the strategy sum
    avg_power: float = 0.0       # strategy increments are weighted by t**avg_power

    @staticmethod
    def preset(name: str) -> "CFRConfig":
        presets = {
            "vanilla": CFRConfig(),
            "plus": CFRConfig(alternating=True, floor=True, avg_power=1.0),
            "linear": CFRConfig(alpha=1.0, beta=1.0, gamma=1.0),
            "dcfr": CFRConfig(alternating=False, alpha=1.5, beta=0.0, gamma=2.0),
        }
        if name not in presets:
            raise ValueError(f"unknown CFR variant {name!r}; choose from {sorted(presets)}")
        return presets[name]


def regret_matching(regret: np.ndarray, tree: GameTree) -> np.ndarray:
    """Map flat regret sums to a flat behavioural strategy."""
    pos = np.maximum(regret, 0.0)
    denom = np.bincount(tree.slot_info, weights=pos, minlength=tree.n_infosets)
    d = denom[tree.slot_info]
    uniform = 1.0 / tree.info_nact[tree.slot_info]
    return np.where(d > 0, pos / np.where(d > 0, d, 1.0), uniform)


def normalise_average(strategy_sum: np.ndarray, tree: GameTree) -> np.ndarray:
    """Turn accumulated reach-weighted strategies into the average strategy."""
    tot = np.bincount(tree.slot_info, weights=strategy_sum, minlength=tree.n_infosets)
    d = tot[tree.slot_info]
    uniform = 1.0 / tree.info_nact[tree.slot_info]
    return np.where(d > 0, strategy_sum / np.where(d > 0, d, 1.0), uniform)


def edge_probabilities(tree: GameTree, sigma: np.ndarray) -> np.ndarray:
    """Probability of every edge under the profile ``sigma``."""
    slot = np.maximum(tree.edge_slot, 0)
    return np.where(tree.edge_slot >= 0, sigma[slot], tree.edge_chance)


def reach_probabilities(tree: GameTree, probs: np.ndarray) -> np.ndarray:
    """Return an ``(3, n_nodes)`` array of reach probabilities.

    Rows are player 0, player 1 and chance; row ``p`` is the product of the
    probabilities of the edges taken by ``p`` on the path from the root.
    """
    reach = np.ones((3, tree.n_nodes))
    ep = tree.edge_player
    fac = np.ones((3, len(ep)))
    fac[0] = np.where(ep == 0, probs, 1.0)
    fac[1] = np.where(ep == 1, probs, 1.0)
    fac[2] = np.where(ep == -1, probs, 1.0)
    for e in tree.layers:
        par, ch = tree.edge_parent[e], tree.edge_child[e]
        reach[:, ch] = reach[:, par] * fac[:, e]
    return reach


def node_values(tree: GameTree, probs: np.ndarray) -> np.ndarray:
    """Expected player-0 utility of every node under the edge probabilities."""
    v = np.where(tree.node_kind == 2, tree.node_util, 0.0)
    for e in reversed(tree.layers):
        np.add.at(v, tree.edge_parent[e], probs[e] * v[tree.edge_child[e]])
    return v


class CFRSolver:
    """Vectorised full-traversal CFR over a compiled :class:`GameTree`."""

    def __init__(self, tree: GameTree, variant: str | CFRConfig = "vanilla") -> None:
        self.tree = tree
        self.config = CFRConfig.preset(variant) if isinstance(variant, str) else variant
        self.regret = np.zeros(tree.n_slots)
        self.strategy_sum = np.zeros(tree.n_slots)
        self.sigma = tree.uniform_strategy()
        self.t = 0
        dec = tree.edge_slot >= 0
        self._dec_edges = np.nonzero(dec)[0]
        self._dec_slot = tree.edge_slot[self._dec_edges]
        self._dec_par = tree.edge_parent[self._dec_edges]
        self._dec_child = tree.edge_child[self._dec_edges]
        self._dec_player = tree.edge_player[self._dec_edges]
        self._slot_player = tree.info_player[tree.slot_info]

    # ------------------------------------------------------------------
    def _pass(self, update_player: int | None) -> None:
        """One traversal; update regrets/averages of ``update_player`` (or both)."""
        tree, cfg, t = self.tree, self.config, self.t
        probs = edge_probabilities(tree, self.sigma)
        reach = reach_probabilities(tree, probs)
        v = node_values(tree, probs)
        sign = np.where(self._dec_player == 0, 1.0, -1.0)
        opp = np.where(self._dec_player == 0, reach[1, self._dec_par],
                       reach[0, self._dec_par]) * reach[2, self._dec_par]
        own = np.where(self._dec_player == 0, reach[0, self._dec_par],
                       reach[1, self._dec_par])
        inst = opp * sign * (v[self._dec_child] - v[self._dec_par])
        inc = own * probs[self._dec_edges] * float(t) ** cfg.avg_power
        if update_player is not None:
            keep = self._dec_player == update_player
            inst = np.where(keep, inst, 0.0)
            inc = np.where(keep, inc, 0.0)
        d_reg = np.bincount(self._dec_slot, weights=inst, minlength=tree.n_slots)
        d_str = np.bincount(self._dec_slot, weights=inc, minlength=tree.n_slots)
        active = (np.ones(tree.n_slots, bool) if update_player is None
                  else self._slot_player == update_player)

        reg = self.regret
        if cfg.alpha is not None and cfg.beta is not None:
            tt = float(t)
            dpos = tt ** cfg.alpha / (tt ** cfg.alpha + 1.0)
            dneg = tt ** cfg.beta / (tt ** cfg.beta + 1.0)
            disc = np.where(reg > 0, dpos, dneg)
            reg = np.where(active, reg * disc, reg)
        reg = reg + d_reg
        if cfg.floor:
            reg = np.where(active, np.maximum(reg, 0.0), reg)
        self.regret = reg
        ss = self.strategy_sum
        if cfg.gamma is not None:
            ss = np.where(active, ss * (t / (t + 1.0)) ** cfg.gamma, ss)
        self.strategy_sum = ss + d_str
        new_sigma = regret_matching(self.regret, tree)
        self.sigma = np.where(active, new_sigma, self.sigma)

    def iterate(self) -> None:
        """Run one CFR iteration."""
        self.t += 1
        if self.config.alternating:
            for p in (0, 1):
                self._pass(p)
        else:
            self._pass(None)

    def run(self, iterations: int) -> "CFRSolver":
        for _ in range(iterations):
            self.iterate()
        return self

    @property
    def average_strategy(self) -> np.ndarray:
        """The time-averaged strategy, which converges to a Nash equilibrium."""
        return normalise_average(self.strategy_sum, self.tree)

    @property
    def current_strategy(self) -> np.ndarray:
        """The last regret-matching strategy (need not converge)."""
        return self.sigma.copy()
