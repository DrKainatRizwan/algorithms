"""Monte-Carlo CFR: external sampling and outcome sampling.

Both samplers walk the compiled tree with plain Python recursion and keep
regrets in flat lists, so a single iteration touches only a small part of the
game.  References: Lanctot et al., "Monte Carlo Sampling for Regret
Minimization in Extensive Games", NIPS 2009.
"""
from __future__ import annotations

import random

import numpy as np

from .cfr import normalise_average
from .games import CHANCE, TERMINAL
from .tree import GameTree


class MCCFRSolver:
    """Sampling-based CFR with ``external`` or ``outcome`` sampling.

    Parameters
    ----------
    tree:
        Compiled game.
    mode:
        ``"external"`` samples chance and opponent actions and enumerates the
        traverser's actions; ``"outcome"`` samples a single trajectory using an
        epsilon-greedy behaviour policy for the traverser.
    seed:
        Seed for the private random generator (runs are fully reproducible).
    epsilon:
        Exploration rate for outcome sampling.
    """

    def __init__(self, tree: GameTree, mode: str = "external", seed: int = 0,
                 epsilon: float = 0.6) -> None:
        if mode not in ("external", "outcome"):
            raise ValueError("mode must be 'external' or 'outcome'")
        self.tree = tree
        self.mode = mode
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.regret = [0.0] * tree.n_slots
        self.strategy_sum = [0.0] * tree.n_slots
        self.t = 0
        self._off = tree.info_offset.tolist()
        self._nact = tree.info_nact.tolist()
        self._kind = tree.node_kind.tolist()
        self._info = tree.node_infoset.tolist()
        self._util = tree.node_util.tolist()
        self._chance_p: dict[int, list[float]] = {}
        # chance probabilities per node, aligned with children order
        by_parent: dict[int, list[float]] = {}
        for e in range(len(tree.edge_parent)):
            if tree.edge_slot[e] < 0:
                by_parent.setdefault(int(tree.edge_parent[e]), []).append(
                    float(tree.edge_chance[e]))
        self._chance_p = by_parent

    # ------------------------------------------------------------------
    def _strategy(self, info: int) -> list[float]:
        o, n = self._off[info], self._nact[info]
        r = [x if x > 0.0 else 0.0 for x in self.regret[o:o + n]]
        s = sum(r)
        if s > 0.0:
            return [x / s for x in r]
        return [1.0 / n] * n

    def _sample(self, probs: list[float]) -> int:
        u = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if u < acc:
                return i
        return len(probs) - 1

    # ---------------------------- external ----------------------------
    def _external(self, node: int, trav: int) -> float:
        kind = self._kind[node]
        if kind == TERMINAL:
            return self._util[node] if trav == 0 else -self._util[node]
        kids = self.tree.children[node]
        if kind == CHANCE:
            k = self._sample(self._chance_p[node])
            return self._external(kids[k], trav)
        info = self._info[node]
        sigma = self._strategy(info)
        if kind == trav:
            vals = [self._external(c, trav) for c in kids]
            v = sum(p * x for p, x in zip(sigma, vals))
            o = self._off[info]
            for a, x in enumerate(vals):
                self.regret[o + a] += x - v
            return v
        o = self._off[info]
        w = float(self.t)  # linear averaging stabilises the sampled average
        for a, p in enumerate(sigma):
            self.strategy_sum[o + a] += w * p
        return self._external(kids[self._sample(sigma)], trav)

    # ---------------------------- outcome -----------------------------
    def _outcome_iteration(self, trav: int) -> None:
        """Sample one trajectory and apply importance-weighted updates."""
        path: list[tuple[int, int, float, float, float]] = []
        node = 0
        weight = 1.0       # 1 / prod of traverser sampling probabilities
        reach_own = 1.0    # prod of traverser's sigma probabilities
        q_own = 1.0        # prod of traverser's sampling probabilities
        eps = self.epsilon
        while self._kind[node] != TERMINAL:
            kind = self._kind[node]
            kids = self.tree.children[node]
            if kind == CHANCE:
                node = kids[self._sample(self._chance_p[node])]
                continue
            info = self._info[node]
            sigma = self._strategy(info)
            if kind == trav:
                n = len(sigma)
                samp = [eps / n + (1.0 - eps) * p for p in sigma]
                a = self._sample(samp)
                path.append((info, a, sigma[a], samp[a], reach_own / q_own))
                reach_own *= sigma[a]
                q_own *= samp[a]
                weight /= samp[a]
            else:
                a = self._sample(sigma)
            node = kids[a]
        u = self._util[node] if trav == 0 else -self._util[node]
        # suffix[j] = prod of the traverser's sigma probabilities after position j
        suffix = [1.0] * (len(path) + 1)
        for j in range(len(path) - 1, -1, -1):
            suffix[j] = suffix[j + 1] * path[j][2]
        w = u * weight
        for j, (info, a, sa, qa, avg_w) in enumerate(path):
            o, n = self._off[info], self._nact[info]
            tail_ha = suffix[j + 1]                # pi_i(h a -> z)
            tail_h = tail_ha * sa                  # pi_i(h -> z)
            for b in range(n):
                self.regret[o + b] += w * ((tail_ha if b == a else 0.0) - tail_h)
            sigma = self._strategy(info)
            for b in range(n):
                self.strategy_sum[o + b] += avg_w * sigma[b]

    # ------------------------------------------------------------------
    def iterate(self) -> None:
        """One sampled iteration (the traverser alternates every call)."""
        self.t += 1
        trav = self.t % 2
        if self.mode == "external":
            self._external(0, trav)
        else:
            self._outcome_iteration(trav)

    def run(self, iterations: int) -> "MCCFRSolver":
        for _ in range(iterations):
            self.iterate()
        return self

    @property
    def average_strategy(self) -> np.ndarray:
        return normalise_average(np.array(self.strategy_sum), self.tree)

    @property
    def current_strategy(self) -> np.ndarray:
        out = np.empty(self.tree.n_slots)
        for i in range(self.tree.n_infosets):
            o = self._off[i]
            out[o:o + self._nact[i]] = self._strategy(i)
        return out
