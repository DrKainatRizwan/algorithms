"""Grid road-network simulator with queue-based signalised intersections.

Every intersection has four single-lane approaches and a two-phase signal
(phase 0 serves NS, phase 1 serves EW).  Switching phases costs ``yellow``
seconds of lost time and is only allowed after ``min_green`` seconds.
Vehicles discharge at the saturation flow rate, move to the downstream
intersection (or leave the network) and respect finite storage capacity, so
queues can spill back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demand import Demand, uniform_demand

# (d_row, d_col) of travel for directions S, N, E, W
DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
PERPENDICULAR = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}
PHASE_DIRS = {0: (0, 1), 1: (2, 3)}


@dataclass(frozen=True)
class SignalConfig:
    """Physical and control parameters shared by all intersections."""

    saturation: float = 0.5   # vehicles / second / green approach
    yellow: int = 2           # lost seconds per phase change
    min_green: int = 6        # seconds before a switch request is honoured
    interval: int = 5         # seconds between control decisions
    capacity: int = 30        # storage per approach
    turn_prob: float = 0.2    # probability of turning at an intersection


class TrafficGrid:
    """Rows x cols grid of signalised intersections."""

    def __init__(self, rows: int = 1, cols: int = 1, demand: Demand | None = None,
                 config: SignalConfig | None = None, seed: int = 0) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("grid must have at least one intersection")
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        self.demand = demand or uniform_demand()
        self.cfg = config or SignalConfig()
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.queue = np.zeros((self.n, 4), dtype=np.int64)
        self.phase = np.zeros(self.n, dtype=np.int64)
        self.pending = np.zeros(self.n, dtype=np.int64)
        self.yellow_left = np.zeros(self.n, dtype=np.int64)
        self.green_time = np.zeros(self.n, dtype=np.int64)
        self.credit = np.zeros((self.n, 4))
        self.arrived = 0       # vehicles accepted into the network
        self.exited = 0
        self.rejected = 0      # blocked at a full boundary approach
        self.queue_seconds = 0  # integral of total queue over time (vehicle-seconds)
        self.switches = 0

    def node_id(self, r: int, c: int) -> int:
        return r * self.cols + c

    def _neighbour(self, node: int, d: int) -> int | None:
        r, c = divmod(node, self.cols)
        nr, nc = r + DELTAS[d][0], c + DELTAS[d][1]
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return self.node_id(nr, nc)
        return None

    def _is_entry(self, node: int, d: int) -> bool:
        """True if approach ``d`` of ``node`` is fed from outside the grid."""
        r, c = divmod(node, self.cols)
        return {0: r == 0, 1: r == self.rows - 1, 2: c == 0, 3: c == self.cols - 1}[d]

    # --------------------------------------------------------------- dynamics
    @property
    def in_network(self) -> int:
        return int(self.queue.sum())

    def green_ok(self, node: int) -> bool:
        """Whether a switch request would currently be honoured."""
        return bool(self.yellow_left[node] == 0 and self.green_time[node] >= self.cfg.min_green)

    def _tick(self) -> None:
        cfg = self.cfg
        # arrivals at boundary approaches
        for node in range(self.n):
            arr = self.demand.sample(self.t, self.rng)
            for d in range(4):
                if arr[d] and self._is_entry(node, d):
                    room = cfg.capacity - self.queue[node, d]
                    take = int(min(arr[d], room))
                    self.queue[node, d] += take
                    self.arrived += take
                    self.rejected += int(arr[d]) - take
        # signals and discharge
        moves: list[tuple[int, int]] = []
        for node in range(self.n):
            if self.yellow_left[node] > 0:
                self.yellow_left[node] -= 1
                if self.yellow_left[node] == 0:
                    self.phase[node] = self.pending[node]
                    self.green_time[node] = 0
                continue
            self.green_time[node] += 1
            for d in PHASE_DIRS[int(self.phase[node])]:
                self.credit[node, d] += cfg.saturation
                while self.credit[node, d] >= 1.0 and self.queue[node, d] > 0:
                    dest = self._route(node, d)
                    if dest is not None and self.queue[dest[0], dest[1]] >= cfg.capacity:
                        break  # spillback: downstream storage is full
                    self.queue[node, d] -= 1
                    self.credit[node, d] -= 1.0
                    if dest is None:
                        self.exited += 1
                    else:
                        moves.append(dest)
                if self.queue[node, d] == 0:
                    self.credit[node, d] = min(self.credit[node, d], 1.0)
        for node, d in moves:  # arrive downstream next second (one-second link delay)
            self.queue[node, d] += 1
        self.queue_seconds += int(self.queue.sum())
        self.t += 1

    def _route(self, node: int, d: int) -> tuple[int, int] | None:
        """Pick the next (node, approach); ``None`` means the vehicle leaves the grid."""
        out = d
        if self.rng.random() < self.cfg.turn_prob:
            out = PERPENDICULAR[d][int(self.rng.integers(2))]
        nxt = self._neighbour(node, out)
        return None if nxt is None else (nxt, out)

    def step(self, actions: np.ndarray | list[int]) -> np.ndarray:
        """Apply one action per intersection (0 keep, 1 switch) for ``interval`` seconds.

        Returns the per-intersection reward: minus the mean local queue length over the interval.
        """
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError(f"expected {self.n} actions, got shape {actions.shape}")
        for node in range(self.n):
            if actions[node] == 1 and self.green_ok(node):
                self.pending[node] = 1 - self.phase[node]
                self.yellow_left[node] = self.cfg.yellow
                self.switches += 1
        acc = np.zeros(self.n)
        for _ in range(self.cfg.interval):
            self._tick()
            acc += self.queue.sum(axis=1)
        return -acc / self.cfg.interval

    # ------------------------------------------------------------ observation
    def group_queues(self, node: int) -> tuple[int, int]:
        q = self.queue[node]
        return int(q[0] + q[1]), int(q[2] + q[3])

    def observe(self, node: int) -> tuple[int, int, int, int]:
        """Raw local observation: (phase, can_switch, NS queue, EW queue)."""
        ns, ew = self.group_queues(node)
        return int(self.phase[node]), int(self.green_ok(node)), ns, ew

    def mean_queue(self) -> float:
        """Average number of queued vehicles in the whole network so far."""
        return self.queue_seconds / max(1, self.t)
