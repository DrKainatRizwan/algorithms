"""Discrete-event simulation of miners gossiping blocks over a latent network.

Block discovery times are exponential with rate ``hashrate / work(target)``
(the memoryless model of proof of work), but every block that is produced is
a genuine one: a valid nonce is ground against the real target and every node
fully validates what it receives, so forks, orphans and reorgs emerge from
the actual consensus rules.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field

from .block import Block, Transaction
from .chain import ConsensusParams, Blockchain, Mempool, REORG
from .pow import bits_to_target, mine_block, work_from_target


@dataclass
class SimConfig:
    hashrates: list[float]                     # relative shares, normalised internally
    params: ConsensusParams = field(default_factory=ConsensusParams)
    latency_mean: float = 1.0
    latency_jitter: float = 0.5                # uniform +-fraction of the mean
    tx_rate: float = 0.0                       # wallet transfers per second
    n_accounts: int = 8
    seed: int = 0


@dataclass
class SimReport:
    duration: float
    blocks_mined: list[int]
    main_chain_blocks: list[int]
    stale_blocks: int
    reorgs: int
    max_reorg_depth: int
    height: int
    converged: bool
    mean_interval: float
    txs_confirmed: int

    @property
    def stale_rate(self) -> float:
        total = sum(self.blocks_mined)
        return self.stale_blocks / total if total else 0.0


class Miner:
    def __init__(self, idx: int, hashrate: float, params: ConsensusParams) -> None:
        self.idx = idx
        self.name = f"miner{idx}"
        self.hashrate = hashrate
        self.chain = Blockchain(params)
        self.mempool = Mempool()
        self.epoch = 0
        self.mined = 0
        self.reorgs = 0
        self.max_reorg = 0


class NetworkSimulation:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.accounts = [f"acct{i}" for i in range(cfg.n_accounts)]
        alloc = tuple((a, 10_000) for a in self.accounts)
        self.params = ConsensusParams(
            **{**cfg.params.__dict__, "genesis_alloc": alloc}) if not cfg.params.genesis_alloc \
            else cfg.params
        total = sum(cfg.hashrates)
        # Scale so the network's aggregate rate yields one block per target_spacing at genesis.
        w0 = work_from_target(bits_to_target(self.params.genesis_bits))
        base = w0 / self.params.target_spacing
        self.miners = [Miner(i, base * h / total, self.params) for i, h in enumerate(cfg.hashrates)]
        self.now = 0.0
        self._seq = 0
        self._heap: list[tuple[float, int, str, tuple]] = []
        self._tx_nonce = {a: 0 for a in self.accounts}
        self.txs_issued = 0

    # ----------------------------------------------------------------- events
    def _push(self, t: float, kind: str, payload: tuple) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, kind, payload))

    def _schedule_mining(self, m: Miner) -> None:
        m.epoch += 1
        target = bits_to_target(m.chain.expected_bits(m.chain.tip))
        rate = m.hashrate / work_from_target(target)
        if rate > 0:
            self._push(self.now + self.rng.expovariate(rate), "mine", (m.idx, m.epoch))

    def _latency(self) -> float:
        c = self.cfg
        return max(0.0, c.latency_mean * (1 + c.latency_jitter * self.rng.uniform(-1, 1)))

    def _broadcast(self, src: int, block: Block) -> None:
        for m in self.miners:
            if m.idx != src:
                self._push(self.now + self._latency(), "deliver", (m.idx, block))

    def _receive(self, m: Miner, block: Block) -> None:
        old_tip = m.chain.tip
        res = m.chain.add_block(block)
        m.mempool.update_after(m.chain, res)
        if res.status == REORG:
            m.reorgs += 1
            m.max_reorg = max(m.max_reorg, res.reorg_depth)
        if m.chain.tip is not old_tip:
            self._schedule_mining(m)

    def _new_tx(self) -> None:
        a, b = self.rng.sample(self.accounts, 2)
        tx = Transaction(a, b, self.rng.randint(1, 5), self.rng.randint(0, 3), self._tx_nonce[a])
        self._tx_nonce[a] += 1
        self.txs_issued += 1
        for m in self.miners:
            m.mempool.add(tx)
        self._push(self.now + self.rng.expovariate(self.cfg.tx_rate), "tx", ())

    # ----------------------------------------------------------------- run
    def run(self, duration: float) -> SimReport:
        for m in self.miners:
            self._schedule_mining(m)
        if self.cfg.tx_rate > 0:
            self._push(self.rng.expovariate(self.cfg.tx_rate), "tx", ())
        draining = False
        while self._heap:
            t, _, kind, payload = heapq.heappop(self._heap)
            if t > duration and not draining:
                draining = True
            if draining and kind != "deliver":
                continue          # stop producing; let in-flight blocks land
            self.now = t
            if kind == "mine":
                idx, epoch = payload
                m = self.miners[idx]
                if epoch != m.epoch:
                    continue
                self._mine_one(m)
            elif kind == "deliver":
                self._receive(self.miners[payload[0]], payload[1])
            elif kind == "tx":
                self._new_tx()
        return self._report(duration)

    def _mine_one(self, m: Miner) -> None:
        chain = m.chain
        txs = m.mempool.select(chain)
        tmpl = chain.block_template(m.name, txs, int(self.now))
        solved = mine_block(tmpl, self.rng.randrange(0, 1 << 31))
        if solved is None:
            self._schedule_mining(m)
            return
        block = solved[0]
        m.mined += 1
        self._receive(m, block)
        self._broadcast(m.idx, block)
        self._schedule_mining(m)

    def _report(self, duration: float) -> SimReport:
        ref = self.miners[0].chain
        main = ref.main_chain()
        counts = [0] * len(self.miners)
        for node in main[1:]:
            counts[int(node.block.txs[0].recipient.removeprefix("miner"))] += 1
        stamps = [n.timestamp for n in main]
        interval = (stamps[-1] - stamps[0]) / max(1, len(main) - 1)
        tips = {m.chain.tip.hash for m in self.miners}
        mined = [m.mined for m in self.miners]
        confirmed = sum(len(n.block.txs) - 1 for n in main)
        return SimReport(
            duration=duration, blocks_mined=mined, main_chain_blocks=counts,
            stale_blocks=sum(mined) - (len(main) - 1),
            reorgs=sum(m.reorgs for m in self.miners),
            max_reorg_depth=max(m.max_reorg for m in self.miners),
            height=ref.height, converged=len(tips) == 1, mean_interval=interval,
            txs_confirmed=confirmed)
