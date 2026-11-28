"""Network construction and experiments: hop scaling, lookup accuracy, churn."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .ids import random_unique_ids
from .network import Network
from .node import DHTNode


def build_network(n: int, bits: int = 32, k: int = 8, alpha: int = 3,
                  seed: int = 0, loss: float = 0.0) -> tuple[Network, list[DHTNode]]:
    """Create ``n`` nodes that join one by one through a random existing node."""
    rng = random.Random(seed)
    net = Network(seed=seed + 1, loss=loss)
    ids = random_unique_ids(n, bits, rng)
    nodes: list[DHTNode] = []
    for i, nid in enumerate(ids):
        node = DHTNode(nid, net, bits, k, alpha)
        if nodes:
            node.bootstrap(rng.choice(nodes).id)
        nodes.append(node)
    return net, nodes


@dataclass
class LookupStats:
    mean_rounds: float
    mean_rpcs: float
    accuracy: float  # fraction of lookups whose result contains the true closest node


def lookup_stats(net: Network, nodes: list[DHTNode], trials: int, seed: int = 0) -> LookupStats:
    """Random lookups from random live nodes to random targets."""
    rng = random.Random(seed)
    bits = nodes[0].bits
    live = [n for n in nodes if n.id in net.online]
    rounds = rpcs = hits = 0
    for _ in range(trials):
        src = rng.choice(live)
        target = rng.getrandbits(bits)
        res = src.lookup(target)
        rounds += res.rounds
        rpcs += res.rpcs
        truth = net.true_closest(target, 1)[0]
        if truth in res.closest or truth == src.id:
            hits += 1
    return LookupStats(rounds / trials, rpcs / trials, hits / trials)


def fit_log2_slope(sizes: list[int], hops: list[float]) -> tuple[float, float]:
    """Least-squares fit ``hops = a*log2(N) + b``; returns ``(a, b)``."""
    xs = [math.log2(s) for s in sizes]
    mx, my = sum(xs) / len(xs), sum(hops) / len(hops)
    sxx = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, hops)) / sxx
    return a, my - a * mx


def churn_experiment(n: int = 200, keys: int = 100, kill_fraction: float = 0.5,
                     republish: bool = False, seed: int = 0, k: int = 8) -> float:
    """Store keys, kill a fraction of nodes, optionally republish; return retrieval rate."""
    net, nodes = build_network(n, k=k, seed=seed)
    rng = random.Random(seed + 7)
    for i in range(keys):
        rng.choice(nodes).put(f"key-{i}", i)
    victims = rng.sample(nodes, int(n * kill_fraction))
    for v in victims:
        net.kill(v.id)
    live = [x for x in nodes if x.id in net.online]
    if republish:
        # Holders republish only what they have in storage (typical hourly job).
        for node in live:
            node.republish()
    ok = 0
    for i in range(keys):
        res = rng.choice(live).get(f"key-{i}")
        if res.found and res.value == i:
            ok += 1
    return ok / keys
