"""Double-spend and selfish-mining analysis: closed forms and Monte Carlo checks."""
from __future__ import annotations

import math
import random


def nakamoto_double_spend(q: float, z: int) -> float:
    """Success probability from Nakamoto's whitepaper (Poisson approximation)."""
    p = 1.0 - q
    if q >= p:
        return 1.0
    lam = z * q / p
    s = 0.0
    for k in range(z + 1):
        poisson = math.exp(-lam) * lam**k / math.factorial(k)
        s += poisson * (1.0 - (q / p) ** (z - k))
    return 1.0 - s


def rosenfeld_double_spend(q: float, z: int, overtake: bool = False) -> float:
    """Exact probability (negative-binomial attacker progress) from Rosenfeld 2014.

    With ``overtake=False`` the attacker wins by merely tying (Nakamoto's
    convention); with ``overtake=True`` the attacker must strictly out-grow the
    honest chain, which is what :func:`simulate_double_spend` models.
    """
    p = 1.0 - q
    if q >= p:
        return 1.0
    if z == 0:
        return 1.0
    s = 0.0
    for k in range(z + 1):
        catch = p ** (k - 1) * q ** (z + 1) if overtake else p**k * q**z
        s += math.comb(k + z - 1, k) * (p**z * q**k - catch)
    return 1.0 - s


def simulate_double_spend(q: float, z: int, trials: int = 20000, seed: int = 0,
                          give_up: int = 40) -> float:
    """Race simulation: attacker mines privately while the merchant waits for z blocks."""
    rng = random.Random(seed)
    wins = 0
    for _ in range(trials):
        honest = attacker = 0
        while honest < z:
            if rng.random() < q:
                attacker += 1
            else:
                honest += 1
        # Attacker needs a strictly longer chain than the honest one.
        deficit = honest - attacker
        while 0 <= deficit <= give_up:
            deficit += -1 if rng.random() < q else 1
        wins += deficit < 0
    return wins / trials


def confirmations_needed(q: float, risk: float, max_z: int = 500) -> int:
    """Smallest z whose exact double-spend probability is below ``risk``."""
    if q >= 0.5:
        raise ValueError("attacker with majority hashrate always succeeds")
    for z in range(1, max_z + 1):
        if rosenfeld_double_spend(q, z) < risk:
            return z
    raise ValueError("risk target unreachable within max_z")


def selfish_mining_revenue_theory(alpha: float, gamma: float) -> float:
    """Relative pool revenue under selfish mining (Eyal & Sirer 2014, eq. 8)."""
    a, g = alpha, gamma
    num = a * (1 - a) ** 2 * (4 * a + g * (1 - 2 * a)) - a**3
    den = 1 - a * (1 + (2 - a) * a)
    return num / den


def selfish_threshold(gamma: float) -> float:
    """Minimum pool share for which selfish mining beats honest mining."""
    return (1 - gamma) / (3 - 2 * gamma)


def simulate_selfish_mining(alpha: float, gamma: float, n_blocks: int = 200000,
                            seed: int = 0) -> float:
    """State-machine simulation of the Eyal-Sirer strategy; returns pool revenue share."""
    rng = random.Random(seed)
    pool = honest = 0
    lead = 0
    tie = False                      # state 0': two public branches of equal length
    for _ in range(n_blocks):
        pool_finds = rng.random() < alpha
        if tie:
            if pool_finds:
                pool += 2
            elif rng.random() < gamma:
                pool += 1
                honest += 1
            else:
                honest += 2
            tie, lead = False, 0
        elif pool_finds:
            lead += 1
        elif lead == 0:
            honest += 1
        elif lead == 1:
            tie, lead = True, 0
        elif lead == 2:
            pool += 2
            lead = 0
        else:
            pool += 1
            lead -= 1
    pool += lead                     # remaining private lead is eventually published
    return pool / (pool + honest)


def orphan_rate_theory(latency: float, spacing: float) -> float:
    """Probability another block is found during one propagation delay: 1-exp(-d/T)."""
    return 1.0 - math.exp(-latency / spacing)
