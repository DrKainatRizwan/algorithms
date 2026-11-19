"""Differential cryptanalysis: characteristic search and last-round key recovery."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from .sbox import ddt, invert
from .spn import (NIBBLES, SPN, active_nibbles, default_permutation, nibble,
                  nibbles, permute)


@dataclass(frozen=True)
class Characteristic:
    """A multi-round differential characteristic.

    ``steps[r] = (delta_in, delta_after_sbox, delta_after_perm, prob)``.
    ``output_diff`` is the difference entering the last S-box layer.
    """

    input_diff: int
    steps: tuple[tuple[int, int, int, float], ...]
    probability: float

    @property
    def output_diff(self) -> int:
        return self.steps[-1][2]

    @property
    def active_sboxes(self) -> int:
        return sum(len(active_nibbles(s[0])) for s in self.steps)


def _perm_diff(d: int, perm: tuple[int, ...]) -> int:
    return permute(d, perm)


def best_characteristic(sbox: tuple[int, ...], rounds: int,
                        perm: tuple[int, ...] | None = None,
                        max_active_input: int = 1) -> Characteristic | None:
    """Branch-and-bound search for the most probable ``rounds``-round characteristic.

    The probability is the product of single S-box transition probabilities
    (Markov-cipher assumption).  Input differences are restricted to at most
    ``max_active_input`` active nibbles; each round has at least one active
    S-box, which yields the pruning bound ``p * q_max**remaining``.
    """
    perm = perm or default_permutation()
    table = ddt(sbox)
    qmax = table[1:, 1:].max() / 16.0
    # transitions[a] = list of (b, prob) sorted by decreasing probability
    trans = {a: sorted(((b, table[a, b] / 16.0) for b in range(16) if table[a, b]),
                       key=lambda t: -t[1]) for a in range(1, 16)}
    best: dict = {"p": 0.0, "path": None}

    @lru_cache(maxsize=None)
    def pdiff(d: int) -> int:
        return _perm_diff(d, perm)

    def expand(d: int):
        """Yield (out_diff_after_sbox, prob) for all non-pruned S-box outputs."""
        act = [(i, nibble(d, i)) for i in range(NIBBLES) if nibble(d, i)]
        for combo in product(*(trans[a] for _, a in act)):
            out, p = 0, 1.0
            for (i, _), (b, q) in zip(act, combo):
                out |= b << (12 - 4 * i)
                p *= q
            yield out, p

    def dfs(r: int, d: int, p: float, path: list) -> None:
        if r == rounds:
            if p > best["p"]:
                best["p"], best["path"] = p, list(path)
            return
        left = rounds - r - 1
        for out, q in expand(d):
            np_ = p * q
            if np_ * qmax ** left <= best["p"]:
                continue
            nd = pdiff(out)
            path.append((d, out, nd, q))
            dfs(r + 1, nd, np_, path)
            path.pop()

    for d0 in range(1, 1 << 16):
        if len(active_nibbles(d0)) > max_active_input:
            continue
        dfs(0, d0, 1.0, [])
    if best["path"] is None:
        return None
    path = best["path"]
    return Characteristic(path[0][0], tuple(path), best["p"])


def characteristic_probability(sbox: tuple[int, ...], ch: Characteristic) -> float:
    """Recompute the product of S-box transition probabilities of ``ch``."""
    table = ddt(sbox)
    p = 1.0
    for d_in, d_out, _, _ in ch.steps:
        for i in range(NIBBLES):
            a, b = nibble(d_in, i), nibble(d_out, i)
            if a == 0 and b == 0:
                continue
            p *= table[a, b] / 16.0
    return p


def empirical_probability(cipher: SPN, ch: Characteristic, trials: int,
                          rng: np.random.Generator) -> float:
    """Measure how often the characteristic's output difference occurs.

    The state after ``R-1`` rounds is inspected directly for random plaintext
    pairs under the cipher's actual keys.
    """
    p = rng.integers(0, 1 << 16, size=trials).astype(np.uint16)
    q = p ^ np.uint16(ch.input_diff)
    a, b = cipher.last_round_input(p), cipher.last_round_input(q)
    return float(np.mean((a ^ b) == ch.output_diff))


@dataclass
class DifferentialResult:
    """Outcome of a last-round key-recovery attack."""

    nibble_positions: list[int]
    scores: np.ndarray          # counts for each guess (flattened, MSB nibble major)
    best_guess: tuple[int, ...]
    pairs_used: int
    pairs_filtered: int

    def ranking(self) -> list[tuple[int, ...]]:
        """Guesses ordered from most to least likely."""
        order = np.argsort(-self.scores, kind="stable")
        return [_unflatten(int(i), len(self.nibble_positions)) for i in order]


def _unflatten(idx: int, k: int) -> tuple[int, ...]:
    return tuple((idx >> (4 * (k - 1 - j))) & 0xF for j in range(k))


def differential_attack(cipher: SPN, ch: Characteristic, n_pairs: int,
                        rng: np.random.Generator) -> DifferentialResult:
    """Recover the active nibbles of the last round key ``K[R]``.

    Chosen plaintext pairs with difference ``ch.input_diff`` are encrypted.
    Pairs whose ciphertext difference is non-zero on inactive nibbles cannot
    follow the characteristic and are discarded.  For every guess of the
    active key nibbles the last layer is peeled off and a vote is cast when
    the resulting difference equals ``ch.output_diff``.
    """
    inv = np.array(invert(cipher.sbox), dtype=np.int64)
    act = active_nibbles(ch.output_diff)
    p = rng.integers(0, 1 << 16, size=n_pairs).astype(np.uint16)
    c1 = cipher.encrypt(p).astype(np.int64)
    c2 = cipher.encrypt(p ^ np.uint16(ch.input_diff)).astype(np.int64)
    inactive_mask = 0
    for i in range(NIBBLES):
        if i not in act:
            inactive_mask |= 0xF << (12 - 4 * i)
    keep = ((c1 ^ c2) & inactive_mask) == 0
    c1, c2 = c1[keep], c2[keep]
    npairs = len(c1)
    joint = np.ones((1, npairs), dtype=bool)
    guesses = np.arange(16)
    for i in act:
        sh = 12 - 4 * i
        a, b = (c1 >> sh) & 0xF, (c2 >> sh) & 0xF
        want = nibble(ch.output_diff, i)
        m = (inv[a[None, :] ^ guesses[:, None]] ^ inv[b[None, :] ^ guesses[:, None]]) == want
        joint = (joint[:, None, :] & m[None, :, :]).reshape(joint.shape[0] * 16, npairs)
    scores = joint.sum(axis=1)
    best = _unflatten(int(np.argmax(scores)), len(act))
    return DifferentialResult(act, scores, best, n_pairs, npairs)


def true_subkey(cipher: SPN, positions: list[int]) -> tuple[int, ...]:
    """Ground-truth nibbles of ``K[R]`` at ``positions`` (for evaluation)."""
    return tuple(nibble(cipher.keys[cipher.rounds], i) for i in positions)


def success_rate(sbox: tuple[int, ...], rounds: int, ch: Characteristic,
                 n_pairs: int, trials: int, seed: int = 0) -> float:
    """Fraction of random keys for which the attack ranks the true subkey first."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(trials):
        cipher = SPN.random(rounds, rng, sbox)
        res = differential_attack(cipher, ch, n_pairs, rng)
        wins += res.best_guess == true_subkey(cipher, res.nibble_positions)
    return wins / trials
