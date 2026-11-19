"""Linear cryptanalysis: trail search (piling-up lemma) and Matsui-style key recovery."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from .sbox import invert, lat, parity
from .spn import (NIBBLES, SPN, active_nibbles, default_permutation, nibble,
                  permute)


@dataclass(frozen=True)
class LinearTrail:
    """A multi-round linear approximation.

    ``steps[r] = (mask_in, mask_after_sbox, mask_after_perm, bias)`` where the
    bias is the magnitude of the per-round bias.  ``output_mask`` applies to the
    state entering the final S-box layer.
    """

    input_mask: int
    steps: tuple[tuple[int, int, int, float], ...]
    bias: float

    @property
    def output_mask(self) -> int:
        return self.steps[-1][2]

    @property
    def active_sboxes(self) -> int:
        return sum(len(active_nibbles(s[0])) for s in self.steps)


def piling_up(biases: list[float]) -> float:
    """Combine independent biases: ``2^(m-1) * prod(eps_i)`` (signed)."""
    out = 0.5
    for e in biases:
        out *= 2 * e
    return out


def best_trail(sbox: tuple[int, ...], rounds: int,
               perm: tuple[int, ...] | None = None,
               max_active_input: int = 1,
               max_active_output: int = 2) -> LinearTrail | None:
    """Branch-and-bound search for the linear trail with the largest bias.

    ``rounds`` is the number of rounds covered (the last S-box layer is left
    for key guessing).  The bound multiplies the running piled-up bias by
    ``2 * eps_max`` for every remaining round, since each round has at least
    one active S-box.  ``max_active_output`` caps the number of key nibbles the
    attack must guess (``16^k`` work).
    """
    perm = perm or default_permutation()
    table = lat(sbox)
    emax = np.abs(table[1:, 1:]).max() / 16.0
    trans = {a: sorted(((b, abs(table[a, b]) / 16.0) for b in range(1, 16) if table[a, b]),
                       key=lambda t: -t[1]) for a in range(1, 16)}
    best: dict = {"b": 0.0, "path": None}

    @lru_cache(maxsize=None)
    def pmask(m: int) -> int:
        return permute(m, perm)

    def dfs(r: int, m: int, bias: float, path: list) -> None:
        if r == rounds:
            if bias > best["b"] and len(active_nibbles(m)) <= max_active_output:
                best["b"], best["path"] = bias, list(path)
            return
        left = rounds - r - 1
        act = [(i, nibble(m, i)) for i in range(NIBBLES) if nibble(m, i)]
        for combo in product(*(trans[a] for _, a in act)):
            out, nb = 0, bias
            for (i, _), (b, e) in zip(act, combo):
                out |= b << (12 - 4 * i)
                nb *= 2 * e
            if nb * (2 * emax) ** left <= best["b"]:
                continue
            nm = pmask(out)
            path.append((m, out, nm, nb / bias if bias else 0.0))
            dfs(r + 1, nm, nb, path)
            path.pop()

    for m0 in range(1, 1 << 16):
        if len(active_nibbles(m0)) > max_active_input:
            continue
        dfs(0, m0, 0.5, [])
    if best["path"] is None:
        return None
    path = best["path"]
    return LinearTrail(path[0][0], tuple(path), best["b"])


def trail_bias(sbox: tuple[int, ...], trail: LinearTrail) -> float:
    """Recompute the piled-up bias of ``trail`` from the LAT."""
    table = lat(sbox)
    biases = []
    for m_in, m_out, _, _ in trail.steps:
        for i in range(NIBBLES):
            a, b = nibble(m_in, i), nibble(m_out, i)
            if a or b:
                biases.append(abs(table[a, b]) / 16.0)
    return abs(piling_up(biases))


def masked_parity(values: np.ndarray, mask: int) -> np.ndarray:
    """Parity of ``values & mask`` for an array of 16-bit words."""
    v = values.astype(np.uint32) & mask
    v ^= v >> 8
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    return (v & 1).astype(np.int64)


def empirical_bias(cipher: SPN, trail: LinearTrail, n: int,
                   rng: np.random.Generator) -> float:
    """Measured ``|P[alpha.x = beta.y] - 1/2|`` on the first ``R-1`` rounds."""
    p = rng.integers(0, 1 << 16, size=n).astype(np.uint16)
    y = cipher.last_round_input(p)
    z = masked_parity(p, trail.input_mask) ^ masked_parity(y, trail.output_mask)
    return abs(float(np.mean(z == 0)) - 0.5)


@dataclass
class LinearResult:
    """Outcome of a Matsui Algorithm-2 style attack."""

    nibble_positions: list[int]
    scores: np.ndarray       # |T - N/2| / N for each guess
    best_guess: tuple[int, ...]
    samples: int

    def ranking(self) -> list[tuple[int, ...]]:
        order = np.argsort(-self.scores, kind="stable")
        k = len(self.nibble_positions)
        return [tuple((int(i) >> (4 * (k - 1 - j))) & 0xF for j in range(k)) for i in order]


def linear_attack(cipher: SPN, trail: LinearTrail, n_known: int,
                  rng: np.random.Generator) -> LinearResult:
    """Recover the active nibbles of ``K[R]`` from ``n_known`` known plaintexts.

    Ciphertext nibbles at the active positions of ``output_mask`` are
    partially decrypted under every guess; the empirical bias of
    ``alpha.P xor beta.x`` is the score, and the true subkey maximises it.
    Counts are first aggregated per (active ciphertext nibbles, plaintext parity)
    class so the cost per guess is ``16^k`` rather than ``N``.
    """
    inv = np.array(invert(cipher.sbox), dtype=np.int64)
    act = active_nibbles(trail.output_mask)
    k = len(act)
    p = rng.integers(0, 1 << 16, size=n_known).astype(np.uint16)
    c = cipher.encrypt(p).astype(np.int64)
    q = masked_parity(p, trail.input_mask)
    idx = np.zeros(n_known, dtype=np.int64)
    for i in act:
        idx = (idx << 4) | ((c >> (12 - 4 * i)) & 0xF)
    classes = 16 ** k
    counts = np.zeros((classes, 2), dtype=np.int64)
    np.add.at(counts, (idx, q), 1)
    # Partial decryption for each (guess, class) pair.
    cls = np.arange(classes)
    gss = np.arange(classes)
    par = np.zeros((classes, classes), dtype=np.int64)
    for j, i in enumerate(act):
        sh = 4 * (k - 1 - j)
        cn = (cls >> sh) & 0xF
        gn = (gss >> sh) & 0xF
        x = inv[cn[None, :] ^ gn[:, None]]
        bm = nibble(trail.output_mask, i)
        par ^= np.array([parity(v & bm) for v in range(16)])[x]
    # z = parity(beta.x) ^ q ; hits where z == 0
    hits = (par == 0) @ counts[:, 0] + (par == 1) @ counts[:, 1]
    scores = np.abs(hits - n_known / 2) / n_known
    best = tuple((int(np.argmax(scores)) >> (4 * (k - 1 - j))) & 0xF for j in range(k))
    return LinearResult(act, scores, best, n_known)


def true_subkey(cipher: SPN, positions: list[int]) -> tuple[int, ...]:
    """Ground-truth nibbles of ``K[R]`` at ``positions``."""
    return tuple(nibble(cipher.keys[cipher.rounds], i) for i in positions)


def success_rate(sbox: tuple[int, ...], rounds: int, trail: LinearTrail,
                 n_known: int, trials: int, seed: int = 0) -> float:
    """Fraction of random keys for which the true subkey scores highest."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(trials):
        cipher = SPN.random(rounds, rng, sbox)
        res = linear_attack(cipher, trail, n_known, rng)
        wins += res.best_guess == true_subkey(cipher, res.nibble_positions)
    return wins / trials


def required_samples(bias: float, confidence_factor: float = 8.0) -> int:
    """Matsui's rule of thumb: roughly ``c / eps^2`` known plaintexts."""
    return int(np.ceil(confidence_factor / (bias * bias)))
