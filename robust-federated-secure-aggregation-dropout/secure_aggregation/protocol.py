"""Four-round secure aggregation with dropout recovery (Bonawitz et al.).

Round 0 (advertise)  every client publishes two DH public keys: one for
                     encrypting shares, one for deriving pairwise masks.
Round 1 (share)      every client Shamir-shares its mask secret key and its
                     self-mask seed among all advertised clients, encrypting
                     each share for its recipient; the server only relays.
Round 2 (masked)     a client sends ``x_u + PRG(b_u) + sum_v +-PRG(s_uv)``.
Round 3 (unmask)     the server announces the survivors; each responding
                     client reveals, for every *dropped* peer, a share of the
                     mask key and, for every *surviving* peer, a share of the
                     self-mask seed -- never both for the same peer.

The server reconstructs the mask keys of dropped clients to cancel their
dangling pairwise masks and the self-mask seeds of survivors to strip the
self-masks, leaving exactly the sum of the survivors' inputs.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .crypto import (G, decrypt, dh_agree, dh_keygen, encrypt, kdf,
                     prg_vector)
from .field import P, shamir_reconstruct, shamir_share
from .quantize import ring_add, ring_sub

STAGES = ("advertise", "share", "masked", "unmask")
_HEADER = struct.Struct(">II")


class AbortError(Exception):
    """The protocol cannot continue safely (too few participants, bad data)."""


def _nonce(sender: int, receiver: int) -> bytes:
    return _HEADER.pack(sender, receiver)


class Client:
    """Honest protocol participant holding a ring-encoded input vector."""

    def __init__(self, uid: int, x: np.ndarray, threshold: int, seed: int,
                 bits: int = 32) -> None:
        if uid <= 0:
            raise ValueError("client ids must be positive")
        self.uid = uid
        self.x = np.asarray(x, dtype=np.uint64)
        self.threshold = threshold
        self.bits = bits
        self._rng = random.Random(seed * 1_000_003 + uid)
        self._directory: dict[int, tuple[int, int]] = {}
        self._held: dict[int, tuple[int, int]] = {}
        self._u2: set[int] = set()

    # -- round 0 ---------------------------------------------------------
    def advertise(self) -> tuple[int, int]:
        """Generate key pairs and return ``(c_pk, s_pk)``."""
        self._c_sk, c_pk = dh_keygen(self._rng)
        self._s_sk, s_pk = dh_keygen(self._rng)
        self._b = self._rng.randrange(1, P)
        return c_pk, s_pk

    # -- round 1 ---------------------------------------------------------
    def share_keys(self, directory: Mapping[int, tuple[int, int]]) -> dict[int, bytes]:
        """Secret-share ``s_sk`` and ``b``; return one ciphertext per peer."""
        if len(directory) < self.threshold:
            raise AbortError("fewer than t clients advertised keys")
        if self.uid not in directory:
            raise AbortError("own keys missing from directory")
        self._directory = dict(directory)
        ids = sorted(directory)
        s_sh = shamir_share(self._s_sk, ids, self.threshold, self._rng)
        b_sh = shamir_share(self._b, ids, self.threshold, self._rng)
        out: dict[int, bytes] = {}
        for v in ids:
            if v == self.uid:
                self._held[self.uid] = (s_sh[v], b_sh[v])
                continue
            key = kdf(dh_agree(self._c_sk, directory[v][0]), b"enc")
            pt = (_HEADER.pack(self.uid, v) + s_sh[v].to_bytes(32, "big")
                  + b_sh[v].to_bytes(32, "big"))
            out[v] = encrypt(key, _nonce(self.uid, v), pt)
        return out

    # -- round 2 ---------------------------------------------------------
    def masked_input(self, inbox: Mapping[int, bytes]) -> np.ndarray:
        """Decrypt incoming shares and return the doubly-masked vector."""
        self._u2 = set(inbox) | {self.uid}
        if len(self._u2) < self.threshold:
            raise AbortError("fewer than t clients completed key sharing")
        for u, blob in inbox.items():
            if u not in self._directory:
                raise AbortError("ciphertext from unknown client")
            key = kdf(dh_agree(self._c_sk, self._directory[u][0]), b"enc")
            pt = decrypt(key, _nonce(u, self.uid), blob)
            if _HEADER.unpack(pt[:8]) != (u, self.uid):
                raise AbortError("misrouted share")
            self._held[u] = (int.from_bytes(pt[8:40], "big"),
                             int.from_bytes(pt[40:72], "big"))
        d = len(self.x)
        y = ring_add(self.x, prg_vector(kdf(self._b, b"self"), d, self.bits), self.bits)
        for v in self._u2:
            if v == self.uid:
                continue
            seed = kdf(dh_agree(self._s_sk, self._directory[v][1]), b"mask")
            m = prg_vector(seed, d, self.bits)
            y = ring_add(y, m, self.bits) if self.uid < v else ring_sub(y, m, self.bits)
        return y

    # -- round 3 ---------------------------------------------------------
    def unmask(self, survivors: Iterable[int]) -> tuple[dict[int, int], dict[int, int]]:
        """Reveal ``s``-shares of dropped peers and ``b``-shares of survivors."""
        surv = set(survivors)
        if not surv <= self._u2:
            raise AbortError("survivor set contains unknown clients")
        if len(surv) < self.threshold:
            raise AbortError("survivor set smaller than threshold")
        dropped = self._u2 - surv
        s_out = {v: self._held[v][0] for v in dropped}
        b_out = {u: self._held[u][1] for u in surv}
        return s_out, b_out


@dataclass
class AggregationResult:
    """Outcome of one protocol execution."""

    total: np.ndarray                 # sum of survivors' inputs (ring elements)
    survivors: list[int]              # clients whose input is in ``total``
    dropped: list[int]                # clients that completed round 1 but not 2
    stats: dict[str, int] = field(default_factory=dict)


def run_protocol(inputs: Mapping[int, np.ndarray], threshold: int,
                 dropouts: Mapping[str, Iterable[int]] | None = None,
                 seed: int = 0, bits: int = 32) -> AggregationResult:
    """Run all four rounds for ``inputs`` (``uid -> ring vector``).

    ``dropouts[stage]`` lists clients that vanish *before* sending in that
    stage.  Raises :class:`AbortError` when fewer than ``threshold`` clients
    remain at any point.
    """
    drop = {s: set(dropouts.get(s, ())) if dropouts else set() for s in STAGES}
    if not 2 <= threshold <= len(inputs):
        raise ValueError("need 2 <= threshold <= number of clients")
    d = len(next(iter(inputs.values())))
    clients = {u: Client(u, x, threshold, seed, bits) for u, x in inputs.items()}
    stats = {"bytes_up": 0, "bytes_down": 0}

    # Round 0
    directory: dict[int, tuple[int, int]] = {}
    for u, c in clients.items():
        if u in drop["advertise"]:
            continue
        directory[u] = c.advertise()
        stats["bytes_up"] += 64
    if len(directory) < threshold:
        raise AbortError("too few clients in round 0")

    # Round 1
    u1 = sorted(directory)
    stats["bytes_down"] += 64 * len(u1) * len(u1)
    outbox: dict[int, dict[int, bytes]] = {}
    for u in u1:
        if u in drop["share"]:
            continue
        outbox[u] = clients[u].share_keys(directory)
        stats["bytes_up"] += sum(len(b) for b in outbox[u].values())
    if len(outbox) < threshold:
        raise AbortError("too few clients in round 1")
    u2 = sorted(outbox)

    # Round 2
    masked: dict[int, np.ndarray] = {}
    for v in u2:
        if v in drop["masked"]:
            continue
        inbox = {u: outbox[u][v] for u in u2 if u != v and v in outbox[u]}
        stats["bytes_down"] += sum(len(b) for b in inbox.values())
        masked[v] = clients[v].masked_input(inbox)
        stats["bytes_up"] += d * bits // 8
    if len(masked) < threshold:
        raise AbortError("too few clients in round 2")
    u3 = sorted(masked)

    # Round 3
    s_shares: dict[int, list[tuple[int, int]]] = {}
    b_shares: dict[int, list[tuple[int, int]]] = {}
    responders = 0
    for u in u3:
        if u in drop["unmask"]:
            continue
        s_out, b_out = clients[u].unmask(u3)
        responders += 1
        stats["bytes_up"] += 32 * (len(s_out) + len(b_out))
        for v, sh in s_out.items():
            s_shares.setdefault(v, []).append((u, sh))
        for w, sh in b_out.items():
            b_shares.setdefault(w, []).append((u, sh))
    if responders < threshold:
        raise AbortError("too few clients in round 3")

    # Server-side unmasking
    total = np.zeros(d, dtype=np.uint64)
    for u in u3:
        total = ring_add(total, masked[u], bits)
    for u in u3:
        b = shamir_reconstruct(b_shares[u][:threshold])
        total = ring_sub(total, prg_vector(kdf(b, b"self"), d, bits), bits)
    dropped = [v for v in u2 if v not in masked]
    for v in dropped:
        s_sk = shamir_reconstruct(s_shares[v][:threshold])
        if pow(G, s_sk, P) != directory[v][1]:
            raise AbortError("reconstructed key inconsistent with advertised key")
        for u in u3:
            seed = kdf(dh_agree(s_sk, directory[u][1]), b"mask")
            m = prg_vector(seed, d, bits)
            # client u added +m when u < v and -m otherwise; undo it.
            total = ring_sub(total, m, bits) if u < v else ring_add(total, m, bits)
    return AggregationResult(total=total, survivors=u3, dropped=dropped, stats=stats)
