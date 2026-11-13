"""1-out-of-2 oblivious transfer (Chou-Orlandi "simplest OT") over Z_p*.

Sender holds message pairs (m0, m1); the receiver holds choice bits. The
receiver learns exactly m_choice and the sender learns nothing about the
choice. The group is Z_p* with p = 2^255 - 19, generator 2. This is meant
for study, not for production use.
"""

from __future__ import annotations

import hashlib
import random

from .primitives import MASK

P = 2**255 - 19
G = 2


def _kdf(point: int, index: int) -> int:
    data = point.to_bytes(32, "big") + index.to_bytes(8, "big")
    return int.from_bytes(hashlib.sha256(data).digest()[:16], "big")


class OTSender:
    """Sender side of a batch of base OTs sharing one public value A."""

    def __init__(self, rng: random.Random) -> None:
        self._a = rng.randrange(2, P - 1)
        self.public = pow(G, self._a, P)

    def respond(self, receiver_msgs: list[int],
                pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Encrypt each pair under the two keys derivable from ``B``."""
        if len(receiver_msgs) != len(pairs):
            raise ValueError("length mismatch")
        a_inv = pow(self.public, -1, P)
        out = []
        for i, (b_pt, (m0, m1)) in enumerate(zip(receiver_msgs, pairs)):
            k0 = _kdf(pow(b_pt, self._a, P), i)
            k1 = _kdf(pow(b_pt * a_inv % P, self._a, P), i)
            out.append(((m0 ^ k0) & MASK, (m1 ^ k1) & MASK))
        return out


class OTReceiver:
    """Receiver side: commits to choice bits, then decrypts its messages."""

    def __init__(self, choices: list[int], rng: random.Random) -> None:
        self.choices = list(choices)
        self._b = [rng.randrange(2, P - 1) for _ in choices]
        self._sender_pub = 0

    def choose(self, sender_public: int) -> list[int]:
        """Produce B_i = g^b (choice 0) or A*g^b (choice 1)."""
        self._sender_pub = sender_public
        out = []
        for c, b in zip(self.choices, self._b):
            pt = pow(G, b, P)
            out.append(pt * sender_public % P if c else pt)
        return out

    def recover(self, ciphertexts: list[tuple[int, int]]) -> list[int]:
        """Decrypt the chosen message of each pair with key H(A^b)."""
        res = []
        for i, (c, b) in enumerate(zip(self.choices, self._b)):
            key = _kdf(pow(self._sender_pub, b, P), i)
            res.append(ciphertexts[i][c] ^ key)
        return res


def run_ot(pairs: list[tuple[int, int]], choices: list[int],
           seed: int = 0) -> list[int]:
    """Run the full 3-message protocol locally and return received messages."""
    rng = random.Random(seed)
    sender = OTSender(rng)
    receiver = OTReceiver(choices, rng)
    b_msgs = receiver.choose(sender.public)
    return receiver.recover(sender.respond(b_msgs, pairs))
