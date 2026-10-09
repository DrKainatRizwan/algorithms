"""Privacy-preserving aggregation protocols built on Paillier."""
from __future__ import annotations

import random
from typing import Sequence

from .encoding import EncryptedNumber
from .paillier import PrivateKey, PublicKey


def encrypt_vector(public: PublicKey, values: Sequence[float], rng: random.Random) -> list[EncryptedNumber]:
    """Encrypt each coordinate of a vector."""
    return [EncryptedNumber.encrypt(public, v, rng) for v in values]


def decrypt_vector(private: PrivateKey, enc: Sequence[EncryptedNumber]) -> list[float]:
    return [e.decrypt(private) for e in enc]


def sum_ciphertexts(items: Sequence[EncryptedNumber]) -> EncryptedNumber:
    """Homomorphic sum of a non-empty sequence of encrypted numbers."""
    if not items:
        raise ValueError("nothing to aggregate")
    acc = items[0]
    for it in items[1:]:
        acc = acc + it
    return acc


def private_sum(private: PrivateKey, client_values: Sequence[float], rng: random.Random) -> float:
    """Server adds client ciphertexts; only the total is ever decrypted."""
    pub = private.public
    enc = [EncryptedNumber.encrypt(pub, v, rng) for v in client_values]
    return sum_ciphertexts(enc).decrypt(private)


def private_mean_vectors(
    private: PrivateKey, client_vectors: Sequence[Sequence[float]], rng: random.Random
) -> list[float]:
    """Coordinate-wise mean of client vectors (e.g. federated gradients)."""
    pub = private.public
    dim = len(client_vectors[0])
    if any(len(v) != dim for v in client_vectors):
        raise ValueError("ragged client vectors")
    encrypted = [encrypt_vector(pub, v, rng) for v in client_vectors]
    k = len(client_vectors)
    totals = [sum_ciphertexts([e[j] for e in encrypted]) for j in range(dim)]
    return [t.decrypt(private) / k for t in totals]


def weighted_sum(
    private: PrivateKey, values: Sequence[float], weights: Sequence[float], rng: random.Random
) -> float:
    """Server-known weights applied to encrypted values: sum_i w_i * Enc(v_i)."""
    pub = private.public
    terms = [EncryptedNumber.encrypt(pub, v, rng) * w for v, w in zip(values, weights, strict=True)]
    return sum_ciphertexts(terms).decrypt(private)


def encrypted_dot(enc_x: Sequence[EncryptedNumber], w: Sequence[float]) -> EncryptedNumber:
    """Dot product of an encrypted feature vector with plaintext model weights."""
    return sum_ciphertexts([x * wi for x, wi in zip(enc_x, w, strict=True)])


def private_vote(private: PrivateKey, ballots: Sequence[int], n_options: int, rng: random.Random) -> list[int]:
    """Tally one-hot encrypted ballots without seeing any individual vote."""
    pub = private.public
    tallies = [EncryptedNumber.encrypt(pub, 0, rng) for _ in range(n_options)]
    for b in ballots:
        if not 0 <= b < n_options:
            raise ValueError("invalid ballot")
        for j in range(n_options):
            tallies[j] = tallies[j] + EncryptedNumber.encrypt(pub, 1 if j == b else 0, rng)
    return [int(round(t.decrypt(private))) for t in tallies]


def blinded_sum_protocol(
    private: PrivateKey, client_values: Sequence[int], rng: random.Random
) -> tuple[int, list[int]]:
    """Additive-mask variant: the decryptor sees only masked ciphertext contents.

    The aggregator homomorphically adds a random mask R to the encrypted total, the
    key holder decrypts total+R, and the aggregator removes R. Returns the total and
    the masked value that the key holder observed.
    """
    pub = private.public
    cts = [pub.raw_encrypt(v, rng) for v in client_values]
    total = cts[0]
    for c in cts[1:]:
        total = pub.add(total, c)
    mask = rng.randrange(pub.n)
    masked = pub.add(total, pub.raw_encrypt(mask, rng))
    seen = private.decrypt(masked)
    result = (seen - mask) % pub.n
    if result > pub.n // 2:
        result -= pub.n
    return result, [seen]
