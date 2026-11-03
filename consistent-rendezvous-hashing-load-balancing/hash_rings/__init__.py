"""Consistent, rendezvous, jump and Maglev hashing for distributed load balancing."""
from .hashing import fnv1a64, hash_key, hash_pair, mix64, unit_float
from .jump_maglev import Maglev, jump_hash, next_prime
from .metrics import coeff_variation, load_counts, moved_fraction, peak_to_mean, snapshot, weighted_error
from .rendezvous import RendezvousHash
from .ring import HashRing

__all__ = [
    "fnv1a64", "hash_key", "hash_pair", "mix64", "unit_float", "Maglev", "jump_hash",
    "next_prime", "coeff_variation", "load_counts", "moved_fraction", "peak_to_mean",
    "snapshot", "weighted_error", "RendezvousHash", "HashRing",
]
