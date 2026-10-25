"""Merkle trees with domain separation, inclusion proofs and mutation analysis.

Leaves and interior nodes are hashed with different one-byte prefixes, which
rules out the classic second-preimage attack where an interior node is passed
off as a leaf.  Two policies for odd-sized levels are supported:

* ``duplicate_odd=False`` (default): the unpaired node is promoted unchanged.
* ``duplicate_odd=True``: the unpaired node is paired with itself, which is
  what Bitcoin does.  This makes ``[a, b, c]`` and ``[a, b, c, c]`` share a
  root (CVE-2012-2459), see :func:`roots_collide`.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"

Proof = list[tuple[bytes, bool]]  # (sibling hash, sibling_is_on_the_right)


def sha256d(data: bytes) -> bytes:
    """Double SHA-256 digest."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def hash_leaf(data: bytes) -> bytes:
    return sha256d(LEAF_PREFIX + data)


def hash_node(left: bytes, right: bytes) -> bytes:
    return sha256d(NODE_PREFIX + left + right)


class MerkleTree:
    """Binary Merkle tree over a sequence of byte-string leaves."""

    def __init__(self, leaves: Sequence[bytes], duplicate_odd: bool = False) -> None:
        self.duplicate_odd = duplicate_odd
        self.leaves = list(leaves)
        level = [hash_leaf(x) for x in self.leaves]
        self.levels: list[list[bytes]] = [level]
        while len(level) > 1:
            nxt: list[bytes] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    nxt.append(hash_node(level[i], level[i + 1]))
                elif duplicate_odd:
                    nxt.append(hash_node(level[i], level[i]))
                else:
                    nxt.append(level[i])
            self.levels.append(nxt)
            level = nxt

    def __len__(self) -> int:
        return len(self.leaves)

    @property
    def root(self) -> bytes:
        if not self.leaves:
            return sha256d(b"")
        return self.levels[-1][0]

    @property
    def height(self) -> int:
        return len(self.levels) - 1

    def proof(self, index: int) -> Proof:
        """Authentication path for leaf ``index`` (O(log n) hashes)."""
        if not 0 <= index < len(self.leaves):
            raise IndexError("leaf index out of range")
        path: Proof = []
        idx = index
        for level in self.levels[:-1]:
            sib = idx ^ 1
            if sib < len(level):
                path.append((level[sib], sib > idx))
            elif self.duplicate_odd:
                path.append((level[idx], True))
            idx //= 2
        return path

    def update(self, index: int, data: bytes) -> None:
        """Replace one leaf and recompute only its path to the root."""
        if not 0 <= index < len(self.leaves):
            raise IndexError("leaf index out of range")
        self.leaves[index] = data
        self.levels[0][index] = hash_leaf(data)
        idx = index
        for h in range(len(self.levels) - 1):
            level = self.levels[h]
            parent = idx // 2
            left = level[2 * parent]
            if 2 * parent + 1 < len(level):
                self.levels[h + 1][parent] = hash_node(left, level[2 * parent + 1])
            elif self.duplicate_odd:
                self.levels[h + 1][parent] = hash_node(left, left)
            else:
                self.levels[h + 1][parent] = left
            idx = parent


def verify_proof(leaf: bytes, proof: Proof, root: bytes) -> bool:
    """Check that ``leaf`` is committed to by ``root`` through ``proof``."""
    acc = hash_leaf(leaf)
    for sibling, sibling_right in proof:
        acc = hash_node(acc, sibling) if sibling_right else hash_node(sibling, acc)
    return acc == root


def merkle_root(leaves: Sequence[bytes], duplicate_odd: bool = False) -> bytes:
    return MerkleTree(leaves, duplicate_odd).root


def roots_collide(leaves: Sequence[bytes]) -> bool:
    """True if appending a copy of the last leaf leaves the duplicate-odd root unchanged.

    This is the mutation weakness of Bitcoin-style trees: two different
    transaction lists with the same root.  The promote policy is immune.
    """
    if not leaves:
        return False
    padded = list(leaves) + [leaves[-1]]
    return merkle_root(leaves, True) == merkle_root(padded, True)
