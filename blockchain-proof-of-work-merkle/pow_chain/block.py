"""Transactions, block headers and blocks."""
from __future__ import annotations

import struct
from dataclasses import dataclass, replace

from .merkle import MerkleTree, sha256d

ZERO_HASH = "00" * 32


@dataclass(frozen=True)
class Transaction:
    """Account-model transfer.  ``sender == ""`` marks a coinbase."""

    sender: str
    recipient: str
    amount: int
    fee: int = 0
    nonce: int = 0

    def serialize(self) -> bytes:
        return f"{self.sender}|{self.recipient}|{self.amount}|{self.fee}|{self.nonce}".encode()

    @property
    def txid(self) -> str:
        return sha256d(self.serialize()).hex()

    @property
    def is_coinbase(self) -> bool:
        return self.sender == ""

    @staticmethod
    def coinbase(recipient: str, amount: int, height: int) -> "Transaction":
        """Reward transaction; the height goes in the nonce to keep txids unique."""
        return Transaction("", recipient, amount, 0, height)


@dataclass(frozen=True)
class BlockHeader:
    version: int
    prev_hash: str
    merkle_root: str
    timestamp: int
    bits: int
    nonce: int = 0

    def serialize(self) -> bytes:
        return struct.pack(
            ">I32s32sIII",
            self.version,
            bytes.fromhex(self.prev_hash),
            bytes.fromhex(self.merkle_root),
            self.timestamp,
            self.bits,
            self.nonce,
        )

    def digest(self) -> bytes:
        return sha256d(self.serialize())

    @property
    def hash(self) -> str:
        return self.digest().hex()

    @property
    def hash_int(self) -> int:
        return int.from_bytes(self.digest(), "big")

    def with_nonce(self, nonce: int) -> "BlockHeader":
        return replace(self, nonce=nonce)


def tx_merkle_root(txs: tuple[Transaction, ...] | list[Transaction]) -> str:
    return MerkleTree([bytes.fromhex(t.txid) for t in txs]).root.hex()


@dataclass(frozen=True)
class Block:
    header: BlockHeader
    txs: tuple[Transaction, ...]

    @property
    def hash(self) -> str:
        return self.header.hash

    @property
    def prev_hash(self) -> str:
        return self.header.prev_hash

    @property
    def timestamp(self) -> int:
        return self.header.timestamp

    @staticmethod
    def assemble(prev_hash: str, txs: list[Transaction], timestamp: int, bits: int,
                 version: int = 1) -> "Block":
        """Build an unmined block (nonce 0) committing to ``txs``."""
        header = BlockHeader(version, prev_hash, tx_merkle_root(txs), timestamp, bits, 0)
        return Block(header, tuple(txs))

    def with_nonce(self, nonce: int) -> "Block":
        return Block(self.header.with_nonce(nonce), self.txs)
