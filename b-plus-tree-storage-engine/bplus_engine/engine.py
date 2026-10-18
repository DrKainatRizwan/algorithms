"""A small ordered key-value store built on the B+ tree.

Adds a write-ahead log of operations, atomic batches, snapshots by
serialisation, and secondary-index style prefix scans over byte-string keys.
Keys are non-negative integers (for the page format) or, for the prefix API,
integers produced by :func:`encode_key`.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional, Tuple

from . import serializer
from .bulk_load import bulk_load
from .tree import BPlusTree

Op = Tuple[str, int, Optional[bytes]]


def encode_key(text: str, width: int = 7) -> int:
    """Order-preserving packing of the first ``width`` bytes of ``text``.

    ``encode_key(a) <= encode_key(b)`` whenever ``a <= b`` bytewise (prefixes
    of equal first ``width`` bytes collide), which lets string prefixes be
    scanned as integer ranges.
    """
    raw = text.encode("utf-8")[:width].ljust(width, b"\0")
    return int.from_bytes(raw, "big")


def prefix_bounds(prefix: str, width: int = 7) -> Tuple[int, int]:
    """Half-open integer range ``[lo, hi)`` covering all keys with ``prefix``."""
    raw = prefix.encode("utf-8")[:width]
    lo = int.from_bytes(raw.ljust(width, b"\0"), "big")
    hi = lo + (1 << (8 * (width - len(raw))))
    return lo, hi


class KVStore:
    """Ordered store with a write-ahead log and atomic batches."""

    def __init__(self, max_keys: int = 64) -> None:
        self.tree = BPlusTree(max_keys)
        self.wal: List[Op] = []

    def put(self, key: int, value: bytes) -> None:
        self.wal.append(("put", key, value))
        self.tree.insert(key, value)

    def get(self, key: int) -> Optional[bytes]:
        return self.tree.get(key)

    def delete(self, key: int) -> bool:
        self.wal.append(("del", key, None))
        return self.tree.delete(key)

    def batch(self, ops: Iterable[Op]) -> None:
        """Apply all operations or none: on failure the tree is restored."""
        ops = list(ops)
        undo: List[Op] = []
        try:
            for kind, key, value in ops:
                old = self.tree.get(key)
                if kind == "put":
                    if not isinstance(value, (bytes, bytearray)):
                        raise TypeError("values must be bytes")
                    undo.append(("put", key, old) if old is not None else ("del", key, None))
                    self.tree.insert(key, value)
                elif kind == "del":
                    if old is not None:
                        undo.append(("put", key, old))
                    self.tree.delete(key)
                else:
                    raise ValueError(f"unknown op {kind!r}")
        except Exception:
            for kind, key, value in reversed(undo):
                if kind == "put":
                    self.tree.insert(key, value)
                else:
                    self.tree.delete(key)
            raise
        self.wal.extend(ops)

    def scan(self, lo: Optional[int] = None, hi: Optional[int] = None) -> Iterator[Tuple[int, bytes]]:
        return self.tree.range(lo, hi)

    def scan_prefix(self, prefix: str) -> Iterator[Tuple[int, bytes]]:
        lo, hi = prefix_bounds(prefix)
        return self.tree.range(lo, hi)

    def snapshot(self, page_size: int = 4096) -> bytes:
        return serializer.dumps(self.tree, page_size)

    @classmethod
    def restore(cls, blob: bytes) -> "KVStore":
        store = cls()
        store.tree = serializer.loads(blob)
        return store

    @classmethod
    def recover(cls, wal: Iterable[Op], max_keys: int = 64) -> "KVStore":
        """Rebuild a store by replaying a log."""
        store = cls(max_keys)
        for kind, key, value in wal:
            if kind == "put":
                store.put(key, value)  # type: ignore[arg-type]
            else:
                store.delete(key)
        return store

    @classmethod
    def from_sorted(cls, pairs: Iterable[Tuple[int, bytes]], max_keys: int = 64, fill: float = 0.9) -> "KVStore":
        store = cls(max_keys)
        store.tree = bulk_load(pairs, max_keys, fill)
        return store
