"""Fixed-size page serialisation of a B+ tree (integer keys, bytes values).

Layout: page 0 is a header (magic, page size, max_keys, root page, entry
count, page count).  Every tree node occupies one page numbered from 1:

    leaf:     type=1 | nkeys | prev | next | keys[q] | (len, bytes)*
    internal: type=2 | nkeys | keys[q] | children[q]  (nkeys + 1 page ids)

Page ids of 0 denote "none" for the sibling links.
"""
from __future__ import annotations

import struct
from typing import Dict, List

from .nodes import InternalNode, LeafNode, Node
from .tree import BPlusTree

MAGIC = b"BPT1"
_HEADER = struct.Struct("<4sIIqqq")
_NODE_HEAD = struct.Struct("<BIqq")


class PageOverflow(ValueError):
    """Raised when a node does not fit in a single page."""


def max_keys_for_page(page_size: int, value_size: int) -> int:
    """Largest ``max_keys`` such that a full leaf of fixed-size values fits."""
    per_entry = 8 + 4 + value_size
    return (page_size - _NODE_HEAD.size) // per_entry


def _encode_node(node: Node, ids: Dict[int, int], page_size: int) -> bytes:
    if node.is_leaf:
        prev_id = ids[node.prev.node_id] if node.prev else 0
        next_id = ids[node.next.node_id] if node.next else 0
        out = [_NODE_HEAD.pack(1, len(node.keys), prev_id, next_id)]
        out.append(struct.pack(f"<{len(node.keys)}q", *node.keys))
        for v in node.values:
            if not isinstance(v, (bytes, bytearray)):
                raise TypeError("values must be bytes")
            out.append(struct.pack("<I", len(v)))
            out.append(bytes(v))
    else:
        out = [_NODE_HEAD.pack(2, len(node.keys), 0, 0)]
        out.append(struct.pack(f"<{len(node.keys)}q", *node.keys))
        out.append(struct.pack(f"<{len(node.children)}q", *(ids[c.node_id] for c in node.children)))
    data = b"".join(out)
    if len(data) > page_size:
        raise PageOverflow(f"node needs {len(data)} bytes, page is {page_size}")
    return data.ljust(page_size, b"\0")


def dumps(tree: BPlusTree, page_size: int = 4096) -> bytes:
    """Serialise ``tree`` into ``(1 + #nodes) * page_size`` bytes."""
    nodes = list(tree.nodes())
    ids = {n.node_id: i + 1 for i, n in enumerate(nodes)}
    header = _HEADER.pack(MAGIC, page_size, tree.max_keys, ids[tree.root.node_id], len(tree), len(nodes))
    pages = [header.ljust(page_size, b"\0")]
    pages += [_encode_node(n, ids, page_size) for n in nodes]
    return b"".join(pages)


def loads(blob: bytes) -> BPlusTree:
    """Rebuild a tree from the output of :func:`dumps`."""
    magic, page_size, max_keys, root_id, count, npages = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if len(blob) != (npages + 1) * page_size:
        raise ValueError("truncated or oversized image")
    raw: Dict[int, tuple] = {}
    built: Dict[int, Node] = {}
    for pid in range(1, npages + 1):
        base = pid * page_size
        kind, n, a, b = _NODE_HEAD.unpack_from(blob, base)
        off = base + _NODE_HEAD.size
        keys: List[int] = list(struct.unpack_from(f"<{n}q", blob, off))
        off += 8 * n
        if kind == 1:
            leaf = LeafNode()
            leaf.keys = keys
            for _ in range(n):
                (ln,) = struct.unpack_from("<I", blob, off)
                off += 4
                leaf.values.append(bytes(blob[off:off + ln]))
                off += ln
            built[pid] = leaf
            raw[pid] = (1, a, b)
        elif kind == 2:
            kids = list(struct.unpack_from(f"<{n + 1}q", blob, off))
            node = InternalNode()
            node.keys = keys
            built[pid] = node
            raw[pid] = (2, kids)
        else:
            raise ValueError(f"bad page type {kind} at page {pid}")
    leaves = []
    for pid, info in raw.items():
        if info[0] == 2:
            built[pid].children = [built[c] for c in info[1]]
        else:
            leaf = built[pid]
            leaf.prev = built[info[1]] if info[1] else None
            leaf.next = built[info[2]] if info[2] else None
            leaves.append(leaf)
    tree = BPlusTree(max_keys)
    tree.root = built[root_id]
    tree._size = count
    return tree


def save(tree: BPlusTree, path: str, page_size: int = 4096) -> None:
    with open(path, "wb") as fh:
        fh.write(dumps(tree, page_size))


def load(path: str) -> BPlusTree:
    with open(path, "rb") as fh:
        return loads(fh.read())
