"""Node types for the B+ tree.

Leaves hold the key/value pairs and are chained in a doubly linked list so
that range scans never have to climb back up the tree.  Internal nodes hold
only routing keys: ``children[i]`` covers keys ``< keys[i]`` and
``children[i + 1]`` covers keys ``>= keys[i]``.
"""
from __future__ import annotations

from itertools import count
from typing import Any, List, Optional

_ids = count(1)


class Node:
    """Common base class; every node has a unique page-like identifier."""

    __slots__ = ("node_id", "keys")

    def __init__(self) -> None:
        self.node_id: int = next(_ids)
        self.keys: List[Any] = []

    @property
    def is_leaf(self) -> bool:
        raise NotImplementedError


class LeafNode(Node):
    """Leaf page: sorted keys, parallel values and sibling links."""

    __slots__ = ("values", "next", "prev")

    def __init__(self) -> None:
        super().__init__()
        self.values: List[Any] = []
        self.next: Optional[LeafNode] = None
        self.prev: Optional[LeafNode] = None

    @property
    def is_leaf(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"Leaf#{self.node_id}({self.keys})"


class InternalNode(Node):
    """Internal page: ``len(children) == len(keys) + 1``."""

    __slots__ = ("children",)

    def __init__(self) -> None:
        super().__init__()
        self.children: List[Node] = []

    @property
    def is_leaf(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Internal#{self.node_id}({self.keys})"


def link_leaves(leaves: List[LeafNode]) -> None:
    """Chain a list of leaves left to right."""
    for a, b in zip(leaves, leaves[1:]):
        a.next = b
        b.prev = a
    if leaves:
        leaves[0].prev = None
        leaves[-1].next = None
