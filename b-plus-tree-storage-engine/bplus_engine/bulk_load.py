"""Bottom-up bulk loading of a B+ tree from sorted input.

Building level by level costs O(n) and produces tightly packed pages with a
controllable fill factor, versus O(n log n) and ~69% average fill for
repeated insertion of random keys.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Tuple

from .nodes import InternalNode, LeafNode, Node, link_leaves
from .tree import BPlusTree


def chunk_sizes(n: int, target: int, minimum: int, maximum: int) -> List[int]:
    """Split ``n`` items into near-equal chunks of about ``target`` items.

    Every chunk lies in ``[minimum, maximum]`` unless ``n < minimum`` (a lone
    chunk, which is only legal for a root).
    """
    if n <= maximum:
        return [n]
    k = max(2, math.ceil(n / target))
    while k > 1 and n // k < minimum:
        k -= 1
    while math.ceil(n / k) > maximum:
        k += 1
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _min_key(node: Node) -> Any:
    while not node.is_leaf:
        node = node.children[0]
    return node.keys[0]


def bulk_load(pairs: Iterable[Tuple[Any, Any]], max_keys: int = 32, fill: float = 0.9) -> BPlusTree:
    """Build a tree from ``pairs`` sorted by strictly increasing key."""
    if not 0.5 <= fill <= 1.0:
        raise ValueError("fill must be in [0.5, 1.0]")
    items = list(pairs)
    for a, b in zip(items, items[1:]):
        if not a[0] < b[0]:
            raise ValueError("bulk_load requires strictly increasing keys")
    tree = BPlusTree(max_keys)
    if not items:
        return tree

    leaf_target = max(tree.min_keys, math.ceil(fill * max_keys))
    sizes = chunk_sizes(len(items), leaf_target, tree.min_keys, max_keys)
    leaves: List[LeafNode] = []
    pos = 0
    for s in sizes:
        leaf = LeafNode()
        chunk = items[pos:pos + s]
        leaf.keys = [k for k, _ in chunk]
        leaf.values = [v for _, v in chunk]
        leaves.append(leaf)
        pos += s
    link_leaves(leaves)

    level: List[Node] = list(leaves)
    min_children, max_children = tree.min_keys + 1, max_keys + 1
    child_target = max(min_children, math.ceil(fill * max_children))
    while len(level) > 1:
        parents: List[Node] = []
        pos = 0
        for s in chunk_sizes(len(level), child_target, min_children, max_children):
            group = level[pos:pos + s]
            node = InternalNode()
            node.children = group
            node.keys = [_min_key(c) for c in group[1:]]
            parents.append(node)
            pos += s
        level = parents
    tree.root = level[0]
    tree._size = len(items)
    return tree
