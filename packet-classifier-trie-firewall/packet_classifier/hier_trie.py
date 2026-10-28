"""Binary prefix tries and the hierarchical (two-level) trie classifier."""
from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from .rules import Packet, Rule

T = TypeVar("T")


class _Node(Generic[T]):
    __slots__ = ("kids", "value")

    def __init__(self) -> None:
        self.kids: list[_Node[T] | None] = [None, None]
        self.value: T | None = None


class PrefixTrie(Generic[T]):
    """Uncompressed binary trie over 32-bit prefixes."""

    def __init__(self) -> None:
        self.root: _Node[T] = _Node()
        self.size = 0

    def node_for(self, value: int, length: int, create: bool = False) -> _Node[T] | None:
        node = self.root
        for i in range(length):
            bit = (value >> (31 - i)) & 1
            nxt = node.kids[bit]
            if nxt is None:
                if not create:
                    return None
                nxt = node.kids[bit] = _Node()
            node = nxt
        return node

    def insert(self, value: int, length: int, item: T) -> None:
        node = self.node_for(value, length, create=True)
        assert node is not None
        if node.value is None:
            self.size += 1
        node.value = item

    def get(self, value: int, length: int) -> T | None:
        node = self.node_for(value, length)
        return None if node is None else node.value

    def matches(self, addr: int) -> Iterator[T]:
        """Yield stored values on the path of ``addr``, shortest prefix first."""
        node: _Node[T] | None = self.root
        i = 0
        while node is not None:
            if node.value is not None:
                yield node.value
            if i == 32:
                break
            node = node.kids[(addr >> (31 - i)) & 1]
            i += 1

    def longest_match(self, addr: int) -> T | None:
        best = None
        for v in self.matches(addr):
            best = v
        return best

    def node_count(self) -> int:
        stack, n = [self.root], 0
        while stack:
            cur = stack.pop()
            n += 1
            stack.extend(k for k in cur.kids if k is not None)
        return n


class HierarchicalTrieClassifier:
    """Source trie whose nodes own destination tries holding rule buckets.

    A rule lives at (source-prefix node, destination-prefix node). Because every
    ancestor of the packet's source path is visited, no backtracking is needed:
    cost is O(W^2 + B) where B is the bucket size of the visited nodes.
    """

    def __init__(self, rules: list[Rule]) -> None:
        self.src_trie: PrefixTrie[PrefixTrie[list[Rule]]] = PrefixTrie()
        self.n_rules = len(rules)
        for r in sorted(rules, key=lambda r: r.priority):
            dst_trie = self.src_trie.get(*r.src)
            if dst_trie is None:
                dst_trie = PrefixTrie()
                self.src_trie.insert(r.src[0], r.src[1], dst_trie)
            bucket = dst_trie.get(*r.dst)
            if bucket is None:
                bucket = []
                dst_trie.insert(r.dst[0], r.dst[1], bucket)
            bucket.append(r)

    def lookup(self, p: Packet) -> Rule | None:
        best: Rule | None = None
        for dst_trie in self.src_trie.matches(p.src):
            for bucket in dst_trie.matches(p.dst):
                for r in bucket:  # buckets are priority-sorted
                    if best is not None and r.priority >= best.priority:
                        break
                    if r.match_rest(p):
                        best = r
                        break
        return best

    def classify(self, p: Packet, default: str = "deny") -> str:
        r = self.lookup(p)
        return r.action if r else default

    def memory_nodes(self) -> int:
        """Total trie nodes across both levels (a memory proxy)."""
        total = self.src_trie.node_count()
        stack = [self.src_trie.root]
        while stack:
            n = stack.pop()
            if n.value is not None:
                total += n.value.node_count()
            stack.extend(k for k in n.kids if k is not None)
        return total
