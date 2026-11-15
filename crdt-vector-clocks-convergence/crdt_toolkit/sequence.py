"""Replicated Growable Array (RGA) for collaborative text editing.

Every character is a node with a globally unique id ``(lamport, node)`` and a
parent id (the character it was inserted after).  Visible order is a
depth-first traversal in which siblings are sorted by descending id, so a
later insertion at the same anchor appears first.  Deletion marks a tombstone.
The merge is a set union with tombstone-OR, hence a join.
"""
from __future__ import annotations

from typing import Any

from .crdts import CRDT

ROOT = (0, "")
NodeId = tuple[int, str]


class RGA(CRDT):
    """Convergent sequence of characters."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        # id -> [char, parent_id, deleted]
        self.nodes: dict[NodeId, list[Any]] = {}
        self.clock = 0

    # -- traversal ---------------------------------------------------------
    def _children(self) -> dict[NodeId, list[NodeId]]:
        kids: dict[NodeId, list[NodeId]] = {}
        for nid, (_, parent, _) in self.nodes.items():
            kids.setdefault(parent, []).append(nid)
        for lst in kids.values():
            lst.sort(reverse=True)
        return kids

    def order(self) -> list[NodeId]:
        """All node ids (including tombstones) in document order."""
        kids = self._children()
        out: list[NodeId] = []
        stack = list(reversed(kids.get(ROOT, [])))
        while stack:
            nid = stack.pop()
            out.append(nid)
            stack.extend(reversed(kids.get(nid, [])))
        return out

    def visible_ids(self) -> list[NodeId]:
        return [n for n in self.order() if not self.nodes[n][2]]

    # -- local operations --------------------------------------------------
    def insert(self, index: int, char: str) -> NodeId:
        """Insert ``char`` so it becomes visible position ``index``."""
        vis = self.visible_ids()
        if not 0 <= index <= len(vis):
            raise IndexError(index)
        parent = ROOT if index == 0 else vis[index - 1]
        self.clock += 1
        nid = (self.clock, self.node_id)
        self.nodes[nid] = [char, parent, False]
        return nid

    def delete(self, index: int) -> NodeId:
        vis = self.visible_ids()
        if not 0 <= index < len(vis):
            raise IndexError(index)
        self.nodes[vis[index]][2] = True
        return vis[index]

    def insert_text(self, index: int, text: str) -> None:
        for i, ch in enumerate(text):
            self.insert(index + i, ch)

    # -- CRDT interface ----------------------------------------------------
    def merge(self, other: "RGA") -> None:  # type: ignore[override]
        for nid, (ch, parent, dead) in other.nodes.items():
            mine = self.nodes.get(nid)
            if mine is None:
                self.nodes[nid] = [ch, parent, dead]
            elif dead:
                mine[2] = True
        self.clock = max(self.clock, other.clock)

    def text(self) -> str:
        return "".join(self.nodes[n][0] for n in self.visible_ids())

    def value(self) -> str:
        return self.text()

    def canonical(self) -> Any:
        return tuple(sorted((nid, v[0], v[1], v[2]) for nid, v in self.nodes.items()))

    def __len__(self) -> int:
        return len(self.visible_ids())
