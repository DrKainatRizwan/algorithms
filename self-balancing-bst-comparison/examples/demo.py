"""Compare treap, splay and red-black trees on four workloads."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balanced_trees import RedBlackTree, SplayTree, Treap, compare_all, format_table

t = Treap(seed=1)
for k in range(10):
    t.insert(k * 3, f"v{k}")
print("Treap keys:", t.keys(), "| 4th smallest:", t.select(3), "| rank(10):", t.rank(10))

rb = RedBlackTree()
for k in range(1000):
    rb.insert(k)
rb.validate()
print(f"Red-black, 1000 sequential inserts: height={rb.height()}, rotations={rb.rotations}")

sp = SplayTree()
for k in range(1000):
    sp.insert(k)
print(f"Splay, 1000 sequential inserts: height={sp.height()} (degenerate path, amortised O(log n))")

print()
print(format_table(compare_all(n=2000, seed=0)))
