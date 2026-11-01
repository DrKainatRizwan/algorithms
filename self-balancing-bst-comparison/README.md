# Self-Balancing BST Comparison: Treap, Splay Tree and Red-Black Tree

## Overview

Three classic balanced search trees implemented from scratch behind one ordered-map interface (`insert`, `get`, `delete`, `floor`, `ceiling`, `range`, `height`, `validate`) with built-in counters for key comparisons and rotations. A workload module generates sequential, uniform, Zipfian and working-set traffic so the trees can be compared on equal terms.

Package layout (`balanced_trees/`):

| Module | Contents |
|---|---|
| `base.py` | `SortedMap` ABC: iteration, height, floor/ceiling/range, counters |
| `treap.py` | Split/merge treap with size augmentation (`rank`, `select`, `split`, `concat`) |
| `splay.py` | Top-down splay tree (Sleator–Tarjan) |
| `redblack.py` | CLRS red-black tree with sentinel NIL and full invariant checker |
| `workloads.py` | Workload generators, benchmark harness, table formatting |

## How the algorithms work

- **Treap** – each key gets an independent random priority; the tree is a BST on keys and a max-heap on priorities. Insertion splits the tree at the key and merges `left + node + right`; deletion merges the node's children. Because priorities are independent of keys, the shape equals that of a randomly-built BST: expected height O(log n) for any insertion order.
- **Splay tree** – no balance information. Every access splays the key to the root using zig-zig and zig-zag steps (done top-down in one pass). Individual operations can cost O(n), but any sequence of m operations costs O(m log n) amortised, and frequently accessed keys stay near the root (working-set and static-optimality properties).
- **Red-black tree** – nodes are coloured so that the root is black, red nodes have black children, and all root-to-leaf paths contain equal black counts. This bounds height by 2·log₂(n+1). Insertion needs at most 2 rotations, deletion at most 3, plus O(log n) recolourings.

## Complexity

| Tree | Search | Insert / delete | Extra space per node |
|---|---|---|---|
| Treap | O(log n) expected | O(log n) expected | priority (+ size) |
| Splay | O(log n) amortised | O(log n) amortised | none |
| Red-black | O(log n) worst case | O(log n) worst case | 1 colour bit + parent |

All use O(n) space overall. `Treap.split`/`concat` run in O(log n) expected time.

## Usage

```python
from balanced_trees import Treap, SplayTree, RedBlackTree

t = Treap(seed=1)
for k in range(10):
    t.insert(k * 3, f"v{k}")
print(t.select(3), t.rank(10))     # 9 4
print(t.floor(10), t.ceiling(10))  # 9 12

rb = RedBlackTree()
for k in range(1000):
    rb.insert(k)
rb.validate()
print(rb.height())                 # 17
```

Run the tests with `python -m pytest -q` and the comparison with `python examples/demo.py`.

## Results

Demo output (n = 2000 keys, 8000 lookups for the skewed workloads; "cmp/ins" for the treap includes the duplicate-check descent plus split and merge steps; treap rotations are not counted on insert since it uses split/merge):

```
workload    tree       height  avg depth  cmp/ins  rot/ins  cmp/get
-------------------------------------------------------------------
sequential  treap          27      14.13    24.42     0.00    14.13
sequential  splay        2000    1000.50     1.00     0.00    21.52
sequential  red-black      19      10.40    16.46     0.99    10.40
random      treap          25      12.69    38.15     0.00    12.69
random      splay          30      16.35    19.71     5.54    20.35
random      red-black      13      10.26     9.80     0.58    10.26
zipf        treap          25      12.69    38.15     0.00     9.81
zipf        splay          30      16.35    19.71     5.54    10.36
zipf        red-black      13      10.26     9.80     0.58     9.28
working-set treap          25      12.69    38.15     0.00    12.77
working-set splay          30      16.35    19.71     5.54     4.92
working-set red-black      13      10.26     9.80     0.58    10.27
```

Observations: the red-black tree has the lowest height and flattest depth; the treap is somewhat deeper but needs no case analysis; the splay tree degenerates into a path after sorted inserts yet more than halves lookup cost on the working-set workload (4.92 vs 10.27 comparisons) and is on par with the others under Zipf traffic. The 28 tests cover dictionary equivalence under random operations, invariants, height bounds, rotation limits, rank/select and split/concat.

## References

- Aragon & Seidel, "Randomized Search Trees", FOCS 1989 / Algorithmica 16, 1996.
- Sleator & Tarjan, "Self-Adjusting Binary Search Trees", J. ACM 32(3), 1985.
- Guibas & Sedgewick, "A Dichromatic Framework for Balanced Trees", FOCS 1978.
- Cormen, Leiserson, Rivest & Stein, *Introduction to Algorithms*, 4th ed., ch. 13.

---

**Author:** Dr. Kainat Rizwan
