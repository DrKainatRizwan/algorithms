# Packet Classifier with Trie-Based Firewall Rule Matching

## Overview

Multi-field packet classification is the core operation of a firewall: given a five-tuple header (source IP, destination IP, source port, destination port, protocol), find the highest-priority rule that matches. This project implements four classifiers from scratch with NumPy used only for seeded workload generation, and checks each against a first-match linear scan:

- `LinearClassifier` – reference O(N) scan.
- `HierarchicalTrieClassifier` – source prefix trie whose nodes own destination prefix tries.
- `TupleSpaceClassifier` – one hash table per (source length, destination length) tuple, with priority pruning.
- `HiCutsClassifier` – a decision tree that cuts the 5-D header space.

It also ships static analysis (shadowed rules, ambiguous conflicts), a text rule parser and a synthetic ClassBench-style rule/traffic generator.

## How it works

**Rules** are hyper-rectangles: prefixes for addresses, ranges for ports, exact-or-wildcard for protocol. The lowest priority number wins.

**Hierarchical trie.** A rule `(S/s, D/d, ...)` is stored in the destination trie hanging off the source-trie node for `S/s`, in a bucket for `D/d`. A lookup walks the packet's source path; at each source node carrying a destination trie it walks the destination path, and scans buckets (kept in priority order) testing ports and protocol. Since every ancestor on the path is visited, no backtracking is needed, and buckets are skipped once they cannot beat the current best.

**Tuple space search.** Rules with the same prefix-length pair are hashed on their masked addresses. Tuples are probed in order of their best priority and the search stops once no remaining tuple can improve on the current match.

**HiCuts.** At each node the field with the most distinct clipped rule projections is cut into equal intervals; rules are replicated into each child they intersect. Recursion stops at `binth` rules, at a depth limit, or when a cut separates nothing. Lookup is a pure descent followed by a short leaf scan.

**Analysis.** `find_shadowed` reports rules fully covered by one higher-priority rule (dead rules); `find_conflicts` reports overlapping rules with different actions where neither contains the other.

## Complexity

| Classifier | Lookup | Space | Update |
|---|---|---|---|
| Linear | O(N) | O(N) | O(1) |
| Hierarchical trie | O(W² + B) | O(N·W) | O(W) |
| Tuple space | O(T + B), T ≤ distinct length pairs | O(N) | O(1) |
| HiCuts | O(depth + binth) | O(N·r), r = replication | rebuild |

W = 32 address bits, B = scanned bucket size.

## Usage

```python
from packet_classifier import HierarchicalTrieClassifier, Packet, parse_ip, parse_rule

rules = [parse_rule("allow tcp 10.0.0.0/8 * 192.168.1.0/24 80-443", 0),
         parse_rule("deny any * * * *", 1)]
fw = HierarchicalTrieClassifier(rules)
print(fw.classify(Packet(parse_ip("10.1.2.3"), parse_ip("192.168.1.9"), 4000, 80, 6)))  # allow
```

Run `python -m pytest -q` for the tests and `python examples/demo.py` for the benchmark.

## Results

Output of `examples/demo.py` (synthetic rule sets, 1500 packets, every classifier verified identical to the linear scan; single-threaded pure Python):

| Rules | Classifier | Build (s) | µs / packet | Notes |
|---:|---|---:|---:|---|
| 200 | linear | – | 150.7 | |
| 200 | hier-trie | 0.003 | 5.6 | 5401 trie nodes |
| 200 | tuple-space | 0.001 | 8.6 | 35 tuples |
| 200 | HiCuts | 0.016 | 5.6 | depth 5, 3.2x replication |
| 1000 | linear | – | 631.3 | |
| 1000 | hier-trie | 0.021 | 6.2 | 25478 trie nodes |
| 1000 | tuple-space | 0.004 | 11.4 | 36 tuples |
| 1000 | HiCuts | 0.555 | 8.3 | depth 12, 65.9x replication |
| 3000 | linear | – | 2375.2 | |
| 3000 | hier-trie | 0.078 | 7.4 | 76378 trie nodes |
| 3000 | tuple-space | 0.011 | 11.9 | 36 tuples |
| 3000 | HiCuts | 1.866 | 9.4 | depth 13, 55.0x replication |

Trie and hash-based lookups stay almost flat as the rule set grows 15x, while the linear scan grows proportionally. HiCuts pays for speed with heavy rule replication on wildcard-rich sets. On the 1000-rule random set, 366 rules are shadowed and 845 pairs conflict.

## References

- Srinivasan, Suri, Varghese. *Packet Classification Using Tuple Space Search.* SIGCOMM 1999.
- Gupta, McKeown. *Packet Classification Using Hierarchical Intelligent Cuttings.* Hot Interconnects 1999.
- Srinivasan, Varghese, Suri, Waldvogel. *Fast and Scalable Layer Four Switching.* SIGCOMM 1998.
- Taylor, Turner. *ClassBench: A Packet Classification Benchmark.* INFOCOM 2005.
- Gupta, McKeown. *Algorithms for Packet Classification.* IEEE Network, 2001.

---

**Author:** Dr. Kainat Rizwan
