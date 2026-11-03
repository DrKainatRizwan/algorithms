# Consistent and Rendezvous Hashing for Distributed Load Balancing

## Overview
From-scratch implementations of four key-to-node assignment schemes used in caches, sharded stores and load balancers: a virtual-node consistent hash ring, weighted rendezvous (HRW) hashing, jump consistent hash and Maglev lookup tables. A small metrics and benchmark harness measures balance, churn and lookup cost. Only the standard library is used; hashing is a seedable FNV-1a + SplitMix64 construction, so results are reproducible.

## How it works
- **Ring** (Karger et al., 1997): each node owns `round(vnodes * weight)` points on a 2^64 ring; a key goes to the first point clockwise (binary search). Removing a node moves only that node's keys. More virtual nodes smooth the load.
- **Rendezvous** (Thaler & Ravishankar, 1998): score every node for the key and pick the maximum. Weighted score `-w / ln(u)` yields exact probability `w_i / sum(w)`. Top-k scores give a replica list.
- **Jump hash** (Lamping & Veach, 2014): a stateless O(ln n) loop; growing n to n+1 buckets moves exactly 1/(n+1) of the keys, only onto the new bucket.
- **Maglev** (Eisenbud et al., 2016): each node has a permutation of a prime-sized table; nodes take turns claiming their next free slot, giving near-equal slot counts and low churn.

## Complexity
| Scheme | Lookup | Add/remove node | Space |
|---|---|---|---|
| Ring | O(log(N v)) | O(v log(N v)) add, O(N v) remove | O(N v) |
| Rendezvous | O(N) | O(1) | O(N) |
| Jump | O(ln N) | append/trim only | O(1) |
| Maglev | O(1) | O(M) rebuild | O(M) |

## Usage
```python
from hash_rings import HashRing, RendezvousHash, jump_hash, Maglev

ring = HashRing(["a", "b", "c"], vnodes=100)
ring.get_node("user:42"); ring.get_nodes("user:42", 2)   # primary / replicas
ring.add_node("d", weight=2.0)

hrw = RendezvousHash(["a", "b", "c"]); hrw.get_node("user:42")
Maglev(["a", "b", "c"], 10007).get_node("user:42")
```
Run `python examples/demo.py` and `python -m pytest -q`.

## Results (demo output)
20 nodes, 20,000 keys, one node removed:

| Scheme | Peak/mean load | Keys moved | Ideal | µs/lookup |
|---|---|---|---|---|
| Ring (v=100) | 1.175 | 0.0486 | 0.0486 | 2.9 |
| Rendezvous | 1.061 | 0.0520 | 0.0520 | 115.2 |
| Maglev (M=10007) | 1.056 | 0.0592 | 0.0503 | 2.6 |

Largest ring ownership with 10 nodes (ideal 0.1): 0.289 (v=1), 0.158 (v=10), 0.122 (v=100), 0.107 (v=1000). Jump hash 10→11 buckets moved 0.0884 (ideal 0.0909). Weighted rendezvous with weights 1:2:5 had a maximum relative share error of 0.6%.

## References
- Karger et al., "Consistent Hashing and Random Trees", STOC 1997.
- Thaler and Ravishankar, "Using Name-Based Mappings to Increase Hit Rates", IEEE/ACM ToN 1998.
- Lamping and Veach, "A Fast, Minimal Memory, Consistent Hash Algorithm", 2014.
- Eisenbud et al., "Maglev: A Fast and Reliable Software Network Load Balancer", NSDI 2016.
- Resch, "Weighted Rendezvous Hashing" (2008).

---

**Author:** Dr. Kainat Rizwan
