# Kademlia DHT Simulator: XOR Routing, Iterative Lookup and Churn

## Overview
A from-scratch, deterministic simulator of the Kademlia distributed hash table. Nodes hold k-buckets keyed on XOR distance, answer `PING`, `FIND_NODE`, `FIND_VALUE` and `STORE` RPCs, join through a bootstrap contact, perform iterative parallel lookups, and replicate values on the k closest peers. An in-memory network supports node failure and packet loss so lookup scaling and data survival under churn can be measured.

## How it works
- **Metric.** `d(a,b) = a XOR b`. It is symmetric, satisfies the triangle inequality, and is unidirectional: for a given point and distance there is exactly one partner point, so lookups for the same key converge along the same path.
- **Routing table.** Bucket `i` holds up to `k` contacts at distance in `[2^i, 2^(i+1))`. Buckets are LRU lists: contacts that answer are moved to the tail; when a bucket is full, the oldest entry is pinged and only evicted if it is dead (long-lived nodes are preferred). A replacement cache keeps fresh candidates.
- **Iterative lookup.** Start from the `k` closest known contacts, query the `alpha` closest unqueried ones per round, merge returned contacts, drop timeouts, and stop when the `k` closest candidates have all responded.
- **Store/get.** `put` looks up the key's ID and stores on the `k` closest nodes; `get` runs `FIND_VALUE`, which short-circuits once a holder is found.
- **Join and refresh.** A new node inserts its bootstrap contact, looks up its own ID, then refreshes each farther bucket with a random ID from that range. `republish` re-stores held keys on the current closest nodes.

## Complexity
- Lookup: O(log N) rounds, O(alpha * log N + k) RPCs expected.
- Routing table: O(k log N) contacts in practice (at most k * bits), O(k log N) per `closest` query.
- Storage: each value is replicated k times, O(k) per key.

## Usage
```python
from kademlia_dht import build_network

net, nodes = build_network(200, k=8, alpha=3, seed=1)
nodes[0].put("hello", "world")
res = nodes[150].get("hello")
print(res.found, res.value, res.rounds, res.rpcs)
net.kill(nodes[10].id)          # inject a failure
```

## Results
Produced by `python examples/demo.py` (32-bit IDs, k=8, alpha=3, 100 random lookups per size):

| N | mean rounds | mean RPCs | exact-closest accuracy |
|---:|---:|---:|---:|
| 50 | 3.18 | 22.7 | 1.00 |
| 100 | 3.43 | 37.5 | 1.00 |
| 200 | 3.63 | 45.7 | 1.00 |
| 400 | 3.90 | 64.7 | 1.00 |
| 800 | 4.22 | 84.7 | 1.00 |

Fit: rounds ≈ 0.25·log2(N) + 1.72, i.e. logarithmic growth.

Churn (200 nodes, 100 keys): with 50% of nodes killed, 99% of keys were still retrievable. With 80% killed and no republishing, retrieval was 59% for k=8 and 100% for k=20, showing the effect of the replication parameter.

Run the tests with `python -m pytest -q` (20 tests).

## References
- P. Maymounkov and D. Mazières, "Kademlia: A Peer-to-Peer Information System Based on the XOR Metric", IPTPS 2002.
- I. Stoica et al., "Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications", SIGCOMM 2001.
- S. Rhea et al., "Handling Churn in a DHT", USENIX ATC 2004.

---

**Author:** Dr. Kainat Rizwan
