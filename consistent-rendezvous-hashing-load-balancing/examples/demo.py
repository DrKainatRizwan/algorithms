import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hash_rings import *
from hash_rings.benchmark import compare

print("== Scheme comparison: 20 nodes, 20000 keys, remove one node ==")
print(f"{'scheme':<14}{'peak/mean':>10}{'moved':>9}{'ideal':>9}{'us/lookup':>11}")
for r in compare():
    print(f"{r['scheme']:<14}{r['peak_to_mean']:>10.3f}{r['moved_on_removal']:>9.4f}{r['ideal_moved']:>9.4f}{r['us_per_lookup']:>11.2f}")

print("\n== Virtual nodes vs max ownership (10 nodes, ideal 0.1) ==")
for v in (1, 10, 100, 1000):
    print(f"vnodes={v:<5} max share={max(HashRing([f'n{i}' for i in range(10)], v).ownership().values()):.4f}")

print("\n== Jump hash growth 10 -> 11 buckets ==")
ks = [hash_key(k) for k in range(50000)]
print(f"moved={sum(jump_hash(k,10)!=jump_hash(k,11) for k in ks)/len(ks):.4f} (ideal {1/11:.4f})")

print("\n== Weighted rendezvous (1:2:5) ==")
h = RendezvousHash(seed=3)
w = {"a": 1, "b": 2, "c": 5}
for n, x in w.items(): h.add_node(n, x)
c = load_counts(h.get_node, range(30000), h.nodes)
print(c, f"max rel. error={weighted_error(c, w):.4f}")
