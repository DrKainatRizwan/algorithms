"""Demo: lookup scaling, accuracy, and churn resilience of a Kademlia DHT."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kademlia_dht import build_network, churn_experiment, fit_log2_slope, lookup_stats

sizes, hops = [], []
print(f"{'N':>6} {'rounds':>8} {'rpcs':>8} {'accuracy':>9}")
for n in (50, 100, 200, 400, 800):
    net, nodes = build_network(n, seed=3)
    st = lookup_stats(net, nodes, trials=100, seed=5)
    sizes.append(n); hops.append(st.mean_rounds)
    print(f"{n:>6} {st.mean_rounds:>8.2f} {st.mean_rpcs:>8.1f} {st.accuracy:>9.2f}")
a, b = fit_log2_slope(sizes, hops)
print(f"fit: rounds ~ {a:.2f}*log2(N) + {b:.2f}")

print("\nChurn (200 nodes, 100 keys, 50% killed):")
for rep in (False, True):
    r = churn_experiment(200, 100, 0.5, republish=rep, seed=2)
    print(f"  republish={rep!s:<5} retrieval rate = {r:.2f}")
print("\nChurn at 80% killed, bucket size k=8 vs k=20 (no republish):")
for k in (8, 20):
    print(f"  k={k:<2} retrieval rate = {churn_experiment(200, 100, 0.8, seed=2, k=k):.2f}")
