"""Benchmark linear, hierarchical-trie, tuple-space and HiCuts classifiers."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packet_classifier import (HiCutsClassifier, HierarchicalTrieClassifier,  # noqa: E402
                               LinearClassifier, TupleSpaceClassifier,
                               find_conflicts, find_shadowed, format_ip,
                               generate_packets, generate_rules, parse_ip,
                               parse_rule, Packet)


def main() -> None:
    rules = [parse_rule(t, i) for i, t in enumerate([
        "allow tcp 10.0.0.0/8 * 192.168.1.0/24 80-443",
        "deny tcp 10.1.0.0/16 * 192.168.1.0/24 80",
        "allow udp * * 8.8.8.8/32 53",
        "deny any * * * *"])]
    clf = HierarchicalTrieClassifier(rules)
    p = Packet(parse_ip("10.1.2.3"), parse_ip("192.168.1.9"), 4000, 80, 6)
    r = clf.lookup(p)
    print(f"{format_ip(p.src)} -> {format_ip(p.dst)}:{p.dport} => {r.action} (rule {r.priority})")
    print("shadowed rules:", [s.priority for s, _ in find_shadowed(rules)])

    print(f"\n{'rules':>6} {'classifier':<10} {'build s':>8} {'us/pkt':>8} {'extra':>28}")
    for n in (200, 1000, 3000):
        rs = generate_rules(n, seed=n)
        pk = generate_packets(rs, 1500, seed=n + 1)
        lin = LinearClassifier(rs)
        t0 = time.perf_counter()
        truth = [lin.lookup(q) for q in pk]
        base = (time.perf_counter() - t0) / len(pk) * 1e6
        print(f"{n:>6} {'linear':<10} {0:>8.3f} {base:>8.1f} {'':>28}")
        for name, cls in (("hier-trie", HierarchicalTrieClassifier),
                          ("tuple-sp", TupleSpaceClassifier),
                          ("hicuts", HiCutsClassifier)):
            t0 = time.perf_counter()
            c = cls(rs)
            build = time.perf_counter() - t0
            t0 = time.perf_counter()
            got = [c.lookup(q) for q in pk]
            us = (time.perf_counter() - t0) / len(pk) * 1e6
            assert got == truth, f"{name} disagrees with linear scan"
            if name == "hier-trie":
                extra = f"{c.memory_nodes()} trie nodes"
            elif name == "tuple-sp":
                extra = f"{c.n_tuples} tuples"
            else:
                extra = f"depth {c.depth()}, replication {c.replication():.1f}x"
            print(f"{n:>6} {name:<10} {build:>8.3f} {us:>8.1f} {extra:>28}")
    rs = generate_rules(1000, seed=1000)
    print(f"\n1000-rule set: {len(find_shadowed(rs))} shadowed, "
          f"{len(find_conflicts(rs))} conflicting pairs")


if __name__ == "__main__":
    main()
