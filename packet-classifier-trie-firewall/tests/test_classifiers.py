import pytest

from packet_classifier import (HiCutsClassifier, HierarchicalTrieClassifier,
                               LinearClassifier, Packet, PrefixTrie,
                               TupleSpaceClassifier, find_conflicts,
                               find_shadowed, format_ip, generate_packets,
                               generate_rules, parse_ip, parse_prefix,
                               parse_rule)

RULES_TEXT = [
    "allow tcp 10.0.0.0/8 * 192.168.1.0/24 80-443",
    "deny tcp 10.1.0.0/16 * 192.168.1.0/24 80",
    "allow udp * * 8.8.8.8/32 53",
    "deny any * * * *",
]


def build():
    return [parse_rule(t, i) for i, t in enumerate(RULES_TEXT)]


def pkt(s, d, sp, dp, pr):
    return Packet(parse_ip(s), parse_ip(d), sp, dp, pr)


ALL = [HierarchicalTrieClassifier, TupleSpaceClassifier, HiCutsClassifier]


def test_ip_roundtrip():
    assert format_ip(parse_ip("192.168.10.7")) == "192.168.10.7"
    assert parse_ip("0.0.0.1") == 1


def test_parse_prefix_masks_host_bits():
    assert parse_prefix("10.1.2.3/8") == (parse_ip("10.0.0.0"), 8)
    assert parse_prefix("*") == (0, 0)


def test_bad_inputs():
    with pytest.raises(ValueError):
        parse_ip("1.2.3")
    with pytest.raises(ValueError):
        parse_rule("allow tcp *", 0)


def test_first_match_semantics():
    lin = LinearClassifier(build())
    assert lin.classify(pkt("10.1.2.3", "192.168.1.5", 999, 80, 6)) == "allow"
    assert lin.classify(pkt("11.1.2.3", "192.168.1.5", 999, 80, 6)) == "deny"
    assert lin.classify(pkt("1.1.1.1", "8.8.8.8", 5, 53, 17)) == "allow"


def test_prefix_trie_longest_match():
    t = PrefixTrie()
    t.insert(parse_ip("10.0.0.0"), 8, "a")
    t.insert(parse_ip("10.1.0.0"), 16, "b")
    assert t.longest_match(parse_ip("10.1.5.5")) == "b"
    assert t.longest_match(parse_ip("10.2.5.5")) == "a"
    assert t.longest_match(parse_ip("11.0.0.0")) is None
    assert list(t.matches(parse_ip("10.1.5.5"))) == ["a", "b"]


def test_trie_default_prefix_and_host_route():
    t = PrefixTrie()
    t.insert(0, 0, "any")
    t.insert(5, 32, "host")
    assert t.longest_match(5) == "host"
    assert t.longest_match(6) == "any"
    assert t.size == 2


@pytest.mark.parametrize("cls", ALL)
def test_handwritten_ruleset_matches_linear(cls):
    rules = build()
    lin, fast = LinearClassifier(rules), cls(rules)
    for p in [pkt("10.1.2.3", "192.168.1.5", 1, 80, 6),
              pkt("10.9.2.3", "192.168.1.5", 1, 443, 6),
              pkt("10.9.2.3", "192.168.1.5", 1, 444, 6),
              pkt("3.3.3.3", "8.8.8.8", 1, 53, 17),
              pkt("3.3.3.3", "8.8.8.8", 1, 53, 6)]:
        assert fast.lookup(p) == lin.lookup(p)


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_ruleset_equivalence(cls, seed):
    rules = generate_rules(300, seed=seed)
    pkts = generate_packets(rules, 600, seed=seed + 10)
    lin, fast = LinearClassifier(rules), cls(rules)
    for p in pkts:
        assert fast.lookup(p) == lin.lookup(p)


def test_no_default_rule_gives_none():
    rules = generate_rules(50, seed=4, default_rule=False)
    lin, hier = LinearClassifier(rules), HierarchicalTrieClassifier(rules)
    pkts = generate_packets([], 200, seed=5, hit_fraction=0.0)
    assert all(hier.lookup(p) == lin.lookup(p) for p in pkts)
    assert hier.classify(pkts[0], default="log") in {"allow", "deny", "log"}


def test_priority_beats_specificity():
    rules = [parse_rule("deny any * * * *", 0),
             parse_rule("allow tcp 10.0.0.1/32 * 10.0.0.2/32 22", 1)]
    p = pkt("10.0.0.1", "10.0.0.2", 1, 22, 6)
    for cls in [LinearClassifier] + ALL:
        assert cls(rules).classify(p) == "deny"


def test_tuple_space_pruning_reduces_probes():
    rules = generate_rules(200, seed=3)
    ts = TupleSpaceClassifier(rules)
    for p in generate_packets(rules, 100, seed=9):
        ts.lookup(p)
    assert ts.probes < 100 * ts.n_tuples


def test_hicuts_structure():
    rules = generate_rules(400, seed=6)
    hc = HiCutsClassifier(rules, binth=6)
    assert hc.depth() > 1
    assert hc.replication() >= 1.0
    assert hc.n_nodes > 1


def test_hicuts_identical_rules_terminates():
    rules = [parse_rule("allow tcp 1.2.3.4/32 * 5.6.7.8/32 80", i) for i in range(30)]
    hc = HiCutsClassifier(rules, binth=4)
    assert hc.lookup(pkt("1.2.3.4", "5.6.7.8", 1, 80, 6)).priority == 0


def test_shadowed_detection():
    rules = build() + [parse_rule("allow tcp 10.2.0.0/16 * 192.168.1.0/24 100", 4)]
    shadowed = find_shadowed(rules)
    assert [r.priority for r, _ in shadowed] == [1, 4]


def test_conflict_detection():
    a = parse_rule("allow tcp 10.0.0.0/8 * * 80", 0)
    b = parse_rule("deny tcp * * 20.0.0.0/8 80", 1)
    c = parse_rule("deny tcp 11.0.0.0/8 * * 80", 2)
    pairs = find_conflicts([a, b, c])
    assert (a, b) in pairs and (a, c) not in pairs


def test_hier_trie_memory_counts():
    h = HierarchicalTrieClassifier(generate_rules(100, seed=7))
    assert h.memory_nodes() >= h.src_trie.node_count()
    assert h.n_rules == 100


def test_generator_deterministic():
    assert generate_rules(40, seed=11) == generate_rules(40, seed=11)
    r = generate_rules(40, seed=11)
    assert generate_packets(r, 20, seed=2) == generate_packets(r, 20, seed=2)
