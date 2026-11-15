import random

from crdt_toolkit import CausalBroadcaster, Network


def test_causal_delivery_buffers_out_of_order():
    a, b, c = (CausalBroadcaster(n) for n in "abc")
    m1 = a.broadcast("question")
    b.receive(m1)
    m2 = b.broadcast("answer")        # causally after m1
    assert c.receive(m2) == []         # held back
    assert [m.payload for m in c.receive(m1)] == ["question", "answer"]


def test_duplicates_ignored():
    a, b = CausalBroadcaster("a"), CausalBroadcaster("b")
    m = a.broadcast(1)
    assert len(b.receive(m)) == 1
    assert b.receive(m) == []
    assert len(b.log) == 1


def test_random_reordering_respects_causality():
    rng = random.Random(3)
    nodes = {n: CausalBroadcaster(n) for n in "abc"}
    inflight = []
    for step in range(40):
        s = rng.choice("abc")
        m = nodes[s].broadcast((s, step))
        inflight += [(d, m) for d in "abc" if d != s]
        rng.shuffle(inflight)
        for _ in range(rng.randint(0, 3)):
            if inflight:
                d, msg = inflight.pop()
                nodes[d].receive(msg)
    for d, msg in inflight:
        nodes[d].receive(msg)
    for n in nodes.values():
        assert not n.buffer
        seen: dict[str, int] = {}
        for m in n.log:        # delivered order must keep per-sender FIFO
            assert m.clock.get(m.sender) == seen.get(m.sender, 0) + 1
            seen[m.sender] = m.clock.get(m.sender)
    assert len({n.delivered for n in nodes.values()}) == 1


def test_network_partition_and_heal():
    net = Network(seed=1)
    net.partition([["a"], ["b"]])
    net.send("a", "b", 1)
    assert net.in_flight() == 0 and net.stats.dropped == 1
    net.heal()
    net.send("a", "b", 2)
    assert net.advance() == [("a", "b", 2)]


def test_network_is_deterministic():
    def trace():
        net = Network(seed=5, drop_prob=0.3, dup_prob=0.3, max_delay=4)
        for i in range(30):
            net.send("a", "b", i)
        out = []
        for _ in range(6):
            out.append([p for _, _, p in net.advance()])
        return out
    assert trace() == trace()
