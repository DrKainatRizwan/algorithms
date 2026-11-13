import itertools
import random

import pytest

from garbled_circuits import SCHEMES, evaluate, garble, run_ot, run_two_party
from garbled_circuits.circuit import (Circuit, adder_circuit, equality_circuit,
                                      max_circuit, millionaires, multiplier_circuit)
from garbled_circuits.ot import OTReceiver, OTSender, _kdf
from garbled_circuits.primitives import bits_to_int, int_to_bits


def plain(c, x, y, wa, wb=None):
    wb = wb or wa
    return bits_to_int(c.evaluate_plain(int_to_bits(x, wa), int_to_bits(y, wb)))


def garbled_run(c, x, y, wa, scheme, seed=1):
    gc = garble(c, scheme, seed)
    labels = {w: gc.encode(w, b) for w, b in zip(c.inputs_a, int_to_bits(x, wa))}
    labels.update({w: gc.encode(w, b) for w, b in zip(c.inputs_b, int_to_bits(y, wa))})
    return bits_to_int(evaluate(gc, labels))


def test_bit_helpers_roundtrip():
    assert bits_to_int(int_to_bits(173, 8)) == 173
    with pytest.raises(ValueError):
        int_to_bits(16, 4)


def test_adder_plain_exhaustive():
    c = adder_circuit(3)
    for x, y in itertools.product(range(8), repeat=2):
        assert plain(c, x, y, 3) == x + y


def test_multiplier_and_comparator_plain():
    m, g = multiplier_circuit(3), millionaires(3)
    for x, y in itertools.product(range(8), repeat=2):
        assert plain(m, x, y, 3) == x * y
        assert plain(g, x, y, 3) == int(x > y)


def test_equality_and_max_plain():
    e, mx = equality_circuit(3), max_circuit(3)
    for x, y in itertools.product(range(8), repeat=2):
        assert plain(e, x, y, 3) == int(x == y)
        assert plain(mx, x, y, 3) == max(x, y)


def test_not_or_mux_gadgets():
    c = Circuit()
    a, b = c.input_a(), c.input_b()
    c.set_outputs([c.or_(a[0], b[0]), c.not_(a[0]), c.mux(a[0], b[0], c.one())])
    for x, y in itertools.product((0, 1), repeat=2):
        assert c.evaluate_plain([x], [y]) == [x | y, 1 - x, y if x else 1]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_garbled_matches_plain_exhaustive(scheme):
    circuits = [(adder_circuit(3), 3), (millionaires(3), 3), (equality_circuit(3), 3),
                (multiplier_circuit(3), 3), (max_circuit(3), 3)]
    for c, w in circuits:
        for x, y in itertools.product(range(8), repeat=2):
            assert garbled_run(c, x, y, w, scheme) == plain(c, x, y, w)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_garbled_not_gate(scheme):
    c = Circuit()
    a, b = c.input_a(), c.input_b()
    c.set_outputs([c.not_(c.and_(a[0], b[0]))])
    for x, y in itertools.product((0, 1), repeat=2):
        assert garbled_run(c, x, y, 1, scheme) == 1 - (x & y)


def test_wide_random_circuit_all_schemes():
    rng = random.Random(7)
    c = multiplier_circuit(8)
    for _ in range(5):
        x, y = rng.randrange(256), rng.randrange(256)
        for s in SCHEMES:
            assert garbled_run(c, x, y, 8, s, seed=rng.randrange(99)) == x * y


def test_table_sizes_follow_theory():
    c = adder_circuit(8)
    n_and, n_xor = c.count("AND"), c.count("XOR")
    assert garble(c, "half_gates").n_ciphertexts == 2 * n_and
    assert garble(c, "free_xor").n_ciphertexts == 4 * n_and
    assert garble(c, "classic").n_ciphertexts == 4 * (n_and + n_xor)
    sizes = [garble(c, s).size_bytes for s in ("half_gates", "free_xor", "classic")]
    assert sizes[0] < sizes[1] < sizes[2]


def test_and_counts_of_gadgets():
    assert millionaires(16).count("AND") == 16
    assert adder_circuit(16).count("AND") == 16
    assert multiplier_circuit(4).count("AND") == 16 + 4 * 8


def test_garbling_is_deterministic_and_seed_sensitive():
    c = millionaires(4)
    a, b, d = garble(c, "half_gates", 3), garble(c, "half_gates", 3), garble(c, "half_gates", 4)
    assert a.tables == b.tables and a.tables != d.tables


def test_input_label_colour_bits_differ():
    for s in SCHEMES:
        gc = garble(millionaires(4), s, 5)
        for l0, l1 in gc.input_labels.values():
            assert (l0 ^ l1) & 1 == 1


def test_wrong_label_does_not_decode_correctly_always():
    c = millionaires(8)
    gc = garble(c, "half_gates", 2)
    rng = random.Random(0)
    bad = 0
    for _ in range(20):
        labels = {w: rng.getrandbits(128) for w in c.inputs_a + c.inputs_b}
        x = [rng.randrange(2) for _ in range(16)]
        good = {w: gc.encode(w, b) for w, b in zip(c.inputs_a + c.inputs_b, x)}
        bad += evaluate(gc, labels) != evaluate(gc, good)
    assert bad > 0


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError):
        garble(millionaires(2), "nope")


def test_ot_receives_chosen_messages():
    rng = random.Random(4)
    pairs = [(rng.getrandbits(128), rng.getrandbits(128)) for _ in range(20)]
    choices = [rng.randrange(2) for _ in pairs]
    got = run_ot(pairs, choices, seed=9)
    assert got == [p[c] for p, c in zip(pairs, choices)]


def test_ot_other_message_stays_hidden():
    rng = random.Random(1)
    sender = OTSender(rng)
    recv = OTReceiver([0], rng)
    ct = sender.respond(recv.choose(sender.public), [(111, 222)])
    assert recv.recover(ct) == [111]
    # Receiver's only key opens slot 0; the slot-1 plaintext is not recovered.
    wrong_key = _kdf(pow(sender.public, recv._b[0], 2**255 - 19), 0)
    assert ct[0][1] ^ wrong_key != 222


def test_ot_receiver_message_hides_choice_distribution():
    rng = random.Random(2)
    sender = OTSender(rng)
    b0 = OTReceiver([0] * 50, rng).choose(sender.public)
    b1 = OTReceiver([1] * 50, rng).choose(sender.public)
    assert len(set(b0)) == 50 and len(set(b1)) == 50 and not set(b0) & set(b1)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_two_party_millionaires(scheme):
    c = millionaires(8)
    rng = random.Random(11)
    for _ in range(6):
        x, y = rng.randrange(256), rng.randrange(256)
        r = run_two_party(c, int_to_bits(x, 8), int_to_bits(y, 8), scheme, seed=x)
        assert r.outputs == [int(x > y)]


def test_two_party_cost_accounting():
    c = adder_circuit(8)
    hg = run_two_party(c, [1] * 8, [0] * 8, "half_gates")
    cl = run_two_party(c, [1] * 8, [0] * 8, "classic")
    assert hg.total_bytes < cl.total_bytes
    assert hg.n_ot == 8 and hg.n_and == 8
    assert bits_to_int(hg.outputs) == 255


def test_protocol_input_validation():
    with pytest.raises(ValueError):
        run_two_party(millionaires(4), [0], [0] * 4)
