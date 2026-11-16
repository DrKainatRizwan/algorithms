import random
import pytest
from zk_sigma import *
from zk_sigma import analysis
from zk_sigma.groups import is_probable_prime


@pytest.fixture(scope="module")
def G():
    return standard_group(128)


def test_primality():
    assert is_probable_prime(2**61 - 1)
    assert not is_probable_prime(2**61 + 1)
    assert not is_probable_prime(561)  # Carmichael


def test_groups_are_valid():
    for bits in (64, 128):
        g = standard_group(bits)
        assert is_probable_prime(g.p) and is_probable_prime(g.q)
        assert g.p == 2 * g.q + 1
        assert g.is_member(g.g) and g.is_member(g.h) and g.g != g.h
        assert pow(g.g, g.q, g.p) == 1


def test_generate_group_deterministic():
    a, b = generate_group(48, seed=3), generate_group(48, seed=3)
    assert a == b and is_probable_prime(a.q)


def test_schnorr_completeness_and_wrong_statement(G):
    rng = random.Random(1)
    x = G.random_scalar(rng)
    rel = schnorr_relation(G, G.exp(G.g, x))
    pf = prove(rel, [x], rng)
    assert verify(rel, pf)
    other = schnorr_relation(G, G.exp(G.g, x + 1))
    assert not verify(other, pf)


def test_prover_rejects_bad_witness(G):
    rel = schnorr_relation(G, G.exp(G.g, 5))
    with pytest.raises(ValueError):
        prove(rel, [6], random.Random(0))


def test_tampered_proof_rejected(G):
    rng = random.Random(2)
    rel = schnorr_relation(G, G.exp(G.g, 77))
    pf = prove(rel, [77], rng)
    bad = Proof(pf.commitment, pf.challenge, ((pf.response[0] + 1) % G.q,))
    assert not verify(rel, bad)
    assert not verify(rel, pf, context=b"other")


def test_context_binding(G):
    rng = random.Random(3)
    rel = schnorr_relation(G, G.exp(G.g, 9))
    pf = prove(rel, [9], rng, context=b"session-1")
    assert verify(rel, pf, context=b"session-1")
    assert not verify(rel, pf, context=b"session-2")


def test_dleq(G):
    rng = random.Random(4)
    x = 12345
    g2 = G.random_element(rng)
    rel = dleq_relation(G, G.g, G.exp(G.g, x), g2, G.exp(g2, x))
    assert verify(rel, prove(rel, [x], rng))
    bad = dleq_relation(G, G.g, G.exp(G.g, x), g2, G.exp(g2, x + 1))
    with pytest.raises(ValueError):
        prove(bad, [x], rng)


def test_okamoto_representation(G):
    rng = random.Random(5)
    bases = [G.random_element(rng) for _ in range(3)]
    xs = [G.random_scalar(rng) for _ in range(3)]
    rel = representation_relation(G, bases, G.multiexp(bases, xs))
    assert verify(rel, prove(rel, xs, rng))


def test_and_relation_with_shared_witness(G):
    rng = random.Random(6)
    x, y = 31, 57
    r1 = schnorr_relation(G, G.exp(G.g, x))
    r2 = schnorr_relation(G, G.exp(G.h, x), G.h)
    both = and_relation([r1, r2], shared=[[0, 1]])
    assert both.n_witness == 1
    assert verify(both, prove(both, [x], rng))
    r3 = schnorr_relation(G, G.exp(G.h, y), G.h)
    mism = and_relation([r1, r3], shared=[[0, 1]])
    with pytest.raises(ValueError):
        prove(mism, [x], rng)


def test_extractor_recovers_witness(G):
    rng = random.Random(7)
    x = G.random_scalar(rng)
    rel = schnorr_relation(G, G.exp(G.g, x))
    pv = Prover(rel, [x], rng)
    com = pv.commit()
    k = pv._k[:]
    c1, c2 = 111, 222
    s1 = tuple((k[0] + c1 * x) % G.q for _ in [0])
    s2 = tuple((k[0] + c2 * x) % G.q for _ in [0])
    p1, p2 = Proof(com, c1, s1), Proof(com, c2, s2)
    assert verify_transcript(rel, com, c1, s1)
    assert extract(rel, p1, p2) == [x]
    with pytest.raises(ValueError):
        extract(rel, p1, p1)


def test_simulated_transcripts_verify(G):
    rng = random.Random(8)
    rel = schnorr_relation(G, G.random_element(rng))  # statement with unknown witness to us
    sim = simulate(rel, 424242, rng)
    assert verify_transcript(rel, sim.commitment, sim.challenge, sim.response)


def test_hvzk_exact_distribution_toy():
    g = toy_group()
    x = 7
    rel = schnorr_relation(g, g.exp(g.g, x))
    real, sim = analysis.transcript_distributions(rel, [x], challenge=5)
    assert analysis.total_variation(real, sim) == 0.0


def test_soundness_error_is_one_over_q():
    g = toy_group()
    rel = schnorr_relation(g, g.exp(g.g, 3))
    rate = analysis.cheating_prover_success_rate(rel, 20000, random.Random(9))
    assert abs(rate - 1 / g.q) < 0.02


def test_weak_fiat_shamir_forgery(G):
    rng = random.Random(10)
    rel = schnorr_relation(G, G.random_element(rng))
    forged = analysis.forge_weak_proof(rel, rng)
    assert analysis.weak_verify(rel, forged)
    assert analysis.forge_against_strong(rel, rng, 50) == 0


def test_pedersen_homomorphism_and_binding(G):
    ped = Pedersen(G)
    c1, c2 = ped.commit(10, 111), ped.commit(32, 222)
    assert ped.open(ped.add(c1, c2), 42, 333)
    assert ped.open(ped.scale(c1, 3), 30, 333)
    assert not ped.open(c1, 11, 111)


def test_opening_and_equality_proofs(G):
    rng = random.Random(11)
    ped = Pedersen(G)
    c1, c2, c3 = ped.commit(5, 10), ped.commit(5, 99), ped.commit(6, 99)
    assert ped.verify_opening(c1, ped.prove_opening(c1, 5, 10, rng))
    assert ped.verify_equal(c1, c2, ped.prove_equal(c1, c2, 10, 99, rng))
    with pytest.raises(ValueError):
        ped.prove_equal(c1, c3, 10, 99, rng)


def test_bit_proof(G):
    rng = random.Random(12)
    ped = Pedersen(G)
    for b in (0, 1):
        c = ped.commit(b, 777)
        pf = ped.prove_bit(c, b, 777, rng)
        assert ped.verify_bit(c, pf)
    c2 = ped.commit(2, 777)
    assert not ped.verify_bit(c2, pf)


def test_or_proof_rejects_tamper(G):
    rng = random.Random(13)
    r0 = schnorr_relation(G, G.random_element(rng))
    x = 99
    r1 = schnorr_relation(G, G.exp(G.g, x))
    pf = prove_or([r0, r1], 1, [x], rng)
    assert verify_or([r0, r1], pf)
    bad = OrProof(pf.commitments, ((pf.challenges[0] + 1) % G.q, pf.challenges[1]), pf.responses)
    assert not verify_or([r0, r1], bad)


def test_range_proof(G):
    rng = random.Random(14)
    rp = RangeProver(Pedersen(G), 8)
    for m in (0, 1, 200, 255):
        c, pf = rp.prove(m, G.random_scalar(rng), rng)
        assert rp.verify(c, pf)
    with pytest.raises(ValueError):
        rp.prove(256, 5, rng)


def test_range_proof_wrong_commitment(G):
    rng = random.Random(15)
    ped = Pedersen(G)
    rp = RangeProver(ped, 6)
    c, pf = rp.prove(20, 1234, rng)
    assert not rp.verify(ped.commit(21, 1234), pf)


def test_schnorr_signature_and_forgery(G):
    rng = random.Random(16)
    x, pub = keygen(G, rng)
    sig = sign(G, x, b"hello", rng)
    assert verify_signature(G, pub, b"hello", sig)
    assert not verify_signature(G, pub, b"hellp", sig)
    _, pub2 = keygen(G, rng)
    assert not verify_signature(G, pub2, b"hello", sig)


def test_nonce_reuse_leaks_key(G):
    rng = random.Random(17)
    x, pub = keygen(G, rng)
    s1 = sign(G, x, b"a", rng, nonce=987654321)
    s2 = sign(G, x, b"b", rng, nonce=987654321)
    assert recover_key_from_nonce_reuse(G, b"a", s1, b"b", s2, pub) == x


def test_batch_verification(G):
    rng = random.Random(18)
    items = []
    for i in range(10):
        x, pub = keygen(G, rng)
        m = f"msg{i}".encode()
        items.append((pub, m, sign(G, x, m, rng)))
    assert batch_verify(G, items, rng)
    pub, m, sg = items[3]
    items[3] = (pub, m, Signature(sg.r, (sg.s + 1) % G.q))
    assert not batch_verify(G, items, rng)


def test_transcript_domain_separation():
    a, b = Transcript("a"), Transcript("b")
    a.absorb_int("x", 5); b.absorb_int("x", 5)
    assert a.challenge(10**30) != b.challenge(10**30)
    c = Transcript("a"); c.absorb_ints("x", [1, 23])
    d = Transcript("a"); d.absorb_ints("x", [12, 3])
    assert c.challenge(2**100) != d.challenge(2**100)
