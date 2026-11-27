import random

import pytest

from private_retrieval import (SchnorrGroup, bloom_psi, dh_psi, dh_psi_cardinality,
                               dictionary_attack, naive_hash_psi)
from private_retrieval.numtheory import is_probable_prime, modinv, random_safe_prime
from private_retrieval.psi import BloomFilter


@pytest.fixture(scope="module")
def group():
    return SchnorrGroup.generate(64, seed=1)


def sets(seed=0, n=40, overlap=15):
    r = random.Random(seed)
    common = [f"c{r.random()}" for _ in range(overlap)]
    a = common + [f"a{i}" for i in range(n - overlap)]
    b = common + [f"b{i}" for i in range(n - overlap + 7)]
    return a, b, {x.encode() for x in common}


def test_primality():
    assert [n for n in range(2, 40) if is_probable_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    assert not is_probable_prime(561)  # Carmichael number
    assert is_probable_prime(2 ** 127 - 1)


def test_safe_prime_and_group(group):
    assert is_probable_prime(group.p) and is_probable_prime(group.q)
    assert group.p == 2 * group.q + 1
    with pytest.raises(ValueError):
        SchnorrGroup(13)  # (13-1)/2 = 6 is not prime
    g = group.hash_to_group(b"x")
    assert group.contains(g) and group.exp(g, group.q) == 1
    assert modinv(3, 11) * 3 % 11 == 1


def test_hash_to_group_deterministic(group):
    assert group.hash_to_group(b"abc") == group.hash_to_group(b"abc")
    assert group.hash_to_group(b"abc") != group.hash_to_group(b"abd")


def test_dh_psi_correct(group):
    a, b, common = sets()
    res = dh_psi(a, b, group, seed=3)
    assert res.intersection == common and res.cardinality == len(common)


def test_dh_psi_edge_cases(group):
    assert dh_psi([], ["x"], group).intersection == set()
    assert dh_psi(["x"], [], group).intersection == set()
    assert dh_psi(["x", "x", "y"], ["y", "y"], group).intersection == {b"y"}
    assert dh_psi(["p"], ["q"], group).cardinality == 0


def test_dh_psi_cardinality_hides_elements(group):
    a, b, common = sets(seed=2)
    res = dh_psi_cardinality(a, b, group, seed=5)
    assert res.cardinality == len(common) and res.intersection == set()


def test_dh_psi_rejects_non_group_element(group):
    from private_retrieval.psi import DHParty
    party = DHParty(group, ["a"], random.Random(0))
    with pytest.raises(ValueError):
        party.blind_foreign([group.p - 1 if not group.contains(group.p - 1) else 2 if not group.contains(2) else 3])


def test_blinded_messages_differ_across_sessions(group):
    from private_retrieval.psi import DHParty
    x = DHParty(group, ["same"], random.Random(1)).blind_own()
    y = DHParty(group, ["same"], random.Random(2)).blind_own()
    assert x != y


def test_naive_hash_leaks_but_dh_does_not(group):
    universe = [f"user{i}" for i in range(500)]
    server = universe[100:130]
    res = naive_hash_psi(["user105", "zzz"], server)
    assert res.intersection == {b"user105"}
    recovered = dictionary_attack(res.extra["hashes"], universe)
    assert recovered == {s.encode() for s in server}


def test_bloom_filter_no_false_negatives_and_fp_rate():
    bf = BloomFilter.for_capacity(500, 0.02)
    for i in range(500):
        bf.add(f"k{i}".encode())
    assert all(f"k{i}".encode() in bf for i in range(500))
    fp = sum(f"z{i}".encode() in bf for i in range(5000)) / 5000
    assert fp < 0.05
    assert abs(bf.expected_fp_rate(500) - 0.02) < 0.01


def test_bloom_psi_superset_of_truth():
    a, b, common = sets(seed=4, n=60, overlap=20)
    res = bloom_psi(a, b, fp_rate=0.01)
    assert common <= res.intersection
    assert len(res.intersection) <= len(common) + 3
