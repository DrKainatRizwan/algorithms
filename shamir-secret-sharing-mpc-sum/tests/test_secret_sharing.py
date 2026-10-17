import itertools
import random

import pytest

from secret_sharing import (BGW, PRIME_61, PRIME_127, GroupParams, Share,
                            additive_secure_sum, berlekamp_welch, deal,
                            is_probable_prime, make_shares, reconstruct,
                            reconstruct_bytes, refresh_shares,
                            robust_reconstruct, secure_equality_test,
                            secure_mean_variance, shamir_secure_sum,
                            share_bytes, verify_share)
from secret_sharing.field import (inv_mod, interpolate_poly, poly_divmod,
                                  poly_eval, next_prime)
from secret_sharing.shamir import consistent_with_secret, is_consistent
from secret_sharing.vss import combine_commitments, find_cheaters


def rng(seed=1):
    return random.Random(seed)


def test_primes_and_inverse():
    assert is_probable_prime(PRIME_61) and is_probable_prime(PRIME_127)
    assert not is_probable_prime(PRIME_61 * 3)
    assert next_prime(100) == 101
    for a in (1, 2, 12345, PRIME_61 - 1):
        assert a * inv_mod(a, PRIME_61) % PRIME_61 == 1
    with pytest.raises(ZeroDivisionError):
        inv_mod(0, 101)


def test_poly_interpolation_and_division():
    p = 10007
    coeffs = [5, 3, 0, 8]
    xs = [1, 2, 3, 4]
    ys = [poly_eval(coeffs, x, p) for x in xs]
    assert interpolate_poly(xs, ys, p) == coeffs
    q, r = poly_divmod([1, 3, 3, 1], [1, 1], p)  # (x+1)^3 / (x+1)
    assert q == [1, 2, 1] and r == [0]


@pytest.mark.parametrize("t,n", [(1, 1), (2, 3), (3, 5), (5, 9)])
def test_every_threshold_subset_recovers(t, n):
    secret = 123456789012345678901234567
    shares = make_shares(secret, t, n, rng=rng(t * 10 + n))
    for combo in itertools.combinations(shares, t):
        assert reconstruct(list(combo)) == secret


def test_below_threshold_reveals_nothing():
    secret = 4242
    shares = make_shares(secret, 4, 6, rng=rng(3))
    few = shares[:3]
    assert reconstruct(few) != secret
    # Any candidate secret is consistent with the 3 shares.
    assert all(consistent_with_secret(few, c) for c in (0, 1, secret, PRIME_127 - 1))


def test_invalid_parameters():
    with pytest.raises(ValueError):
        make_shares(1, 5, 3)
    with pytest.raises(ValueError):
        make_shares(PRIME_61, 2, 3, prime=PRIME_61)


def test_homomorphic_addition_and_scaling():
    r = rng(5)
    a = make_shares(1000, 3, 5, rng=r)
    b = make_shares(2345, 3, 5, rng=r)
    summed = [x + y for x, y in zip(a, b)]
    assert reconstruct(summed[:3]) == 3345
    assert reconstruct([s.scale(7) for s in a[1:4]]) == 7000
    assert reconstruct([s.add_const(5) for s in a[:3]]) == 1005
    with pytest.raises(ValueError):
        a[0] + b[1]


def test_consistency_check_detects_tampering():
    shares = make_shares(77, 3, 6, rng=rng(8))
    assert is_consistent(shares, 3)
    bad = shares[:]
    bad[4] = Share(bad[4].x, (bad[4].y + 1) % PRIME_127)
    assert not is_consistent(bad, 3)


def test_refresh_keeps_secret_but_changes_shares():
    shares = make_shares(999, 3, 5, rng=rng(2))
    new = refresh_shares(shares, 3, rng(9))
    assert reconstruct(new[:3]) == 999
    assert all(a.y != b.y for a, b in zip(shares, new))
    mixed = [shares[0], shares[1], new[2]]
    assert reconstruct(mixed) != 999


def test_bytes_roundtrip():
    msg = b"The quick brown fox jumps over the lazy dog \x00\xff"
    parties = share_bytes(msg, 3, 5, rng=rng(4))
    assert reconstruct_bytes([parties[4], parties[0], parties[2]]) == msg
    assert reconstruct_bytes(parties[:3]) == msg
    assert reconstruct_bytes(share_bytes(b"", 2, 3, rng=rng(1))[:2]) == b""


def test_feldman_vss_verifies_and_catches_cheaters():
    group = GroupParams.generate(48, seed=11)
    assert pow(group.g, group.q, group.p) == 1
    d = deal(31337, 3, 5, group, rng(6))
    assert all(verify_share(s, d.commitments, group) for s in d.shares)
    forged = list(d.shares)
    forged[2] = Share(3, (forged[2].y + 1) % group.q, group.q)
    assert find_cheaters(forged, d.commitments, group) == [3]
    assert reconstruct(list(d.shares[:3])) == 31337


def test_feldman_commitments_are_homomorphic():
    group = GroupParams.generate(48, seed=11)
    r = rng(1)
    d1, d2 = deal(10, 3, 4, group, r), deal(20, 3, 4, group, r)
    comm = combine_commitments(d1.commitments, d2.commitments, group)
    for s1, s2 in zip(d1.shares, d2.shares):
        assert verify_share(s1 + s2, comm, group)


def test_berlekamp_welch_corrects_errors():
    secret = 555
    shares = make_shares(secret, 3, 9, rng=rng(12))  # corrects up to 3 errors
    bad = list(shares)
    for i in (0, 4, 7):
        bad[i] = Share(bad[i].x, (bad[i].y + 17 + i) % PRIME_127)
    value, wrong = robust_reconstruct(bad, 3)
    assert value == secret and sorted(wrong) == [1, 5, 8]
    coeffs, none = berlekamp_welch(shares, 3)
    assert coeffs[0] == secret and none == []


def test_berlekamp_welch_fails_with_too_many_errors():
    shares = make_shares(5, 3, 7, rng=rng(2))  # corrects up to 2
    bad = list(shares)
    r = rng(99)
    for i in range(4):
        bad[i] = Share(bad[i].x, r.randrange(PRIME_127))
    with pytest.raises(ValueError):
        robust_reconstruct(bad, 3)


def test_additive_and_shamir_secure_sum():
    inputs = [12, 345, 6789, 10, 1]
    assert additive_secure_sum(inputs, seed=3) == sum(inputs)
    res = shamir_secure_sum(inputs, threshold=3, seed=3)
    assert res.total == sum(inputs) and res.contributors == [0, 1, 2, 3, 4]


def test_secure_sum_dropouts():
    inputs = [10, 20, 30, 40, 50, 60]
    res = shamir_secure_sum(inputs, 3, dropped_after=[0, 5], seed=1)
    assert res.total == sum(inputs)  # shares were already dealt
    res = shamir_secure_sum(inputs, 3, dropped_before=[1, 2], seed=1)
    assert res.total == 10 + 40 + 50 + 60 and res.contributors == [0, 3, 4, 5]
    with pytest.raises(ValueError):
        shamir_secure_sum(inputs, 5, dropped_after=[0, 1, 2], seed=1)


def test_bgw_arithmetic():
    eng = BGW(n=5, t=2, seed=7)
    a, b, c = eng.input(6), eng.input(7), eng.input(100)
    assert eng.open(eng.mul(a, b)) == 42
    expr = eng.add(eng.mul(eng.mul(a, b), c), eng.scale(a, 3))  # 6*7*100 + 18
    assert eng.open(expr) == 4218
    assert eng.open(eng.sub(a, b)) == PRIME_127 - 1
    assert eng.net.rounds == 3
    with pytest.raises(ValueError):
        BGW(n=4, t=2)


def test_bgw_product_degree_is_reduced():
    eng = BGW(n=5, t=2, seed=1)
    prod = eng.mul(eng.input(3), eng.input(9))
    assert is_consistent(prod, 3)  # still a degree-t sharing
    assert reconstruct(prod[2:5]) == 27


def test_mean_variance_matches_plain_statistics():
    vals = [4, 8, 15, 16, 23, 42]
    mean, var, net = secure_mean_variance(vals, t=2, seed=5)
    assert mean == pytest.approx(sum(vals) / 6)
    expected = sum((v - mean) ** 2 for v in vals) / 6
    assert var == pytest.approx(expected)
    assert net.messages > 0


def test_secure_equality():
    assert secure_equality_test(77, 77, seed=1)
    assert not secure_equality_test(77, 78, seed=1)
    assert not secure_equality_test(0, 1, seed=2)
