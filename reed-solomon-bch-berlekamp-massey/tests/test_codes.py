import random

import pytest

from reed_solomon_bch import (BCH, BlockInterleaver, DecodingFailure, GF2m,
                              ReedSolomon, add_burst, add_symbol_errors,
                              berlekamp_massey, pgz_locator)
from reed_solomon_bch.experiments import (bch_error_sweep, rs_outcome_rates)


def test_field_axioms():
    f = GF2m(8)
    rng = random.Random(1)
    for _ in range(200):
        a, b, c = (rng.randrange(1, 256) for _ in range(3))
        assert f.mul(a, f.inv(a)) == 1
        assert f.mul(a, b) == f.mul(b, a)
        assert f.mul(a, b ^ c) == f.mul(a, b) ^ f.mul(a, c)
        assert f.div(f.mul(a, b), b) == a
    assert f.pow(2, 255) == 1


def test_non_primitive_polynomial_rejected():
    with pytest.raises(ValueError):
        GF2m(4, 0b10101)  # x^4+x^2+1 is reducible


def test_poly_divmod_roundtrip():
    f = GF2m(5)
    rng = random.Random(2)
    a = [rng.randrange(32) for _ in range(9)]
    b = [rng.randrange(32) for _ in range(4)] + [1]
    q, r = f.poly_divmod(a, b)
    assert f.trim(f.poly_add(f.poly_mul(q, b), r)) == f.trim(a)
    assert len(r) <= 4


def test_minimal_poly_has_root():
    f = GF2m(4)
    for s in (1, 3, 5, 7):
        mp = f.minimal_poly(s)
        coeffs = [(mp >> i) & 1 for i in range(mp.bit_length())]
        assert f.poly_eval(coeffs, f.alpha_pow(s)) == 0
    assert f.minimal_poly(1) == 0b10011


def test_rs_generator_roots_and_codeword():
    rs = ReedSolomon(15, 9, m=4)
    for i in range(rs.nsym):
        assert rs.field.poly_eval(rs.generator, rs.field.alpha_pow(1 + i)) == 0
    cw = rs.encode(list(range(9)))
    assert rs.is_codeword(cw)
    assert rs.extract(cw) == list(range(9))


def test_rs_clean_roundtrip():
    rs = ReedSolomon(255, 223)
    rng = random.Random(3)
    msg = [rng.randrange(256) for _ in range(223)]
    assert rs.decode(rs.encode(msg)) == msg


@pytest.mark.parametrize("method", ["bm", "pgz"])
def test_rs_corrects_up_to_t_errors(method):
    rs = ReedSolomon(31, 19, m=5, method=method)
    rng = random.Random(4)
    for e in range(rs.t + 1):
        for _ in range(10):
            msg = [rng.randrange(32) for _ in range(19)]
            noisy, pos = add_symbol_errors(rs.encode(msg), e, rng, 32)
            word, found = rs.decode_full(noisy)
            assert rs.extract(word) == msg
            assert found == pos


def test_rs_errors_and_erasures_budget():
    rs = ReedSolomon(63, 47, m=6)  # nsym = 16
    rng = random.Random(5)
    for errs, eras in [(0, 16), (4, 8), (7, 2), (3, 10), (8, 0)]:
        assert 2 * errs + eras <= rs.nsym
        for _ in range(8):
            msg = [rng.randrange(64) for _ in range(47)]
            cw = rs.encode(msg)
            pos = rng.sample(range(63), errs + eras)
            noisy = list(cw)
            for p in pos:
                noisy[p] ^= rng.randrange(1, 64)
            assert rs.decode(noisy, pos[:eras]) == msg


def test_rs_nonstandard_fcr():
    rs = ReedSolomon(31, 21, m=5, fcr=0)
    rng = random.Random(6)
    msg = [rng.randrange(32) for _ in range(21)]
    noisy, _ = add_symbol_errors(rs.encode(msg), 5, rng, 32)
    assert rs.decode(noisy) == msg


def test_rs_beyond_capacity_never_silently_returns_original_wrong_parity():
    rs = ReedSolomon(31, 21, m=5)
    stats = rs_outcome_rates(rs, rs.t + 3, 200, seed=7)
    assert abs(sum(stats.values()) - 1) < 1e-9
    # beyond capacity, most failures are detected rather than miscorrected
    assert stats["corrected"] == 0
    assert stats["detected"] > 0.5


def test_rs_too_many_erasures():
    rs = ReedSolomon(15, 11, m=4)
    cw = rs.encode([1] * 11)
    with pytest.raises(DecodingFailure):
        rs.decode(cw, erasures=[0, 1, 2, 3, 4])


def test_rs_bytes_interface():
    rs = ReedSolomon(64, 48)
    data = b"Reed-Solomon codes protect data on CDs, QR codes and deep-space links."
    stream = rs.encode_bytes(data)
    rng = random.Random(8)
    for b in range(0, len(stream), rs.n):
        block, _ = add_symbol_errors(stream[b:b + rs.n], 8, rng)
        stream[b:b + rs.n] = block
    assert rs.decode_bytes(stream, len(data)) == data


def test_bm_and_pgz_agree():
    rs = ReedSolomon(31, 21, m=5)
    f = rs.field
    rng = random.Random(9)
    for e in range(1, rs.t + 1):
        cw = rs.encode([rng.randrange(32) for _ in range(21)])
        noisy, _ = add_symbol_errors(cw, e, rng, 32)
        s = rs.syndromes(noisy)
        c1, l1 = berlekamp_massey(f, s)
        c2, l2 = pgz_locator(f, s, rs.t)
        assert l1 == l2 == e
        assert f.trim(c1) == f.trim(c2)


def test_bm_linear_complexity_of_geometric_sequence():
    f = GF2m(4)
    a = f.alpha_pow(3)
    seq = [f.pow(a, i) for i in range(8)]
    c, l = berlekamp_massey(f, seq)
    assert l == 1 and c == [1, a]


def test_bch_parameters_match_tables():
    assert (BCH(4, 1).n, BCH(4, 1).k) == (15, 11)
    assert (BCH(4, 2).n, BCH(4, 2).k) == (15, 7)
    assert (BCH(4, 3).n, BCH(4, 3).k) == (15, 5)
    assert BCH(5, 3).k == 16
    assert BCH(7, 4).k == 99


def test_bch_min_distance_at_least_designed():
    for t in (1, 2, 3):
        code = BCH(4, t)
        assert code.minimum_distance() >= code.designed_distance
    assert BCH(4, 2).minimum_distance() == 5


def test_bch_corrects_t_bit_errors_exhaustively_for_small_code():
    code = BCH(4, 2)
    msg = [1, 0, 1, 1, 0, 0, 1]
    cw = code.encode(msg)
    for i in range(code.n):
        for j in range(i, code.n):
            noisy = list(cw)
            noisy[i] ^= 1
            if j != i:
                noisy[j] ^= 1
            assert code.decode(noisy) == msg


def test_bch_larger_code_random_errors():
    code = BCH(8, 8)
    rng = random.Random(10)
    rows = bch_error_sweep(code, 8, 5, seed=10)
    assert all(rate == 1.0 for _, rate in rows)
    assert code.k == 191
    msg = [rng.randrange(2) for _ in range(code.k)]
    assert code.decode(code.encode(msg)) == msg


def test_bch_detects_some_overload():
    code = BCH(5, 2)
    rows = dict(bch_error_sweep(code, 6, 40, seed=11))
    assert rows[2] == 1.0
    assert rows[6] < 0.2


def test_interleaver_spreads_burst():
    depth, width = 8, 31
    rs = ReedSolomon(31, 21, m=5)  # t = 5
    il = BlockInterleaver(depth, width)
    rng = random.Random(12)
    msgs = [[rng.randrange(32) for _ in range(21)] for _ in range(depth)]
    flat = [s for m in msgs for s in rs.encode(m)]
    tx = il.interleave(flat)
    tx = add_burst(tx, 17, 36, rng, 32)  # 36 consecutive errors, 5 per word max
    rx = il.deinterleave(tx)
    for i, m in enumerate(msgs):
        assert rs.decode(rx[i * width:(i + 1) * width]) == m


def test_burst_without_interleaver_fails():
    rs = ReedSolomon(31, 21, m=5)
    rng = random.Random(13)
    cw = rs.encode([rng.randrange(32) for _ in range(21)])
    noisy = add_burst(cw, 3, 12, rng, 32)
    with pytest.raises(DecodingFailure):
        rs.decode(noisy)


def test_interleaver_roundtrip_and_validation():
    il = BlockInterleaver(3, 4)
    data = list(range(12))
    assert il.deinterleave(il.interleave(data)) == data
    with pytest.raises(ValueError):
        il.interleave([1, 2, 3])
