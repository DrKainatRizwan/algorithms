import numpy as np

from power_analysis import (SBOX, cpa_attack, dpa_attack_multibit, encrypt_block,
                            expand_key, invert_key_schedule, key_rank, recover_key,
                            second_order_cpa, simulate_masked_traces,
                            simulate_traces, success_curve, traces_to_success)
from power_analysis.attacks import pearson_columns

KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")


def test_sbox_known_values():
    assert SBOX[0] == 0x63 and SBOX[1] == 0x7C and SBOX[0x53] == 0xED
    assert len(set(SBOX)) == 256


def test_fips197_appendix_b():
    pt = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    assert encrypt_block(KEY, pt).hex() == "3925841d02dc09fbdc118597196a0b32"


def test_fips197_appendix_c():
    key = bytes(range(16))
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    assert encrypt_block(key, pt).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"


def test_key_schedule_last_round():
    assert bytes(expand_key(KEY)[10]).hex() == "d014f9a8c9ee2589e13f0cc8b6630ca6"


def test_invert_key_schedule():
    assert invert_key_schedule(bytes(expand_key(KEY)[10])) == KEY


def test_bad_key_length():
    try:
        expand_key(b"short")
    except ValueError:
        return
    raise AssertionError


def test_pearson_matches_numpy():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(50, 3)); t = rng.normal(size=(50, 2))
    r = pearson_columns(h, t)
    assert np.isclose(r[1, 0], np.corrcoef(h[:, 1], t[:, 0])[0, 1])


def test_traces_deterministic():
    a = simulate_traces(KEY, 20, seed=3); b = simulate_traces(KEY, 20, seed=3)
    assert np.array_equal(a.traces, b.traces) and a.traces.shape == (20, 32)


def test_cpa_recovers_full_key():
    ts = simulate_traces(KEY, 400, noise_sigma=1.0, seed=5)
    assert recover_key(ts, cpa_attack) == KEY


def test_cpa_recovers_with_heavy_noise():
    ts = simulate_traces(KEY, 3000, noise_sigma=4.0, seed=6)
    assert key_rank(cpa_attack(ts, 0), KEY[0]) == 0


def test_dpa_recovers_byte():
    ts = simulate_traces(KEY, 1500, noise_sigma=1.0, seed=7)
    ok = sum(int(np.argmax(dpa_attack_multibit(ts, b))) == KEY[b] for b in range(4))
    assert ok >= 3


def test_cpa_fails_with_few_traces():
    ts = simulate_traces(KEY, 5, noise_sigma=3.0, seed=8)
    assert key_rank(cpa_attack(ts, 0), KEY[0]) > 0


def test_masking_defeats_first_order_cpa():
    ts = simulate_masked_traces(KEY, 3000, noise_sigma=0.5, seed=9)
    assert key_rank(cpa_attack(ts, 0), KEY[0]) > 5


def test_second_order_breaks_masking():
    ts = simulate_masked_traces(KEY, 6000, noise_sigma=0.5, seed=10)
    assert key_rank(second_order_cpa(ts, 2), KEY[2]) == 0


def test_success_curve_monotone_trend():
    c = success_curve(lambda n, s: simulate_traces(KEY, n, 2.0, s), cpa_attack, 0,
                      [5, 50, 400], n_experiments=10)
    assert c["success_rate"][-1] >= c["success_rate"][0]
    assert c["success_rate"][-1] == 1.0
    assert traces_to_success(c, 0.9) is not None
