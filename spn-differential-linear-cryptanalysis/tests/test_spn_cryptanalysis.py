import numpy as np
import pytest

from spn_cryptanalysis import (SBOX_AFFINE, SBOX_HEYS, SBOX_PRESENT, SPN,
                               best_characteristic, best_trail, ddt,
                               differential_attack, lat, linear_attack,
                               piling_up, sbox_report, walsh_spectrum)
from spn_cryptanalysis import differential as D
from spn_cryptanalysis import linear as L
from spn_cryptanalysis import sbox as S
from spn_cryptanalysis import spn as C
from spn_cryptanalysis import analysis as A


def test_ddt_rows_sum_and_first_row():
    for sb in (SBOX_HEYS, SBOX_PRESENT):
        t = ddt(sb)
        assert (t.sum(axis=1) == 16).all()
        assert t[0, 0] == 16 and t[0, 1:].sum() == 0
        assert (t % 2 == 0).all()


def test_lat_matches_walsh_and_parseval():
    t = lat(SBOX_HEYS)
    assert (walsh_spectrum(SBOX_HEYS) == 2 * t).all()
    # Parseval: each column of squared Walsh values sums to 2^(2n)
    w = walsh_spectrum(SBOX_PRESENT)
    assert ((w[:, 1:] ** 2).sum(axis=0) == 256).all()


def test_sbox_metrics_known_values():
    assert S.differential_uniformity(SBOX_PRESENT) == 4
    assert S.linearity(SBOX_PRESENT) == 4
    assert S.differential_uniformity(SBOX_HEYS) == 8
    assert S.nonlinearity(SBOX_PRESENT) == 4
    assert S.differential_uniformity(SBOX_AFFINE) == 16


def test_invert_and_random_sbox():
    rng = np.random.default_rng(0)
    sb = S.random_sbox(rng)
    inv = S.invert(sb)
    assert all(inv[sb[x]] == x for x in range(16))
    with pytest.raises(ValueError):
        S.invert((0,) * 16)


def test_cipher_roundtrip_and_bijection():
    rng = np.random.default_rng(3)
    c = SPN.random(4, rng)
    allp = np.arange(1 << 16, dtype=np.uint16)
    ct = c.encrypt(allp)
    assert len(np.unique(ct)) == 1 << 16
    assert (c.decrypt(ct) == allp).all()
    assert c.decrypt(c.encrypt(0xBEEF)) == 0xBEEF


def test_permutation_involution_and_nibble_helpers():
    x = 0x1234
    assert C.permute(C.permute(x)) == x
    assert C.nibbles(x) == [1, 2, 3, 4]
    assert C.from_nibbles([1, 2, 3, 4]) == x
    assert C.set_nibble(x, 2, 0xF) == 0x12F4
    assert C.active_nibbles(0x0F00) == [1]


def test_piling_up_lemma_numeric():
    # two independent approximations with biases .25 and .125
    assert piling_up([0.25, 0.125]) == pytest.approx(2 * 0.25 * 0.125)


def test_best_characteristic_matches_published_value():
    ch = best_characteristic(SBOX_HEYS, 3)
    assert ch.probability == pytest.approx(27 / 1024)
    assert D.characteristic_probability(SBOX_HEYS, ch) == pytest.approx(ch.probability)
    assert len(ch.steps) == 3


def test_characteristic_probability_decreases_with_rounds():
    ps = [best_characteristic(SBOX_HEYS, r).probability for r in (1, 2, 3)]
    assert ps[0] > ps[1] > ps[2]


def test_empirical_probability_close_to_theory():
    rng = np.random.default_rng(5)
    ch = best_characteristic(SBOX_HEYS, 3)
    c = SPN.random(4, rng)
    emp = D.empirical_probability(c, ch, 100000, rng)
    assert abs(emp - ch.probability) < 0.012


def test_differential_attack_recovers_subkey():
    rng = np.random.default_rng(11)
    ch = best_characteristic(SBOX_HEYS, 3)
    c = SPN.random(4, rng)
    res = differential_attack(c, ch, 600, rng)
    assert res.best_guess == D.true_subkey(c, res.nibble_positions)
    assert res.best_guess == res.ranking()[0]
    assert res.pairs_filtered < res.pairs_used


def test_best_trail_bias_consistent_with_lat():
    tr = best_trail(SBOX_HEYS, 3)
    assert L.trail_bias(SBOX_HEYS, tr) == pytest.approx(tr.bias)
    assert tr.bias > 1 / 32


def test_empirical_linear_bias_matches():
    rng = np.random.default_rng(7)
    tr = best_trail(SBOX_HEYS, 3)
    # the bias is key dependent; the piling-up value matches on average
    emp = np.mean([L.empirical_bias(SPN.random(4, rng), tr, 50000, rng)
                   for _ in range(12)])
    assert abs(emp - tr.bias) < 0.012


def test_linear_attack_recovers_subkey():
    rng = np.random.default_rng(13)
    tr = best_trail(SBOX_HEYS, 3)
    c = SPN.random(4, rng)
    res = linear_attack(c, tr, 20000, rng)
    assert res.best_guess == L.true_subkey(c, res.nibble_positions)
    assert res.best_guess == res.ranking()[0]


def test_success_rates_increase_with_data():
    ch = best_characteristic(SBOX_HEYS, 3)
    lo = D.success_rate(SBOX_HEYS, 4, ch, 10, 15, seed=2)
    hi = D.success_rate(SBOX_HEYS, 4, ch, 500, 15, seed=2)
    assert hi > lo and hi >= 0.8


def test_present_sbox_is_stronger_than_heys():
    rh = sbox_report("heys", SBOX_HEYS)
    rp = sbox_report("present", SBOX_PRESENT)
    assert rp.best_diff_char < rh.best_diff_char
    assert rp.best_linear_bias < rh.best_linear_bias
    assert rp.bijective and rp.diff_uniformity == 4


def test_affine_sbox_has_deterministic_distinguishers():
    ch = best_characteristic(SBOX_AFFINE, 3)
    assert ch.probability == pytest.approx(1.0)
    tr = best_trail(SBOX_AFFINE, 3)
    assert tr.bias == pytest.approx(0.5)


def test_avalanche_near_half_for_enough_rounds():
    rng = np.random.default_rng(1)
    c = SPN.random(5, rng, SBOX_PRESENT)
    assert 0.4 < A.avalanche(c, 2000, rng) < 0.6
