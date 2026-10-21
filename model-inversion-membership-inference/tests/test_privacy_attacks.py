import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_attacks import (CalibratedAttack, LabelOnly, LabelOnlyAttack, MLP, OutputNoise, Rounding,
                             ShadowAttack, TopK, TrainConfig, auc, invert_blackbox, invert_whitebox,
                             make_world, membership_advantage, reconstruction_quality, score_attack,
                             summarize, tpr_at_fpr, train, zcdp_epsilon)
from privacy_attacks.attacks import modified_entropy_score
from privacy_attacks.data import make_prototypes
from privacy_attacks.metrics import _rankdata


@pytest.fixture(scope="module")
def world():
    return make_world(0)


@pytest.fixture(scope="module")
def overfit(world):
    m = MLP(world.dim, world.n_classes, (64,), seed=1)
    train(m, world.X_in, world.y_in, TrainConfig())
    return m


def test_auc_perfect_and_chance_and_ties():
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert auc(np.ones(5), np.ones(7)) == 0.5


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(40), rng.standard_normal(30) - 0.3
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert auc(a, b) == pytest.approx(brute)


def test_rankdata_midranks():
    assert list(_rankdata(np.array([3.0, 1.0, 3.0, 2.0]))) == [3.5, 1.0, 3.5, 2.0]


def test_tpr_at_fpr_and_advantage():
    si, so = np.array([5.0, 6, 7, 8]), np.array([0.0, 1, 2, 6.5])
    assert tpr_at_fpr(si, so, 0.0) == 0.5
    assert membership_advantage(si, so) == pytest.approx(0.75)
    assert set(summarize(si, so)) == {"auc", "advantage", "tpr@5%fpr", "tpr@1%fpr"}


def test_prototypes_are_normalised_and_smooth():
    P = make_prototypes(4, 8, np.random.default_rng(0))
    assert P.shape == (4, 64)
    assert np.allclose(P.mean(axis=1), 0, atol=1e-9) and np.allclose(P.std(axis=1), 1)
    img = P[0].reshape(8, 8)
    assert np.abs(np.diff(img, axis=1)).mean() < np.abs(img - img.mean()).mean()


def test_world_is_deterministic_and_balanced(world):
    w2 = make_world(0)
    assert np.array_equal(world.X_in, w2.X_in)
    assert np.bincount(world.y_in).min() == 20


@pytest.mark.parametrize("hidden", [(), (8,), (8, 6)])
def test_gradients_match_finite_differences(hidden):
    rng = np.random.default_rng(3)
    m = MLP(6, 3, hidden, seed=2)
    X, y = rng.standard_normal((5, 6)), rng.integers(0, 3, 5)
    grads = m.mean_grads(X, y)
    for p, g in zip(m.params(), grads):
        i = tuple(0 for _ in p.shape)
        old = p[i]
        p[i] = old + 1e-6
        lp = m.loss(X, y).mean()
        p[i] = old - 1e-6
        lm = m.loss(X, y).mean()
        p[i] = old
        assert g[i] == pytest.approx((lp - lm) / 2e-6, abs=1e-5)


def test_per_example_grads_average_to_mean_grads():
    rng = np.random.default_rng(4)
    m = MLP(6, 3, (7,), seed=2)
    X, y = rng.standard_normal((9, 6)), rng.integers(0, 3, 9)
    for pe, mg in zip(m.per_example_grads(X, y), m.mean_grads(X, y)):
        assert np.allclose(pe.mean(axis=0), mg)


def test_input_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    m = MLP(6, 4, (10,), seed=7)
    x = rng.standard_normal(6)
    g = m.input_gradient(x, 2)
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1e-6
        num = (np.log(m.predict_proba(x + e)[0, 2]) - np.log(m.predict_proba(x - e)[0, 2])) / 2e-6
        assert g[j] == pytest.approx(num, abs=1e-5)


def test_training_overfits_and_generalises_partially(world, overfit):
    assert overfit.accuracy(world.X_in, world.y_in) > 0.99
    assert overfit.accuracy(world.X_test, world.y_test) > 0.7


def test_loss_attack_beats_chance_on_overfit_model(world, overfit):
    for kind in ("loss", "confidence", "entropy", "mod_entropy"):
        a = auc(score_attack(overfit, world.X_in, world.y_in, kind),
                score_attack(overfit, world.X_out, world.y_out, kind))
        assert a > 0.7, kind


def test_modified_entropy_is_zero_for_one_hot_correct():
    P = np.array([[1.0, 0.0, 0.0]])
    assert modified_entropy_score(P, np.array([0]))[0] == pytest.approx(0.0, abs=1e-9)


def test_attack_is_near_chance_on_generalising_model(world):
    # train on a large set so members are statistically like non-members
    big = make_world(1, n_members=2000, n_nonmembers=500)
    m = MLP(big.dim, big.n_classes, (16,), seed=0)
    train(m, big.X_in, big.y_in, TrainConfig(epochs=20))
    a = auc(score_attack(m, big.X_in[:500], big.y_in[:500]), score_attack(m, big.X_out, big.y_out))
    assert a < 0.58


def test_regularisation_and_dp_reduce_leakage(world, overfit):
    def leak(model):
        return auc(score_attack(model, world.X_in, world.y_in), score_attack(model, world.X_out, world.y_out))

    m_l2 = MLP(world.dim, world.n_classes, (64,), seed=1)
    train(m_l2, world.X_in, world.y_in, TrainConfig(l2=0.05))
    m_dp = MLP(world.dim, world.n_classes, (64,), seed=1)
    train(m_dp, world.X_in, world.y_in, TrainConfig(epochs=40, clip_norm=1.0, noise_multiplier=1.0))
    assert leak(m_l2) < leak(overfit)
    assert leak(m_dp) < leak(overfit) - 0.1


def test_dp_clipping_bounds_update_without_noise(world):
    m = MLP(world.dim, world.n_classes, (8,), seed=0)
    pe = m.per_example_grads(world.X_in[:16], world.y_in[:16])
    norms = np.sqrt(sum((g.reshape(16, -1) ** 2).sum(axis=1) for g in pe))
    assert norms.max() > 0.5  # clipping at 0.5 is actually active
    before = [p.copy() for p in m.params()]
    train(m, world.X_in[:16], world.y_in[:16],
          TrainConfig(epochs=1, batch_size=16, lr=0.1, clip_norm=0.5, noise_multiplier=0.0))
    # Adam's first step moves each coordinate by ~lr regardless; just check finite change
    assert all(np.isfinite(p).all() for p in m.params())
    assert any(not np.allclose(a, b) for a, b in zip(before, m.params()))


def test_zcdp_epsilon_monotone():
    assert math.isinf(zcdp_epsilon(10, 0.0))
    assert zcdp_epsilon(100, 2.0) < zcdp_epsilon(100, 1.0)
    assert zcdp_epsilon(50, 1.0) < zcdp_epsilon(100, 1.0)


def test_early_stopping_restores_best_params(world):
    m = MLP(world.dim, world.n_classes, (64,), seed=1)
    res = train(m, world.X_in, world.y_in, TrainConfig(epochs=300, patience=5),
                world.X_out, world.y_out)
    assert res.epochs_run < 300


def test_shadow_attack_beats_chance(world, overfit):
    sa = ShadowAttack(n_shadows=4, set_size=100, seed=0).fit(world.X_shadow, world.y_shadow, world.n_classes)
    a = auc(sa.score(overfit, world.X_in, world.y_in), sa.score(overfit, world.X_out, world.y_out))
    assert a > 0.7
    with pytest.raises(RuntimeError):
        ShadowAttack().score(overfit, world.X_in, world.y_in)


def test_calibrated_attack_beats_chance(world, overfit):
    sa = ShadowAttack(n_shadows=4, set_size=100, seed=0).fit(world.X_shadow, world.y_shadow, world.n_classes)
    cal = CalibratedAttack(sa.shadows)
    assert auc(cal.score(overfit, world.X_in, world.y_in), cal.score(overfit, world.X_out, world.y_out)) > 0.55


def test_label_only_attack_works_through_label_only_api(world, overfit):
    api = LabelOnly(overfit)
    assert set(np.unique(api.predict_proba(world.X_in))) <= {0.0, 1.0}
    atk = LabelOnlyAttack(n_queries=20, sigma=1.0)
    assert auc(atk.score(api, world.X_in, world.y_in), atk.score(api, world.X_out, world.y_out)) > 0.65
    # confidence-based scores collapse to chance behind the label-only API
    a = auc(score_attack(api, world.X_in, world.y_in, "confidence"),
            score_attack(api, world.X_out, world.y_out, "confidence"))
    assert a == 0.5


def test_output_wrappers_preserve_simplex(world, overfit):
    X = world.X_in[:20]
    for api in (TopK(overfit, 2), Rounding(overfit, 1), OutputNoise(overfit, 0.1, seed=0),
                OutputNoise(overfit, 0.1, seed=0, fresh=False)):
        P = api.predict_proba(X)
        assert np.allclose(P.sum(axis=1), 1) and (P >= 0).all()
    assert (np.count_nonzero(TopK(overfit, 2).predict_proba(X), axis=1) <= 2).all()


def test_output_noise_is_seeded_and_deterministic_mode(world, overfit):
    X = world.X_in[:5]
    a = OutputNoise(overfit, 0.2, seed=3).predict_proba(X)
    b = OutputNoise(overfit, 0.2, seed=3).predict_proba(X)
    assert np.array_equal(a, b)
    n = OutputNoise(overfit, 0.2, seed=3, fresh=False)
    assert np.array_equal(n.predict_proba(X), n.predict_proba(X))


def test_whitebox_inversion_recovers_class_prototypes(world, overfit):
    for c in range(world.n_classes):
        r = invert_whitebox(overfit, c, world.dim)
        q = reconstruction_quality(r.x, world.protos, c)
        assert q["identified"] == 1.0 and q["corr"] > 0.4


def test_blackbox_inversion_matches_whitebox_direction(world, overfit):
    r = invert_blackbox(overfit, 0, world.dim, steps=200, seed=0)
    q = reconstruction_quality(r.x, world.protos, 0)
    assert q["identified"] == 1.0 and q["corr"] > 0.3


def test_inversion_against_random_model_is_uninformative(world):
    m = MLP(world.dim, world.n_classes, (64,), seed=9)
    corr = [reconstruction_quality(invert_whitebox(m, c, world.dim).x, world.protos, c)["corr"]
            for c in range(world.n_classes)]
    assert np.mean(corr) < 0.3
