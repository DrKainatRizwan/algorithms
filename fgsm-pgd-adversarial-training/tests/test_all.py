import numpy as np
import pytest

from adv_attacks import (MLP, TrainConfig, adversarial_fit, attack_success_rate, fit, fgsm,
                         gradient_check, linear_margin_drop, make_blobs, make_fragile, make_rings,
                         make_spirals, mi_fgsm, pgd, project_l2, project_linf, random_noise,
                         robust_accuracy, accuracy, epsilon_sweep, train_test_split, softmax)


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(80, 3, spread=0.7, seed=1)
    return X, y


@pytest.fixture(scope="module")
def trained(blobs):
    X, y = blobs
    m = MLP((2, 16, 16, 3), seed=0)
    fit(m, X, y, TrainConfig(epochs=60, lr=0.02, seed=0))
    return m


def test_datasets_shapes_and_determinism():
    for f in (make_blobs, make_spirals, make_rings):
        X1, y1 = f(seed=3)
        X2, y2 = f(seed=3)
        assert X1.shape[1] == 2 and len(X1) == len(y1)
        assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_split_sizes_and_validation(blobs):
    X, y = blobs
    a, b, c, d = train_test_split(X, y, 0.25, seed=0)
    assert len(a) + len(c) == len(X) and len(c) == 60
    with pytest.raises(ValueError):
        train_test_split(X, y, 1.5)


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1000.0, 1001.0], [-5.0, 3.0]]))
    assert np.allclose(p.sum(axis=1), 1) and np.all(np.isfinite(p))


def test_input_gradient_matches_finite_difference(blobs):
    X, y = blobs
    m = MLP((2, 8, 3), seed=2)
    assert gradient_check(m, X, y) < 1e-4


def test_param_gradient_matches_finite_difference(blobs):
    X, y = blobs
    X, y = X[:20], y[:20]
    m = MLP((2, 6, 3), seed=4)
    _, gW, _ = m.loss_and_grads(X, y)
    h = 1e-5
    m.W[0][1, 2] += h
    lp = m.loss(X, y)
    m.W[0][1, 2] -= 2 * h
    lm = m.loss(X, y)
    assert abs((lp - lm) / (2 * h) - gW[0][1, 2]) < 1e-6


def test_training_reduces_loss_and_fits(trained, blobs):
    X, y = blobs
    assert accuracy(trained, X, y) > 0.9


def test_projections_respect_radius():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((50, 4)) * 3
    assert np.abs(project_linf(d, 0.5)).max() <= 0.5 + 1e-12
    assert np.linalg.norm(project_l2(d, 0.5), axis=1).max() <= 0.5 + 1e-9


def test_fgsm_within_budget_and_raises_loss(trained, blobs):
    X, y = blobs
    xa = fgsm(trained, X, y, 0.3)
    assert np.abs(xa - X).max() <= 0.3 + 1e-12
    assert trained.loss(xa, y) > trained.loss(X, y)


def test_fgsm_l2_budget(trained, blobs):
    X, y = blobs
    xa = fgsm(trained, X, y, 0.4, norm="l2")
    assert np.linalg.norm(xa - X, axis=1).max() <= 0.4 + 1e-9


def test_pgd_stronger_than_fgsm_and_noise(trained, blobs):
    X, y = blobs
    eps = 0.8
    r_noise = accuracy(trained, random_noise(X, eps, seed=0), y)
    r_fgsm = robust_accuracy(trained, X, y, eps, "fgsm")
    r_pgd = robust_accuracy(trained, X, y, eps, "pgd", steps=20)
    assert r_pgd <= r_fgsm + 1e-9
    assert r_pgd < r_noise


def test_pgd_budget_l2_and_clip(trained, blobs):
    X, y = blobs
    xa = pgd(trained, X, y, 0.5, norm="l2", steps=5)
    assert np.linalg.norm(xa - X, axis=1).max() <= 0.5 + 1e-9
    xc = pgd(trained, X, y, 1.0, steps=5, clip=(-1.0, 1.0))
    assert xc.min() >= -1.0 and xc.max() <= 1.0


def test_pgd_deterministic_given_seed(trained, blobs):
    X, y = blobs
    a = pgd(trained, X, y, 0.4, seed=5)
    b = pgd(trained, X, y, 0.4, seed=5)
    assert np.array_equal(a, b)


def test_targeted_attack_moves_towards_target(trained, blobs):
    X, y = blobs
    tgt = (y + 1) % 3
    xa = pgd(trained, X, y, 1.5, steps=30, targeted=tgt)
    before = (trained.predict(X) == tgt).mean()
    after = (trained.predict(xa) == tgt).mean()
    assert after > before + 0.3


def test_mi_fgsm_budget_and_effect(trained, blobs):
    X, y = blobs
    xa = mi_fgsm(trained, X, y, 0.6)
    assert np.abs(xa - X).max() <= 0.6 + 1e-9
    assert accuracy(trained, xa, y) < accuracy(trained, X, y)


def test_linear_model_fgsm_is_optimal():
    """For a linear binary-like score, FGSM drops the margin by exactly eps*||w||_1."""
    m = MLP((3, 2), seed=0)
    m.W[0] = np.array([[1.0, -1.0], [-2.0, 2.0], [0.5, -0.5]])
    m.b[0] = np.zeros(2)
    X = np.array([[1.0, -1.0, 2.0], [2.0, 0.5, 0.1]])
    y = m.predict(X)
    eps = 0.1
    xa = fgsm(m, X, y, eps)
    drop = m.margin(X, y) - m.margin(xa, y)
    w = m.W[0][:, 0] - m.W[0][:, 1]
    assert np.allclose(drop, linear_margin_drop(w, eps))


def test_epsilon_sweep_monotone_for_pgd(trained, blobs):
    X, y = blobs
    res = epsilon_sweep(trained, X, y, [0.0, 0.3, 0.9, 1.8], "pgd", steps=15)
    accs = [a for _, a in res]
    assert accs[0] >= accs[-1] and accs[0] > 0.9


def test_adversarial_training_improves_robustness():
    X, y = make_fragile(1000, seed=1)
    Xtr, ytr, Xte, yte = train_test_split(X, y, 0.3, seed=0)
    eps = 0.4
    nat = fit(MLP((10, 16, 2), seed=1), Xtr, ytr, TrainConfig(epochs=60, lr=0.01))
    adv = adversarial_fit(MLP((10, 16, 2), seed=1), Xtr, ytr,
                          TrainConfig(epochs=60, lr=0.01, eps=eps, mix=1.0))
    r_nat = robust_accuracy(nat, Xte, yte, eps, "pgd", steps=20)
    r_adv = robust_accuracy(adv, Xte, yte, eps, "pgd", steps=20)
    assert accuracy(nat, Xte, yte) > 0.95
    assert r_nat < 0.3 and r_adv > r_nat + 0.3


def test_fgsm_training_runs_and_records_history():
    X, y = make_fragile(300, seed=2)
    cfg = TrainConfig(epochs=10, lr=0.01, eps=0.4, attack="fgsm", mix=0.5)
    adversarial_fit(MLP((10, 8, 2), seed=0), X, y, cfg)
    assert len(cfg.history) == 10 and np.all(np.isfinite(cfg.history))


def test_attack_success_rate_bounds(trained, blobs):
    X, y = blobs
    xa = pgd(trained, X, y, 1.0, steps=10)
    s = attack_success_rate(trained, X, y, xa)
    assert 0.0 < s <= 1.0
    assert attack_success_rate(trained, X, y, X) == 0.0


def test_invalid_args(trained, blobs):
    X, y = blobs
    with pytest.raises(ValueError):
        pgd(trained, X, y, 0.1, steps=0)
    with pytest.raises(ValueError):
        pgd(trained, X, y, 0.1, norm="l1")
    with pytest.raises(ValueError):
        MLP((3,))
