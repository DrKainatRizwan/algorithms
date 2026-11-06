import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from gradient_leakage import *
from gradient_leakage.model import softmax
from gradient_leakage.optimize import _tv_grad


def small(seed=0, d=16, h=8, k=3):
    rng = np.random.default_rng(seed)
    return MLP(d, h, k, rng), rng


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).standard_normal((5, 4)) * 50)
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()


def test_model_gradient_matches_finite_difference():
    m, rng = small()
    X, y = rng.random((4, 16)), np.array([0, 1, 2, 1])
    g = m.gradients(X, y)
    for name in ("W1", "b2"):
        arr = getattr(m, name)
        for idx in [(0,) * arr.ndim, (arr.shape[0] - 1,) + (0,) * (arr.ndim - 1)]:
            old = arr[idx]
            arr[idx] = old + 1e-6; lp = m.loss(X, y)
            arr[idx] = old - 1e-6; lm = m.loss(X, y)
            arr[idx] = old
            assert abs((lp - lm) / 2e-6 - g[name][idx]) < 1e-7


def test_matching_gradient_matches_finite_difference():
    m, rng = small(1)
    y = np.array([0, 2, 1])
    target = m.gradients(rng.random((3, 16)), y)
    X = rng.random((3, 16))
    _, g = matching_loss_and_grad(m, X, y, target)
    for idx in [(0, 0), (1, 5), (2, 15)]:
        e = np.zeros_like(X); e[idx] = 1e-6
        num = (matching_loss_and_grad(m, X + e, y, target)[0]
               - matching_loss_and_grad(m, X - e, y, target)[0]) / 2e-6
        assert num == pytest.approx(g[idx], abs=1e-8)


def test_matching_loss_zero_at_truth():
    m, rng = small(2)
    X, y = rng.random((2, 16)), np.array([1, 0])
    loss, g = matching_loss_and_grad(m, X, y, m.gradients(X, y))
    assert loss < 1e-20 and np.abs(g).max() < 1e-10


def test_analytic_reconstruction_exact_for_single_sample():
    m, rng = small(3)
    x = rng.random((1, 16)); y = np.array([2])
    g = m.gradients(x, y)
    assert np.allclose(analytic_reconstruct(g), x[0], atol=1e-9)


def test_label_recovery():
    m, rng = small(4)
    for lab in range(3):
        g = m.gradients(rng.random((1, 16)), np.array([lab]))
        assert recover_label(g) == lab


def test_analytic_fails_on_batches():
    m, rng = small(5)
    X, y = rng.random((4, 16)), np.array([0, 1, 2, 0])
    rec = analytic_reconstruct(m.gradients(X, y))
    assert min(mse(rec, x) for x in X) > 1e-3


def test_adam_minimises_quadratic():
    opt, x = Adam((3,), lr=0.1), np.array([3.0, -2.0, 1.0])
    for _ in range(500):
        x = opt.step(x, 2 * x)
    assert np.abs(x).max() < 1e-2


def test_optimisation_attack_recovers_single_sample():
    model, X, y, _ = build_victim(seed=0)
    r = attack_batch(model, X, y, batch=1, seed=1, iters=800, restarts=3)
    assert r["psnr"] > 25


def test_optimisation_attack_small_batch_beats_random():
    model, X, y, _ = build_victim(seed=0)
    r = attack_batch(model, X, y, batch=2, seed=1, iters=800, restarts=3)
    rng = np.random.default_rng(9)
    base = np.mean([psnr(X[0], rng.random(64)) for _ in range(20)])
    assert r["psnr"] > base + 5


def test_noise_defense_degrades_attack():
    model, X, y, _ = build_victim(seed=0)
    clean = attack_batch(model, X, y, 1, 1, iters=500)["psnr"]
    noisy = attack_batch(model, X, y, 1, 1, "noise", 1.0, iters=500)["psnr"]
    assert noisy < clean - 5


def test_clip_and_noise_clips_norm_without_noise():
    m, rng = small(6)
    g = m.gradients(rng.random((3, 16)), np.array([0, 1, 2]))
    out = clip_and_noise(g, clip=0.01, sigma=0.0, rng=rng)
    n = np.sqrt(sum((v ** 2).sum() for v in out.values()))
    assert n == pytest.approx(0.01, rel=1e-6)


def test_topk_keeps_requested_fraction():
    m, rng = small(7)
    g = m.gradients(rng.random((3, 16)), np.array([0, 1, 2]))
    out = top_k_sparsify(g, 0.1)
    total = sum(v.size for v in g.values())
    nz = sum(int((v != 0).sum()) for v in out.values())
    assert abs(nz - round(0.1 * total)) <= 1


def test_quantize_levels_and_error_bound():
    g = {"a": np.linspace(-1, 1, 101)}
    q = quantize(g, 5)
    assert len(np.unique(q["a"])) <= 5
    assert np.abs(q["a"] - g["a"]).max() <= 0.25 + 1e-12


def test_apply_defense_unknown_raises():
    with pytest.raises(ValueError):
        apply_defense({"a": np.ones(2)}, "bogus", 1.0, np.random.default_rng(0))


def test_metrics_and_matching():
    a = np.random.default_rng(0).random((3, 8))
    perm, m = match_reconstructions(a, a[[2, 0, 1]])
    assert m == pytest.approx(0) and list(perm) == [1, 2, 0]
    assert psnr(a[0], a[0]) == 100.0 and ssim_global(a[0], a[0]) == pytest.approx(1.0)
    assert psnr(a[0], a[1]) < 30


def test_dataset_deterministic_and_bounded():
    spec = DatasetSpec(n_samples=30)
    X1, y1 = make_dataset(spec, np.random.default_rng(0))
    X2, y2 = make_dataset(spec, np.random.default_rng(0))
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert X1.min() >= 0 and X1.max() <= 1 and X1.shape == (30, 64)


def test_training_reduces_loss():
    model, X, y, _ = build_victim(seed=0, pretrain_epochs=0)
    before = model.loss(X, y)
    train_sgd(model, X, y, 5, 0.1, 16, np.random.default_rng(0))
    assert model.loss(X, y) < before


def test_tv_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.random((1, 16))
    def tv(X):
        I = X.reshape(1, 4, 4)
        dx = np.diff(I, axis=2, append=I[:, :, -1:]); dy = np.diff(I, axis=1, append=I[:, -1:, :])
        return np.sqrt(dx ** 2 + dy ** 2 + 1e-6).sum()
    g = _tv_grad(X, 4)
    for i in (0, 5, 15):
        e = np.zeros_like(X); e[0, i] = 1e-6
        assert (tv(X + e) - tv(X - e)) / 2e-6 == pytest.approx(g[0, i], abs=1e-6)
