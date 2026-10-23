import numpy as np
import pytest

from ppo_control import (MLP, GaussianPolicy, PendulumEnv, PointMassEnv, PPOAgent, PPOConfig,
                         ReachEnv2D, RolloutBuffer, RunningMeanStd, ValueNet, VecEnv,
                         compute_gae, evaluate, gradient_check)
from ppo_control.nets import Adam, clip_grad_norm


def test_mlp_gradient_check():
    rng = np.random.default_rng(0)
    net = MLP([3, 5, 2], rng)
    x, t = rng.standard_normal((6, 3)), rng.standard_normal((6, 2))

    def f():
        out = net.forward(x, store=True)
        return float(0.5 * ((out - t) ** 2).sum()), net.backward(out - t)

    assert gradient_check(f, net.params) < 1e-5


def test_policy_loss_gradient_check():
    rng = np.random.default_rng(1)
    pol = GaussianPolicy(4, 2, (8,), rng, init_log_std=-0.3)
    pol.net.W[-1] *= 50  # make outputs non-trivial
    obs, act = rng.standard_normal((16, 4)), rng.standard_normal((16, 2))
    old = pol.log_prob(obs, act) + rng.normal(0, 0.05, 16)
    adv = rng.standard_normal(16)

    def f():
        loss, g, _, _ = pol.ppo_loss_grads(obs, act, old, adv, 0.2, 0.01)
        return loss, g

    assert gradient_check(f, pol.params) < 1e-4


def test_value_loss_gradient_check_with_clipping():
    rng = np.random.default_rng(2)
    vn = ValueNet(3, (6,), rng)
    obs, ret = rng.standard_normal((12, 3)), rng.standard_normal(12) * 3
    old = vn.predict(obs) + rng.normal(0, 0.1, 12)

    def f():
        return vn.loss_grads(obs, ret, old, clip=0.2)

    assert gradient_check(f, vn.params) < 1e-4


def test_policy_gradient_zero_when_clipped():
    rng = np.random.default_rng(3)
    pol = GaussianPolicy(2, 1, (4,), rng)
    obs, act = rng.standard_normal((5, 2)), rng.standard_normal((5, 1))
    old = pol.log_prob(obs, act) - 2.0  # ratio = e^2 >> 1+eps
    adv = np.ones(5)                     # positive advantage -> clipped, no gradient
    loss, g, kl, cf = pol.ppo_loss_grads(obs, act, old, adv, 0.2, 0.0)
    assert cf == 1.0
    assert all(np.allclose(x, 0) for x in g)


def test_ratio_is_one_at_old_policy():
    rng = np.random.default_rng(4)
    pol = GaussianPolicy(3, 2, (8,), rng)
    obs = rng.standard_normal((10, 3))
    act, logp = pol.sample(obs, rng)
    _, _, kl, cf = pol.ppo_loss_grads(obs, act, logp, rng.standard_normal(10), 0.2, 0.0)
    assert kl == pytest.approx(0, abs=1e-12) and cf == 0.0


def test_gae_matches_naive_and_lambda_one_is_mc():
    rng = np.random.default_rng(5)
    T, n = 7, 3
    r, v = rng.standard_normal((T, n)), rng.standard_normal((T, n))
    d = (rng.random((T, n)) < 0.2).astype(float)
    lv = rng.standard_normal(n)
    adv, ret = compute_gae(r, v, d, lv, 0.9, 0.8)
    for e in range(n):
        for t in range(T):
            tot, coef = 0.0, 1.0
            for k in range(t, T):
                nv = lv[e] if k == T - 1 else v[k + 1, e]
                delta = r[k, e] + 0.9 * nv * (1 - d[k, e]) - v[k, e]
                tot += coef * delta
                coef *= 0.9 * 0.8 * (1 - d[k, e])
                if d[k, e]:
                    break
            assert adv[t, e] == pytest.approx(tot)
    # lambda = 1 with no dones: returns are discounted MC returns bootstrapped by last value
    adv1, ret1 = compute_gae(r, v, np.zeros((T, n)), lv, 0.9, 1.0)
    for t in range(T):
        mc = sum(0.9 ** (k - t) * r[k] for k in range(t, T)) + 0.9 ** (T - t) * lv
        assert np.allclose(ret1[t], mc)


def test_minibatches_cover_all_samples_and_normalise():
    buf = RolloutBuffer(6, 4, 2, 1)
    rng = np.random.default_rng(0)
    for i in range(6):
        buf.add(np.full((4, 2), i), np.zeros((4, 1)), np.zeros(4), rng.random(4), np.zeros(4), rng.random(4))
    buf.finish(np.zeros(4), 0.99, 0.95)
    batches = list(buf.minibatches(3, rng))
    assert sum(len(b["adv"]) for b in batches) == 24
    allobs = np.concatenate([b["obs"][:, 0] for b in batches])
    assert sorted(allobs) == sorted(np.repeat(np.arange(6), 4))
    adv = np.concatenate([b["adv"] for b in batches])
    assert abs(adv.mean()) < 1e-9 and adv.std() == pytest.approx(1, abs=1e-6)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3, 2, (500, 4))
    rms = RunningMeanStd((4,), eps=1e-12)
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(0), atol=1e-8)
    assert np.allclose(rms.var, data.var(0), atol=1e-8)


def test_adam_minimises_quadratic_and_clip_norm():
    p = [np.array([5.0, -3.0])]
    opt = Adam(p, lr=0.1)
    for _ in range(500):
        opt.step([2 * p[0]])
    assert np.abs(p[0]).max() < 1e-2
    g = [np.array([3.0, 4.0])]
    assert clip_grad_norm(g, 1.0) == pytest.approx(5.0)
    assert np.linalg.norm(g[0]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("cls", [PointMassEnv, ReachEnv2D, PendulumEnv])
def test_envs_deterministic_and_terminate(cls):
    def run(seed):
        e = cls(seed)
        o = e.reset()
        trace, done, n = [o], False, 0
        while not done:
            o, r, done, _ = e.step(np.full(e.act_dim, 0.3))
            trace.append(np.append(o, r))
            n += 1
        return n, np.concatenate(trace)
    n1, t1 = run(7)
    n2, t2 = run(7)
    assert n1 == n2 == cls.max_steps
    assert np.array_equal(t1, t2)
    assert np.isfinite(t1).all()


def test_env_clips_actions():
    e = PointMassEnv(0)
    e.reset()
    s = e.state.copy()
    e.step(np.array([100.0]))
    a = e.state.copy()
    e.state = s.copy(); e.t = 0
    e.step(np.array([1.0]))
    assert np.allclose(a, e.state)


def test_vecenv_autoreset_records_returns():
    v = VecEnv(lambda s: PointMassEnv(s), 3, seed=0)
    v.reset()
    for _ in range(PointMassEnv.max_steps):
        obs, r, d = v.step(np.zeros((3, 1)))
    assert d.all() and len(v.finished) == 3 and obs.shape == (3, 2)


def test_training_improves_pointmass():
    v = VecEnv(lambda s: PointMassEnv(s), 8, seed=0)
    ag = PPOAgent(v, PPOConfig(seed=0))
    base = evaluate(PointMassEnv(99), lambda o: np.zeros(1), 8)
    ag.train(30)
    trained = evaluate(PointMassEnv(99), ag.act, 8)
    assert trained > base * 0.2 and trained > -40  # base is ~ -270


def test_training_is_reproducible():
    def run():
        v = VecEnv(lambda s: PointMassEnv(s), 4, seed=3)
        ag = PPOAgent(v, PPOConfig(seed=5, n_envs=4, rollout_len=32))
        ag.train(3)
        return np.concatenate([p.ravel() for p in ag.policy.params])
    assert np.array_equal(run(), run())


def test_kl_early_stopping_limits_epochs():
    v = VecEnv(lambda s: ReachEnv2D(s), 4, seed=0)
    ag = PPOAgent(v, PPOConfig(seed=0, lr_pi=5e-2, target_kl=1e-6, epochs=10))
    stats = ag.update(ag.collect())
    assert stats["epochs_run"] < 10
