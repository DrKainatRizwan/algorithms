import numpy as np
import pytest

from policy_gradient import (A2C, Adam, CartPole, Corridor, MLP, Reinforce, VectorEnv,
                             discounted_returns, evaluate_policy, gae, log_softmax,
                             moving_average, normalize, softmax)
from policy_gradient.reinforce import policy_gradient_logits


def test_softmax_normalised_and_stable():
    p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isclose(p.sum(), 1.0) and np.all(np.isfinite(p))
    assert np.allclose(np.exp(log_softmax(np.array([[1.0, 2.0]]))), softmax(np.array([[1.0, 2.0]])))


def test_discounted_returns():
    g = discounted_returns(np.array([1.0, 1.0, 1.0]), 0.5)
    assert np.allclose(g, [1.75, 1.5, 1.0])
    g2 = discounted_returns(np.array([0.0]), 0.5, bootstrap=4.0)
    assert np.allclose(g2, [2.0])


def test_gae_lambda_one_matches_td_return_and_lambda_zero_td():
    r = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[0.5], [0.4], [0.3]])
    nv = np.array([[0.4], [0.3], [0.0]])
    d = np.array([[0.0], [0.0], [1.0]])
    adv1, tgt1 = gae(r, v, nv, d, 0.9, 1.0)
    assert np.allclose(tgt1[:, 0], discounted_returns(r[:, 0], 0.9))
    adv0, _ = gae(r, v, nv, d, 0.9, 0.0)
    assert np.allclose(adv0[:, 0], r[:, 0] + 0.9 * nv[:, 0] * (1 - d[:, 0]) - v[:, 0])


def test_gae_done_cuts_trace():
    r = np.ones((2, 1)); v = np.zeros((2, 1)); nv = np.ones((2, 1))
    adv, _ = gae(r, v, nv, np.array([[1.0], [0.0]]), 0.9, 0.95)
    assert np.isclose(adv[0, 0], 1.0)


def test_normalize():
    x = normalize(np.arange(10.0))
    assert abs(x.mean()) < 1e-9 and abs(x.std() - 1) < 1e-6


def test_mlp_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([3, 5, 2], rng)
    x = rng.normal(size=(4, 3)); w = rng.normal(size=(4, 2))
    out, acts = net.forward(x, cache=True)
    grads = net.backward(acts, w)
    loss = lambda: float((net.forward(x) * w).sum())
    for p, g in zip(net.params, grads):
        flat, gf = p.reshape(-1), g.reshape(-1)
        for i in range(0, flat.size, 3):
            old = flat[i]
            flat[i] = old + 1e-6; hi = loss()
            flat[i] = old - 1e-6; lo = loss()
            flat[i] = old
            assert abs((hi - lo) / 2e-6 - gf[i]) < 1e-5


@pytest.mark.parametrize("ent", [0.0, 0.3])
def test_policy_gradient_logit_gradient(ent):
    rng = np.random.default_rng(1)
    z = rng.normal(size=(6, 3)); a = rng.integers(0, 3, 6); w = rng.normal(size=6)
    _, g = policy_gradient_logits(z, a, w, ent)
    num = np.zeros_like(z)
    for idx in np.ndindex(*z.shape):
        zp = z.copy(); zp[idx] += 1e-6
        zm = z.copy(); zm[idx] -= 1e-6
        num[idx] = (policy_gradient_logits(zp, a, w, ent)[0] - policy_gradient_logits(zm, a, w, ent)[0]) / 2e-6
    assert np.allclose(g, num, atol=1e-6)


def test_adam_minimises_quadratic():
    p = [np.array([5.0, -3.0])]
    opt = Adam(p, lr=0.1)
    for _ in range(500):
        opt.step([2 * p[0]])
    assert np.abs(p[0]).max() < 1e-2


def test_adam_clipping_bounds_first_step():
    p = [np.zeros(2)]
    opt = Adam(p, lr=1.0, clip_norm=1.0)
    opt.step([np.array([1e6, 0.0])])
    assert np.isclose(abs(p[0][0]), 1.0, atol=1e-3)


def test_cartpole_deterministic_and_terminates():
    e1, e2 = CartPole(seed=3), CartPole(seed=3)
    assert np.allclose(e1.reset(), e2.reset())
    steps = 0
    while True:
        _, _, term, trunc = e1.step(1)
        steps += 1
        if term or trunc:
            break
    assert term and steps < 50  # constant push topples the pole quickly


def test_cartpole_truncates_at_limit():
    env = CartPole(max_steps=5)
    env.reset()
    for _ in range(5):
        _, _, term, trunc = env.step(0)
    assert trunc or term


def test_corridor_optimal_policy_return():
    env = Corridor(length=7)
    env.reset()
    total = 0.0
    while True:
        _, r, term, trunc = env.step(1)
        total += r
        if term or trunc:
            break
    assert np.isclose(total, env.optimal_return(1.0))


def test_vector_env_autoreset_and_final_obs():
    venv = VectorEnv(lambda s: CartPole(seed=s), 3, seed=0)
    venv.reset()
    for _ in range(40):
        obs, r, term, trunc, final = venv.step(np.ones(3, dtype=int))
    assert obs.shape == (3, 4) and len(venv.finished_returns) >= 3
    assert np.all(np.abs(final[:, 2]) < 0.5)


def test_reinforce_rejects_bad_baseline():
    with pytest.raises(ValueError):
        Reinforce(4, 2, baseline="bogus")


def test_reinforce_learns_corridor():
    env = Corridor(length=7)
    agent = Reinforce(7, 2, hidden=16, gamma=0.99, lr=0.02, baseline="mean", seed=0)
    agent.train(env, 150)
    assert agent.probs(env.reset())[0, 1] > 0.9
    assert evaluate_policy(agent, env, 3) > 0.9


def test_a2c_learns_corridor():
    venv = VectorEnv(lambda s: Corridor(length=7), 4)
    agent = A2C(7, 2, hidden=16, lr=0.02, seed=0)
    agent.train(venv, 120, n_steps=8)
    assert agent.probs(Corridor(length=7).reset())[0, 1] > 0.9


def test_a2c_improves_cartpole():
    venv = VectorEnv(lambda s: CartPole(seed=s), 8, seed=1)
    agent = A2C(4, 2, seed=0)
    rets = agent.train(venv, 250, n_steps=16)
    first = np.mean(rets[:20]); last = np.mean(rets[-20:])
    assert last > 2 * first


def test_training_is_reproducible():
    def run():
        env = Corridor(length=5)
        a = Reinforce(5, 2, hidden=8, seed=7)
        return a.train(env, 10), a.policy.params[0].copy()
    r1, w1 = run(); r2, w2 = run()
    assert r1 == r2 and np.array_equal(w1, w2)


def test_moving_average():
    assert np.allclose(moving_average([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
