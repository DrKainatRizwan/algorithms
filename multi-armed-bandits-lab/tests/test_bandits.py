import numpy as np
import pytest

from bandit_lab import (EXP3, AdversarialBandit, BernoulliBandit, EpsilonGreedy,
                        GaussianBandit, SlidingWindowUCB, Softmax, SwitchingBandit,
                        ThompsonBernoulli, ThompsonGaussian, UCB1, UCBTuned,
                        alternating_adversary, average_regret, bernoulli_kl,
                        best_arm_rate, exp3_regret_bound, lai_robbins_constant,
                        log_fit_slope, run, run_adversarial, ucb1_regret_bound)

PROBS = [0.2, 0.4, 0.8]


def env(seed=1):
    return BernoulliBandit(PROBS, seed=seed)


def test_bernoulli_validation_and_mean():
    with pytest.raises(ValueError):
        BernoulliBandit([0.5, 1.5])
    e = env()
    rs = [e.pull(2) for _ in range(4000)]
    assert abs(np.mean(rs) - 0.8) < 0.03


def test_pull_bounds():
    with pytest.raises(IndexError):
        env().pull(3)


def test_gaussian_rewards_clipped():
    e = GaussianBandit([0.5, 0.9], sigma=0.5, seed=0)
    rs = [e.pull(1) for _ in range(500)]
    assert min(rs) >= 0 and max(rs) <= 1


def test_switching_environment_changes_means():
    e = SwitchingBandit([[0.9, 0.1], [0.1, 0.9]], period=10, seed=0)
    assert e.best_mean() == 0.9 and e.means()[0] == 0.9
    for _ in range(10):
        e.pull(0)
    assert e.means()[1] == 0.9


def test_tried_each_arm_first():
    p = UCB1(3, seed=0)
    first = []
    for _ in range(3):
        a = p.select()
        first.append(a)
        p.update(a, 0.5)
    assert sorted(first) == [0, 1, 2]


@pytest.mark.parametrize("make", [
    lambda s: EpsilonGreedy(3, 0.1, seed=s),
    lambda s: EpsilonGreedy(3, 0.5, decay=True, seed=s),
    lambda s: UCB1(3, seed=s),
    lambda s: UCBTuned(3, seed=s),
    lambda s: ThompsonBernoulli(3, seed=s),
])
def test_policies_find_best_arm(make):
    res = run(make(0), env(), 3000)
    assert best_arm_rate(res, 2, last=500) > 0.8


def test_thompson_gaussian_on_gaussian_env():
    e = GaussianBandit([0.3, 0.5, 0.7], sigma=0.1, seed=3)
    res = run(ThompsonGaussian(3, sigma=0.1, seed=0), e, 1500)
    assert best_arm_rate(res, 2, last=300) > 0.8


def test_softmax_probabilities_normalised():
    p = Softmax(3, 0.1, seed=0)
    for a, r in [(0, 0.1), (1, 0.5), (2, 0.9)]:
        p.update(a, r)
    pr = p.probabilities()
    assert abs(pr.sum() - 1) < 1e-12 and pr.argmax() == 2


def test_regret_monotone_and_nonnegative():
    res = run(UCB1(3, seed=0), env(), 500)
    assert np.all(np.diff(res.regret) >= -1e-12)
    assert res.regret[0] >= 0


def test_thompson_beats_uniform_random_regret():
    class Uniform(EpsilonGreedy):
        def select(self):
            return int(self.rng.integers(3))
    ts = run(ThompsonBernoulli(3, seed=0), env(), 2000).final_regret
    un = run(Uniform(3, seed=0), env(), 2000).final_regret
    assert ts < 0.2 * un


def test_ucb_regret_sublinear_and_below_bound():
    curve = average_regret(lambda s: UCB1(3, seed=s), lambda s: env(s), 3000, 5)
    assert curve[-1] < ucb1_regret_bound(PROBS, 3000)
    assert curve[-1] - curve[1499] < curve[1499]  # growth slows down


def test_fixed_epsilon_has_linear_regret():
    curve = average_regret(lambda s: EpsilonGreedy(3, 0.2, seed=s), lambda s: env(s), 3000, 4)
    # roughly constant slope: second half adds about as much as the first
    assert (curve[-1] - curve[1499]) > 0.7 * curve[1499]


def test_kl_and_lai_robbins():
    assert bernoulli_kl(0.3, 0.3) == pytest.approx(0.0, abs=1e-9)
    assert bernoulli_kl(0.2, 0.8) > 0
    assert lai_robbins_constant([0.5, 0.5]) == 0
    assert lai_robbins_constant(PROBS) > 0


def test_log_fit_slope_recovers_log_curve():
    t = np.arange(1, 2001)
    assert log_fit_slope(3.0 * np.log(t)) == pytest.approx(3.0, rel=1e-6)


def test_exp3_probabilities_valid_and_learns():
    p = EXP3(2, horizon=2000, seed=0)
    assert abs(p.probabilities().sum() - 1) < 1e-12
    table = np.tile([0.2, 0.8], (2000, 1))
    res = run_adversarial(p, AdversarialBandit(table), 2000)
    assert best_arm_rate(res, 1, last=500) > 0.7
    assert res.final_regret < exp3_regret_bound(2, 2000)


def test_exp3_regret_within_bound_on_alternating_adversary():
    table = alternating_adversary(3000)
    r_exp = run_adversarial(EXP3(2, horizon=3000, seed=0), AdversarialBandit(table), 3000)
    assert r_exp.final_regret < exp3_regret_bound(2, 3000)


def test_adversarial_exhaustion_and_validation():
    with pytest.raises(ValueError):
        AdversarialBandit(np.full((5, 2), 2.0))
    e = AdversarialBandit(np.full((2, 2), 0.5))
    e.pull(0); e.pull(0)
    with pytest.raises(RuntimeError):
        e.pull(0)


def test_sliding_window_adapts_better_than_ucb_on_switching():
    sched = [[0.8, 0.5], [0.2, 0.5]]
    mk = lambda s: SwitchingBandit(sched, period=3000, seed=s)
    sw = average_regret(lambda s: SlidingWindowUCB(2, window=400, c=0.5, seed=s), mk, 6000, 3)[-1]
    st = average_regret(lambda s: UCB1(2, seed=s), mk, 6000, 3)[-1]
    assert sw < st


def test_determinism_with_seed():
    a = run(UCB1(3, seed=5), env(9), 300)
    b = run(UCB1(3, seed=5), env(9), 300)
    assert np.array_equal(a.arms, b.arms)
