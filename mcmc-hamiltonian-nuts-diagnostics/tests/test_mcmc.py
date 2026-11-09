import numpy as np
import pytest

from mcmc_samplers import (
    AdaptiveMetropolis, Banana, BayesianLogisticRegression, CorrelatedGaussian,
    GaussianMixture, HMC, MALA, NUTS, NealsFunnel, RandomWalkMetropolis,
    autocorrelation, effective_sample_size, leapfrog, split_rhat, summarize,
)
from mcmc_samplers.hmc import DualAveraging, hamiltonian


def targets():
    return [
        CorrelatedGaussian(np.array([[2.0, 0.8], [0.8, 1.0]])),
        Banana(),
        GaussianMixture(np.array([[-2.0, 0.0], [2.0, 1.0], [0.0, 3.0]])),
        NealsFunnel(4),
        BayesianLogisticRegression.synthetic(50, 3, seed=1)[0],
    ]


@pytest.mark.parametrize("t", targets())
def test_gradients_match_finite_differences(t):
    x = np.random.default_rng(0).normal(size=t.dim) * 0.7
    assert np.allclose(t.grad_log_prob(x), t.numerical_grad(x), atol=1e-4)


def test_leapfrog_is_reversible():
    t = Banana()
    q, p = np.array([0.5, -0.3]), np.array([0.2, 0.9])
    q1, p1 = leapfrog(t, q, p, 0.05, 30)
    q2, p2 = leapfrog(t, q1, -p1, 0.05, 30)
    assert np.allclose(q2, q, atol=1e-9) and np.allclose(-p2, p, atol=1e-9)


def test_leapfrog_energy_error_scales_quadratically():
    t = CorrelatedGaussian(np.eye(2))
    q, p = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    errs = []
    for eps in (0.2, 0.1):
        q1, p1 = leapfrog(t, q, p, eps, int(round(1.0 / eps)))
        errs.append(abs(hamiltonian(t, q1, p1) - hamiltonian(t, q, p)))
    assert errs[0] / errs[1] > 3.0


def test_rwm_recovers_gaussian_moments():
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    s = RandomWalkMetropolis(CorrelatedGaussian(cov), seed=1).sample(30000, np.zeros(2), burn=1000)
    assert np.allclose(s.mean(0), 0, atol=0.15)
    assert np.allclose(np.cov(s.T), cov, atol=0.25)


def test_adaptive_metropolis_learns_covariance():
    cov = np.array([[4.0, 1.8], [1.8, 1.0]])
    am = AdaptiveMetropolis(CorrelatedGaussian(cov), seed=2)
    s = am.sample(8000, np.zeros(2), burn=500)
    assert np.allclose(am.cov_, cov, atol=0.8)
    assert 0.1 < am.acceptance_rate < 0.6
    assert np.allclose(s.mean(0), 0, atol=0.4)


def test_mala_moments_and_acceptance():
    m = MALA(CorrelatedGaussian(np.eye(3)), step=0.8, seed=3)
    s = m.sample(8000, np.ones(3), burn=500)
    assert np.allclose(s.mean(0), 0, atol=0.15)
    assert np.allclose(s.var(0), 1, atol=0.25)
    assert m.acceptance_rate > 0.5


def test_hmc_gaussian_moments_and_adaptation():
    t = CorrelatedGaussian.ill_conditioned(5, 20.0, seed=4)
    h = HMC(t, n_leapfrog=15, jitter=0.3, seed=5)
    s = h.sample(3000, np.zeros(5), warmup=300)
    assert np.allclose(s.mean(0), 0, atol=0.3)
    assert abs(np.trace(np.cov(s.T)) / np.trace(t.cov) - 1) < 0.2
    assert 0.6 < h.acceptance_rate < 0.98


def test_nuts_gaussian_moments():
    t = CorrelatedGaussian.ill_conditioned(4, 30.0, seed=6)
    n = NUTS(t, seed=7)
    s = n.sample(2000, np.zeros(4), warmup=300)
    assert abs(np.trace(np.cov(s.T)) / np.trace(t.cov) - 1) < 0.2
    assert n.divergences == 0
    assert max(n.depths) <= 10


def test_nuts_finds_banana_mean():
    # E[x2] = -b (E[x1^2] - a^2) = -b (a^2 - a^2) = 0 and E[x1] = 0.
    n = NUTS(Banana(), seed=8)
    s = n.sample(6000, np.array([1.0, -1.0]), warmup=500)
    assert abs(s[:, 0].mean()) < 0.35
    assert abs(s[:, 1].mean()) < 1.0
    # the banana is curved: x2 is strongly (negatively) tied to x1^2
    assert np.corrcoef(s[:, 0] ** 2, s[:, 1])[0, 1] < -0.5
    assert abs(s[:, 0].var() - 4.0) < 1.0


def test_nuts_bimodal_mixture_visits_both_modes():
    t = GaussianMixture(np.array([[-2.0, 0.0], [2.0, 0.0]]), sigma=1.0)
    s = NUTS(t, seed=9).sample(3000, np.array([-2.0, 0.0]), warmup=300)
    frac = (s[:, 0] > 0).mean()
    assert 0.25 < frac < 0.75


def test_logistic_posterior_near_truth():
    t, beta = BayesianLogisticRegression.synthetic(400, 3, seed=10)
    s = NUTS(t, seed=11).sample(1000, np.zeros(3), warmup=200)
    assert np.allclose(s.mean(0), beta, atol=0.5)


def test_dual_averaging_converges_to_target_acceptance():
    da = DualAveraging(1.0, 0.8)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        eps = da.eps
        da.update(float(np.clip(1.2 - 0.7 * eps + 0.02 * rng.normal(), 0, 1)))
    assert abs((1.2 - 0.7 * da.final) - 0.8) < 0.05


def test_autocorrelation_of_ar1():
    rng = np.random.default_rng(0)
    x = np.zeros(20000)
    for i in range(1, x.size):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    ac = autocorrelation(x)
    assert ac[0] == pytest.approx(1.0)
    assert ac[1] == pytest.approx(0.7, abs=0.03)
    assert ac[2] == pytest.approx(0.49, abs=0.04)


def test_ess_iid_and_ar1():
    rng = np.random.default_rng(1)
    iid = rng.normal(size=(4, 2000))
    assert effective_sample_size(iid) == pytest.approx(8000, rel=0.25)
    ar = np.zeros((4, 4000))
    for i in range(1, 4000):
        ar[:, i] = 0.9 * ar[:, i - 1] + rng.normal(size=4)
    # theoretical ESS ratio (1-rho)/(1+rho) ~ 0.0526
    ratio = effective_sample_size(ar) / ar.size
    assert 0.03 < ratio < 0.09


def test_split_rhat_detects_non_convergence():
    rng = np.random.default_rng(2)
    good = rng.normal(size=(4, 1000))
    bad = good + np.array([[0.0], [0.0], [3.0], [3.0]])
    assert split_rhat(good) < 1.02
    assert split_rhat(bad) > 1.5


def test_summarize_shapes_and_determinism():
    t = CorrelatedGaussian(np.eye(2))
    runs = [np.stack([NUTS(t, seed=k).sample(300, np.zeros(2), warmup=100) for k in (1, 2)])
            for _ in range(2)]
    assert np.array_equal(runs[0], runs[1])
    r = summarize(runs[0])
    assert r["mean"].shape == (2,) and np.all(r["rhat"] < 1.1)
