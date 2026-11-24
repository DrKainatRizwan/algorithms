import numpy as np
import pytest

from online_learning import (AdaGrad, Ball, Box, ExponentiatedGradient, FTRL,
                             FollowThePerturbedLeader, FollowTheLeader, Hedge,
                             L1Ball, LinearLoss, LogisticLoss, OnlineGradientDescent,
                             OnlineNewtonStep, Simplex, SquaredLoss, aggregate,
                             alternating_experts, best_in_hindsight, expert_losses,
                             ftl_adversary, growth_exponent, hedge_eta,
                             hedge_regret_bound, logistic_stream, minimise_over_domain,
                             ogd_regret_bound, project_simplex, regression_stream,
                             regret, regret_curve_linear, run_online,
                             stochastic_linear, strongly_convex_regret_bound)


def test_simplex_projection_properties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.normal(size=7) * 3
        p = project_simplex(y)
        assert p.min() >= 0 and abs(p.sum() - 1) < 1e-12
        # optimality: no random simplex point is closer
        for _ in range(20):
            q = rng.dirichlet(np.ones(7))
            assert np.linalg.norm(p - y) <= np.linalg.norm(q - y) + 1e-12


def test_l1_ball_projection_inside_and_optimal():
    dom = L1Ball(5, 1.0)
    y = np.array([2.0, -1.0, 0.5, 0.0, 0.1])
    p = dom.project(y)
    assert abs(np.abs(p).sum() - 1.0) < 1e-12
    assert np.array_equal(dom.project(np.array([0.1, 0, 0, 0, 0.0])), [0.1, 0, 0, 0, 0])


@pytest.mark.parametrize("dom", [Ball(4, 2.0), Simplex(4), Box(4, -1, 3), L1Ball(4, 1.5)])
def test_linear_oracle_beats_random_points(dom):
    rng = np.random.default_rng(1)
    g = rng.normal(size=4)
    best = g @ dom.argmin_linear(g)
    for _ in range(200):
        x = dom.project(rng.normal(size=4) * 2)
        assert best <= g @ x + 1e-9


def test_ball_norm_projection_matches_euclid_for_identity_and_is_feasible():
    ball = Ball(3, 1.0)
    y = np.array([3.0, 0.0, 4.0])
    assert np.allclose(ball.project_norm(y, np.eye(3)), ball.project(y), atol=1e-9)
    A = np.diag([1.0, 5.0, 0.2])
    x = ball.project_norm(y, A)
    assert np.linalg.norm(x) <= 1.0 + 1e-9
    rng = np.random.default_rng(2)
    f = lambda v: (v - y) @ A @ (v - y)
    for _ in range(300):
        v = ball.project(rng.normal(size=3))
        assert f(x) <= f(v) + 1e-9


def test_losses_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    x = rng.normal(size=4)
    for loss in (SquaredLoss(rng.normal(size=4), 0.3, lam=0.2),
                 LogisticLoss(rng.normal(size=4), -1.0)):
        num = np.array([(loss.value(x + 1e-6 * e) - loss.value(x - 1e-6 * e)) / 2e-6
                        for e in np.eye(4)])
        assert np.allclose(num, loss.grad(x), atol=1e-6)


def test_aggregate_matches_sum_of_losses():
    rng = np.random.default_rng(4)
    x = rng.normal(size=3)
    for make in (lambda: LinearLoss(rng.normal(size=3)),
                 lambda: SquaredLoss(rng.normal(size=3), rng.normal(), lam=0.1),
                 lambda: LogisticLoss(rng.normal(size=3), 1.0)):
        ls = [make() for _ in range(6)]
        agg = aggregate(ls)
        assert agg.value(x) == pytest.approx(sum(l.value(x) for l in ls))
        assert np.allclose(agg.grad(x), sum(l.grad(x) for l in ls))


def test_fista_solves_ridge_regression_exactly_when_interior():
    losses = regression_stream(200, 3, seed=5, lam=0.01)
    x, _ = best_in_hindsight(losses, Ball(3, 10.0))
    A = np.array([l.a for l in losses])
    b = np.array([l.b for l in losses])
    exact = np.linalg.solve(A.T @ A + 200 * 0.01 * np.eye(3), A.T @ b)
    assert np.allclose(x, exact, atol=1e-5)


def test_fista_respects_constraint():
    losses = logistic_stream(300, 4, seed=6)
    dom = Ball(4, 0.5)
    x = minimise_over_domain(aggregate(losses), dom)
    assert np.linalg.norm(x) <= 0.5 + 1e-9
    assert np.linalg.norm(x) > 0.49  # constraint is active for separable-ish data


def test_ogd_iterates_stay_feasible_and_regret_within_bound():
    T, d = 2000, 5
    dom = Ball(d, 1.0)
    losses = stochastic_linear(T, d, seed=7)
    G = 2.0
    eta = dom.diameter() / (G * np.sqrt(T))
    run = run_online(OnlineGradientDescent(dom, eta, "const"), losses)
    assert all(dom.contains(x) for x in run.predictions)
    r = regret(run, losses, dom)
    assert 0 <= r <= ogd_regret_bound(dom.diameter(), G, T)


def test_ftl_has_linear_regret_but_ftrl_and_ogd_sublinear():
    dom = Box(1, -1, 1)
    out = {}
    for T in (200, 800):
        losses = ftl_adversary(T)
        out[T] = (regret(run_online(FollowTheLeader(dom), losses), losses, dom),
                  regret(run_online(FTRL(dom, 1 / np.sqrt(T)), losses), losses, dom),
                  regret(run_online(OnlineGradientDescent(dom, 1.0), losses), losses, dom))
    assert out[800][0] > 0.9 * 800 - 5              # FTL: ~ T
    assert out[800][1] < 0.5 * out[800][0]          # FTRL much smaller
    assert out[800][2] < 0.2 * out[800][0]
    assert growth_exponent([200, 800], [out[200][0], out[800][0]]) > 0.9


def test_ogd_regret_grows_like_sqrt_on_stochastic_linear():
    dom = Ball(4, 1.0)
    hs, rs = [250, 1000, 4000], []
    for T in hs:
        losses = stochastic_linear(T, 4, seed=8)
        eta = dom.diameter() / (2.0 * np.sqrt(T))
        rs.append(regret(run_online(OnlineGradientDescent(dom, eta, "const"), losses),
                         losses, dom))
    assert growth_exponent(hs, rs) < 0.85


def test_hedge_matches_exponentiated_gradient_and_meets_bound():
    T, n = 1000, 8
    losses = expert_losses(T, n, seed=9)
    eta = hedge_eta(n, T)
    r1 = run_online(Hedge(n, eta), losses)
    r2 = run_online(ExponentiatedGradient(n, eta), losses)
    assert np.allclose(r1.predictions, r2.predictions, atol=1e-8)
    reg = regret(r1, losses, Simplex(n))
    assert reg <= hedge_regret_bound(n, T)
    assert all(abs(p.sum() - 1) < 1e-9 for p in r1.predictions)


def test_hedge_beats_ftl_on_alternating_experts():
    T = 1000
    losses = alternating_experts(T)
    ftl = regret(run_online(FollowTheLeader(Simplex(2)), losses), losses, Simplex(2))
    hed = regret(run_online(Hedge(2, hedge_eta(2, T)), losses), losses, Simplex(2))
    assert ftl > 0.4 * T and hed < 0.1 * T


def test_fpl_sublinear_and_deterministic_given_seed():
    T, n = 1500, 6
    losses = expert_losses(T, n, seed=10)
    eta = np.sqrt((1 + np.log(n)) / T)
    a = run_online(FollowThePerturbedLeader(n, eta, seed=1), losses)
    b = run_online(FollowThePerturbedLeader(n, eta, seed=1), losses)
    assert np.array_equal(a.predictions, b.predictions)
    assert regret(a, losses, Simplex(n)) < 0.15 * T


def test_strongly_convex_schedule_gives_logarithmic_regret():
    dom = Ball(3, 1.0)
    lam = 0.5
    hs, rs = [200, 800, 3200], []
    for T in hs:
        losses = regression_stream(T, 3, seed=11, lam=lam)
        run = run_online(OnlineGradientDescent(dom, schedule="strong"), losses)
        r = regret(run, losses, dom)
        rs.append(r)
        G = 1.0 + 1.0 + lam  # |a||residual| + lam |x|, residual <= ~1.x
        assert r <= strongly_convex_regret_bound(G * 1.5, lam, T)
    assert growth_exponent(hs, rs) < 0.5


def test_strong_schedule_requires_strong_convexity():
    dom = Ball(2)
    with pytest.raises(ValueError):
        run_online(OnlineGradientDescent(dom, schedule="strong"),
                   [LinearLoss(np.array([1.0, 0.0]))])


def test_ons_feasible_and_low_regret_on_exp_concave_losses():
    T, d = 1500, 4
    dom = Ball(d, 1.0)
    losses = regression_stream(T, d, seed=12, noise=0.05)
    alpha = min(l.exp_concavity for l in losses)
    gamma = OnlineNewtonStep.default_gamma(alpha, grad_bound=2.5, diameter=dom.diameter())
    run = run_online(OnlineNewtonStep(dom, gamma, eps=1.0 / (gamma * dom.diameter()) ** 2),
                     losses)
    assert all(np.linalg.norm(x) <= 1 + 1e-8 for x in run.predictions)
    r = regret(run, losses, dom)
    assert r < 0.5 * ogd_regret_bound(dom.diameter(), 2.5, T)


def test_adagrad_stays_in_box_and_learns():
    T, d = 1500, 5
    dom = Box(d, -1, 1)
    losses = stochastic_linear(T, d, seed=13)
    run = run_online(AdaGrad(dom, eta=0.5), losses)
    assert all(dom.contains(x) for x in run.predictions)
    assert regret(run, losses, dom) < 0.1 * T
    with pytest.raises(TypeError):
        AdaGrad(Ball(3))


def test_regret_curve_linear_final_value_matches_regret():
    T = 300
    dom = Simplex(4)
    losses = expert_losses(T, 4, seed=14)
    run = run_online(Hedge(4, hedge_eta(4, T)), losses)
    curve = regret_curve_linear(run, np.array([l.g for l in losses]), dom)
    assert curve[-1] == pytest.approx(regret(run, losses, dom))
    assert len(curve) == T
