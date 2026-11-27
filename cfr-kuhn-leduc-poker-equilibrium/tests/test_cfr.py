import itertools

import numpy as np
import pytest

from counterfactual_regret import (CFRConfig, CFRSolver, KUHN_VALUE, KuhnPoker,
                                   LeducHoldem, MCCFRSolver, best_response, combine,
                                   expected_value, exploitability, infoset_reach,
                                   kuhn_alpha, kuhn_equilibrium, regret_bound,
                                   regret_matching)
from counterfactual_regret.cfr import edge_probabilities, reach_probabilities


def test_tree_sizes(kuhn, leduc):
    assert kuhn.n_nodes == 58 and kuhn.n_infosets == 12 and kuhn.n_slots == 24
    assert leduc.n_infosets == 288 and leduc.n_nodes == 9457


def test_terminal_reach_sums_to_one(kuhn, leduc):
    for tree in (kuhn, leduc):
        probs = edge_probabilities(tree, tree.uniform_strategy())
        reach = reach_probabilities(tree, probs).prod(axis=0)
        assert reach[tree.node_kind == 2].sum() == pytest.approx(1.0)


def test_kuhn_rules():
    g = KuhnPoker()
    s = g.initial()
    for a in ("2", "0", "p", "b", "b"):   # K vs J, check, bet, call
        s = g.apply(s, a)
    assert g.node_kind(s) == 2 and g.utility(s) == 2.0
    s = g.apply(g.apply(g.apply(g.apply(g.initial(), "0"), "2"), "b"), "p")
    assert g.utility(s) == 1.0           # player 1 folds to a bet


def test_leduc_rules():
    g = LeducHoldem()
    s = g.initial()
    for a in ("0", "2", "r", "f"):       # p0 raises, p1 folds
        s = g.apply(s, a)
    assert g.node_kind(s) == 2 and g.utility(s) == 1.0
    s = g.initial()
    for a in ("0", "4", "c", "c", "1", "r", "c"):  # pair of jacks vs king, board J
        s = g.apply(s, a)
    assert g.node_kind(s) == 2
    assert g.utility(s) == pytest.approx(5.0)       # pair beats high card: win 1 + 4
    assert g.actions(g.apply(g.apply(g.initial(), "0"), "2")) == ["c", "r"]


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1 / 3])
def test_kuhn_analytic_equilibrium(kuhn, alpha):
    s = kuhn_equilibrium(kuhn, alpha)
    assert exploitability(kuhn, s) < 1e-12
    assert expected_value(kuhn, s) == pytest.approx(KUHN_VALUE)


def test_invalid_alpha_and_variant(kuhn):
    with pytest.raises(ValueError):
        kuhn_equilibrium(kuhn, 0.5)
    with pytest.raises(ValueError):
        CFRConfig.preset("nonsense")


def test_uniform_exploitability_kuhn(kuhn):
    assert exploitability(kuhn, kuhn.uniform_strategy()) == pytest.approx(0.4583333333, abs=1e-6)


def test_best_response_matches_brute_force(kuhn):
    rng = np.random.default_rng(3)
    sigma = kuhn.uniform_strategy()
    for i in range(kuhn.n_infosets):       # random opponent
        sl = kuhn.infoset_slice(i)
        w = rng.random(2)
        sigma[sl] = w / w.sum()
    value, br = best_response(kuhn, sigma, 0)
    mine = [i for i in range(kuhn.n_infosets) if kuhn.info_player[i] == 0]
    best = -np.inf
    for bits in itertools.product([0, 1], repeat=len(mine)):
        pure = sigma.copy()
        for i, b in zip(mine, bits):
            sl = kuhn.infoset_slice(i)
            pure[sl] = [1 - b, b]
        best = max(best, expected_value(kuhn, pure))
    assert value == pytest.approx(best)
    assert expected_value(kuhn, combine(kuhn, br, sigma)) == pytest.approx(best)


def test_regret_matching_unit(kuhn):
    r = np.zeros(kuhn.n_slots)
    r[0], r[1] = 3.0, -1.0
    s = regret_matching(r, kuhn)
    assert s[0] == 1.0 and s[1] == 0.0
    assert s[2] == pytest.approx(0.5)


def test_vanilla_cfr_converges_kuhn(kuhn):
    solver = CFRSolver(kuhn, "vanilla").run(2000)
    avg = solver.average_strategy
    assert exploitability(kuhn, avg) < 0.01
    assert expected_value(kuhn, avg) == pytest.approx(KUHN_VALUE, abs=5e-3)
    assert avg.shape == (kuhn.n_slots,)


def test_error_decreases_with_iterations(kuhn):
    solver = CFRSolver(kuhn, "vanilla")
    solver.run(100)
    early = exploitability(kuhn, solver.average_strategy)
    solver.run(1900)
    late = exploitability(kuhn, solver.average_strategy)
    assert late < early


@pytest.mark.parametrize("variant", ["vanilla", "plus", "linear", "dcfr"])
def test_all_variants_yield_valid_strategies(kuhn, variant):
    solver = CFRSolver(kuhn, variant).run(300)
    for s in (solver.average_strategy, solver.current_strategy):
        sums = np.bincount(kuhn.slot_info, weights=s)
        assert np.allclose(sums, 1.0) and (s >= 0).all()
    assert exploitability(kuhn, solver.average_strategy) < 0.05


def test_cfr_plus_beats_vanilla_kuhn(kuhn):
    plus = exploitability(kuhn, CFRSolver(kuhn, "plus").run(500).average_strategy)
    vanilla = exploitability(kuhn, CFRSolver(kuhn, "vanilla").run(500).average_strategy)
    assert plus < vanilla and plus < 1e-3


def test_cfr_recovers_kuhn_structure(kuhn):
    s = CFRSolver(kuhn, "plus").run(3000).average_strategy
    alpha = kuhn_alpha(kuhn, s)
    assert 0.0 <= alpha <= 1 / 3 + 0.01
    k_bet = s[kuhn.info_offset[kuhn.key_index["0#K:"]] + 1]
    assert k_bet == pytest.approx(3 * alpha, abs=0.03)
    q_bet = s[kuhn.info_offset[kuhn.key_index["0#Q:"]] + 1]
    assert q_bet < 0.02


def test_leduc_cfr_plus(leduc):
    solver = CFRSolver(leduc, "plus").run(150)
    avg = solver.average_strategy
    assert exploitability(leduc, avg) < 0.02
    assert -0.10 < expected_value(leduc, avg) < -0.07


def test_exploitability_below_regret_bound(kuhn):
    t = 1000
    avg = CFRSolver(kuhn, "vanilla").run(t).average_strategy
    assert exploitability(kuhn, avg) <= regret_bound(2, 2.0, kuhn.n_infosets // 2, t)


def test_equilibrium_not_beaten_by_uniform(kuhn):
    eq = kuhn_equilibrium(kuhn, 0.2)
    uni = kuhn.uniform_strategy()
    assert expected_value(kuhn, combine(kuhn, eq, uni)) >= KUHN_VALUE - 1e-12
    # player 1's equilibrium strategy caps player 0 at the game value
    assert expected_value(kuhn, combine(kuhn, uni, eq)) <= KUHN_VALUE + 1e-12


def test_infoset_reach_root(kuhn):
    reach = infoset_reach(kuhn, kuhn.uniform_strategy())
    assert reach[kuhn.key_index["0#J:"]] == pytest.approx(1 / 3)


@pytest.mark.parametrize("mode,iters", [("external", 20000), ("outcome", 100000)])
def test_mccfr_converges_kuhn(kuhn, mode, iters):
    solver = MCCFRSolver(kuhn, mode, seed=7).run(iters)
    assert exploitability(kuhn, solver.average_strategy) < 0.03


def test_mccfr_deterministic_given_seed(kuhn):
    a = MCCFRSolver(kuhn, "external", seed=5).run(2000).average_strategy
    b = MCCFRSolver(kuhn, "external", seed=5).run(2000).average_strategy
    c = MCCFRSolver(kuhn, "external", seed=6).run(2000).average_strategy
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_mccfr_leduc_improves(leduc):
    base = exploitability(leduc, leduc.uniform_strategy())
    s = MCCFRSolver(leduc, "external", seed=1).run(20000).average_strategy
    assert exploitability(leduc, s) < 0.2 * base


def test_mccfr_bad_mode(kuhn):
    with pytest.raises(ValueError):
        MCCFRSolver(kuhn, "bogus")
