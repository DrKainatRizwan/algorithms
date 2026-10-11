import numpy as np
import pytest

from graph_message_passing import (GNN, GATLayer, cross_entropy, dirichlet_energy,
                                   edge_homophily, evaluate, gcn_normalize,
                                   gradient_check, mean_normalize, smoothing_curve,
                                   softmax, split_masks, stochastic_block_model, train)


@pytest.fixture(scope="module")
def graph():
    return stochastic_block_model([25, 25, 25], 0.25, 0.02, n_features=8, seed=1)


def test_sbm_shape_and_symmetry(graph):
    assert graph.adj.shape == (75, 75)
    assert np.allclose(graph.adj, graph.adj.T)
    assert np.all(np.diag(graph.adj) == 0)
    assert graph.n_classes == 3


def test_sbm_homophily(graph):
    assert edge_homophily(graph) > 0.7


def test_sbm_deterministic():
    a = stochastic_block_model([10, 10], 0.3, 0.05, seed=5)
    b = stochastic_block_model([10, 10], 0.3, 0.05, seed=5)
    assert np.array_equal(a.adj, b.adj) and np.array_equal(a.x, b.x)


def test_gcn_normalization_symmetric_and_spectrum(graph):
    a = gcn_normalize(graph.adj)
    assert np.allclose(a, a.T)
    assert np.linalg.eigvalsh(a).max() == pytest.approx(1.0, abs=1e-9)


def test_mean_normalize_rows():
    adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0], ], float)
    p = mean_normalize(adj)
    assert np.allclose(p.sum(axis=1), 1.0)
    iso = mean_normalize(np.zeros((2, 2)))
    assert np.all(iso == 0)


def test_split_masks_disjoint(graph):
    tr, va, te = split_masks(graph.y, 4, 15, seed=0)
    assert tr.sum() == 12 and va.sum() == 15
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    assert all(graph.y[tr].tolist().count(c) == 4 for c in range(3))


def test_softmax_and_loss_gradient():
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert np.allclose(softmax(z).sum(axis=1), 1.0)
    y = np.array([2, 0])
    mask = np.array([True, True])
    loss, grad = cross_entropy(z, y, mask)
    assert loss == pytest.approx((-np.log(softmax(z)[0, 2]) - np.log(1 / 3)) / 2)
    assert np.allclose(grad.sum(axis=1), 0.0)


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat", "mlp"])
def test_gradients_match_finite_differences(graph, kind):
    model = GNN(kind, graph.adj, [8, 6, 3], dropout=0.0, seed=3)
    mask = np.zeros(graph.n_nodes, dtype=bool)
    mask[::3] = True
    assert gradient_check(model, graph, mask) < 1e-4


def test_gat_attention_rows_sum_to_one_on_neighbourhood(graph):
    rng = np.random.default_rng(0)
    layer = GATLayer(graph.adj, 8, 4, rng, heads=2)
    layer.forward(graph.x)
    support = (graph.adj + np.eye(graph.n_nodes)) > 0
    for alpha in layer.attention():
        assert np.allclose(alpha.sum(axis=1), 1.0)
        assert np.all(alpha[~support] == 0)


def test_gat_requires_divisible_heads(graph):
    with pytest.raises(ValueError):
        GATLayer(graph.adj, 8, 5, np.random.default_rng(0), heads=2)


def test_unknown_kind_rejected(graph):
    with pytest.raises(ValueError):
        GNN("transformer", graph.adj, [8, 3])


def test_dropout_only_in_training(graph):
    model = GNN("gcn", graph.adj, [8, 8, 3], dropout=0.5, seed=0)
    a = model.forward(graph.x)
    b = model.forward(graph.x)
    assert np.array_equal(a, b)
    c = model.forward(graph.x, train=True)
    assert not np.allclose(a, c)


def test_training_beats_mlp_on_graph_task():
    g = stochastic_block_model([40, 40, 40], 0.2, 0.01, n_features=16, signal=0.5, seed=2)
    tr, va, te = split_masks(g.y, 5, 30, seed=0)
    accs = {}
    for kind in ("gcn", "mlp"):
        m = GNN(kind, g.adj, [16, 16, 3], dropout=0.3, seed=0)
        train(m, g, tr, va, epochs=150)
        accs[kind] = evaluate(m, g, te)
    assert accs["gcn"] > accs["mlp"] + 0.05
    assert accs["gcn"] > 0.8


@pytest.mark.parametrize("kind", ["sage", "gat"])
def test_other_architectures_learn(kind):
    g = stochastic_block_model([30, 30], 0.25, 0.01, n_features=8, signal=1.0, seed=4)
    tr, va, te = split_masks(g.y, 5, 10, seed=0)
    m = GNN(kind, g.adj, [8, 8, 2], dropout=0.2, seed=0)
    h = train(m, g, tr, va, epochs=200)
    assert h.loss[-1] < h.loss[0]
    assert evaluate(m, g, te) > 0.8


def test_training_is_deterministic():
    g = stochastic_block_model([20, 20], 0.3, 0.02, seed=7)
    tr, va, _ = split_masks(g.y, 4, 8, seed=0)
    outs = []
    for _ in range(2):
        m = GNN("gcn", g.adj, [16, 8, 2], seed=1)
        train(m, g, tr, va, epochs=30)
        outs.append(m.forward(g.x))
    assert np.array_equal(outs[0], outs[1])


def test_oversmoothing_energy_decreases(graph):
    curve = smoothing_curve(graph, 20)
    assert curve[-1] < 0.2 * curve[0]
    assert dirichlet_energy(np.ones((75, 3)), graph.adj) == pytest.approx(0.0)
