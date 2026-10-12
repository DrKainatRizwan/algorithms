import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from ids_forest import (IsolationForest, ExtendedIsolationForest, average_path_length, make_traffic,
                        roc_auc, average_precision, precision_recall_f1, best_f1_threshold,
                        IntrusionDetector, TRAFFIC_FEATURES)
from ids_forest.tree import IsolationTree


def test_average_path_length_values():
    assert average_path_length(1) == 0.0
    assert average_path_length(2) == 1.0
    assert average_path_length(256) == pytest.approx(10.2448, abs=1e-3)
    arr = average_path_length(np.array([1, 2, 3, 10]))
    assert np.all(np.diff(arr) > 0) or arr[0] == 0


def test_tree_depth_bounded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))
    t = IsolationTree(6, rng).fit(X)
    pl = t.path_lengths(X)
    assert pl.max() <= 6 + average_path_length(64) + 1e-9
    assert (pl > 0).all()


def test_outlier_scores_higher_2d():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 2))
    f = IsolationForest(n_trees=100, seed=1).fit(X)
    s = f.score_samples(np.array([[0.0, 0.0], [8.0, 8.0]]))
    assert s[1] > s[0] and s[1] > 0.6 and s[0] < 0.5


def test_scores_in_unit_interval():
    X = np.random.default_rng(2).normal(size=(300, 4))
    s = IsolationForest(n_trees=30, seed=2).fit(X).score_samples(X)
    assert ((s > 0) & (s <= 1)).all()


def test_determinism():
    X = np.random.default_rng(3).normal(size=(200, 3))
    a = IsolationForest(n_trees=20, seed=7).fit(X).score_samples(X)
    b = IsolationForest(n_trees=20, seed=7).fit(X).score_samples(X)
    c = IsolationForest(n_trees=20, seed=8).fit(X).score_samples(X)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_contamination_controls_flag_rate():
    X = np.random.default_rng(4).normal(size=(1000, 3))
    f = IsolationForest(n_trees=50, contamination=0.1, seed=4).fit(X)
    rate = f.predict(X).mean()
    assert 0.09 <= rate <= 0.11


def test_constant_data_does_not_crash():
    X = np.ones((50, 3))
    f = IsolationForest(n_trees=5, seed=0).fit(X)
    s = f.score_samples(X)
    assert np.allclose(s, s[0])


def test_invalid_args():
    with pytest.raises(ValueError):
        IsolationForest(n_trees=0)
    with pytest.raises(ValueError):
        IsolationForest(contamination=0.9)
    with pytest.raises(RuntimeError):
        IsolationForest().score_samples(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        ExtendedIsolationForest(extension_level=5).fit(np.random.default_rng(0).normal(size=(30, 2)))


def test_extended_level_zero_is_axis_parallel():
    X = np.random.default_rng(5).normal(size=(100, 3))
    f = ExtendedIsolationForest(n_trees=5, extension_level=0, seed=5).fit(X)
    stack = [f.trees[0].root]
    while stack:
        n = stack.pop()
        if n.left is None:
            continue
        assert np.count_nonzero(n.normal) == 1
        stack += [n.left, n.right]


def test_extended_removes_axis_artifact():
    # Two orthogonal clusters on a cross shape; a point off the arms but on the axes' bounding
    # box corner should be flagged more strongly by the extended forest.
    rng = np.random.default_rng(6)
    a = np.column_stack([rng.normal(0, 3, 400), rng.normal(0, 0.2, 400)])
    b = np.column_stack([rng.normal(0, 0.2, 400), rng.normal(0, 3, 400)])
    X = np.vstack([a, b])
    probe = np.array([[2.5, 2.5], [0.0, 0.0]])
    ext = ExtendedIsolationForest(n_trees=200, seed=6).fit(X).score_samples(probe)
    assert ext[0] > ext[1]


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.ones(4)) == 0.5
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert precision_recall_f1(y, np.array([0, 1, 1, 1])) == pytest.approx((2 / 3, 1.0, 0.8))
    thr, f1 = best_f1_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert f1 == 1.0 and 0.2 < thr < 0.8


def test_traffic_generator():
    X, y, k = make_traffic(300, 40, seed=1)
    assert X.shape == (340, len(TRAFFIC_FEATURES)) and y.sum() == 40
    assert set(k[y == 1]) == {0, 1, 2, 3} and (X >= 0).all()
    X2, _, _ = make_traffic(300, 40, seed=1)
    assert np.array_equal(X, X2)


def test_detector_end_to_end_and_explain():
    Xtr, ytr, _ = make_traffic(1500, 0, seed=10, attacks=())
    Xte, yte, kind = make_traffic(800, 80, seed=11)
    det = IntrusionDetector(n_trees=100, contamination=0.05, seed=0, feature_names=TRAFFIC_FEATURES).fit(Xtr)
    s = det.score(Xte)
    assert roc_auc(yte, s) > 0.9
    assert det.predict(Xte)[yte == 1].mean() > 0.6
    # brute-force flows should be explained by login failures
    i = np.flatnonzero(kind == 3)[0]
    assert "failed_logins" in [n for n, _ in det.explain(Xte[i], top=3)]


def test_feature_importance_normalised():
    X, _, _ = make_traffic(400, 20, seed=3)
    for cls in (IsolationForest, ExtendedIsolationForest):
        imp = cls(n_trees=20, seed=0).fit(np.log1p(X)).feature_importance(X)
        assert imp.sum() == pytest.approx(1.0) and (imp >= 0).all()
