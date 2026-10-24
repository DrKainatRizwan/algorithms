import random

import numpy as np
import pytest

from secure_aggregation import FLConfig, make_dataset, train
from secure_aggregation.federated import aggregate, local_update, sigmoid


@pytest.fixture(scope="module")
def data():
    return make_dataset(n_clients=16, n_per_client=50, dim=6, seed=0)


def test_sigmoid_stable():
    z = np.array([-800.0, 0.0, 800.0])
    assert np.allclose(sigmoid(z), [0.0, 0.5, 1.0])


def test_local_update_reduces_loss(data):
    from secure_aggregation.federated import log_loss
    x, y = data.shards[0]
    w0 = np.zeros(6)
    d = local_update(w0, x, y, 20, 0.5)
    assert log_loss(w0 + d, x, y) < log_loss(w0, x, y)


def test_secure_mean_matches_plain_mean():
    rng = np.random.default_rng(0)
    ups = {u: rng.normal(scale=0.2, size=5) for u in range(1, 9)}
    cfg = FLConfig(aggregator="secure")
    got = aggregate(ups, cfg, random.Random(0), 1)
    want = np.mean(list(ups.values()), axis=0)
    assert np.allclose(got, want, atol=1e-4)


def test_training_learns_and_secure_equals_plain(data):
    plain = train(data, FLConfig(aggregator="plain", rounds=8, seed=1))
    sec = train(data, FLConfig(aggregator="secure", rounds=8, seed=1))
    assert plain["accuracy"] > 0.75
    assert abs(plain["accuracy"] - sec["accuracy"]) < 0.02


def test_training_survives_dropouts(data):
    out = train(data, FLConfig(aggregator="secure", rounds=8, dropout_rate=0.25, seed=3))
    assert out["accuracy"] > 0.75 and out["skipped_rounds"] == 0


def test_bucketed_beats_plain_under_attack(data):
    kw = dict(rounds=8, clip=2.0, byzantine_frac=0.2, n_buckets=8, seed=2)
    plain = train(data, FLConfig(aggregator="plain", **kw))
    buck = train(data, FLConfig(aggregator="bucketed", **kw))
    assert buck["loss"] < plain["loss"]


def test_training_deterministic(data):
    a = train(data, FLConfig(aggregator="secure", rounds=3, seed=5))
    b = train(data, FLConfig(aggregator="secure", rounds=3, seed=5))
    assert np.array_equal(a["w"], b["w"])


def test_unknown_aggregator():
    with pytest.raises(ValueError):
        aggregate({1: np.zeros(2)}, FLConfig(aggregator="nope"), random.Random(0), 0)
