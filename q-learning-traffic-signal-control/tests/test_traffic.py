import numpy as np
import pytest

from traffic_rl import (EpsilonSchedule, FixedTime, LongestQueueFirst, MaxPressure, SignalConfig,
                        StateEncoder, TabularAgent, TrafficGrid, asymmetric_demand, evaluate,
                        make_policy, rush_hour_demand, run_controller, train, uniform_demand)


def test_vehicle_conservation():
    env = TrafficGrid(2, 2, uniform_demand(0.15), seed=3)
    ctl = MaxPressure()
    for _ in range(100):
        env.step(ctl.act(env))
    assert env.arrived == env.exited + env.in_network
    assert env.arrived > 0


def test_determinism_same_seed():
    a = run_controller(TrafficGrid(2, 2, uniform_demand(0.1)), FixedTime(15), 60, seed=5)
    b = run_controller(TrafficGrid(2, 2, uniform_demand(0.1)), FixedTime(15), 60, seed=5)
    assert a == b


def test_yellow_blocks_discharge():
    cfg = SignalConfig(yellow=3, min_green=0, interval=1)
    env = TrafficGrid(1, 1, uniform_demand(0.0), cfg)
    env.queue[0] = [5, 5, 5, 5]
    env.step([1])  # request switch -> yellow starts and the interval ticks once
    assert env.yellow_left[0] == 2
    assert env.queue.sum() == 20  # nothing left during yellow
    env.step([0]); env.step([0])
    assert env.phase[0] == 1 and env.green_time[0] == 0


def test_min_green_ignores_early_switch():
    env = TrafficGrid(1, 1, uniform_demand(0.0), SignalConfig(min_green=10, interval=5))
    env.step([1])
    assert env.switches == 0 and env.phase[0] == 0
    env.step([1])
    assert env.switches == 0  # only 5 s of green so far
    env.step([1])
    assert env.switches == 1


def test_saturation_flow_rate():
    cfg = SignalConfig(saturation=0.5, interval=10, turn_prob=0.0)
    env = TrafficGrid(1, 1, uniform_demand(0.0), cfg)
    env.queue[0, 0] = 20
    env.step([0])
    assert env.queue[0, 0] == 15  # 0.5 veh/s for 10 s
    assert env.exited == 5


def test_spillback_blocks_upstream():
    cfg = SignalConfig(capacity=4, turn_prob=0.0, interval=10)
    env = TrafficGrid(2, 1, uniform_demand(0.0), cfg)
    env.queue[0, 0] = 4
    env.queue[1, 0] = 4  # downstream full and held on red (phase 0 serves it, so use phase 1)
    env.phase[1] = 1
    env.step([0, 0])
    assert env.queue[0, 0] == 4  # nothing could move
    assert env.queue.max() <= cfg.capacity


def test_boundary_rejection_when_full():
    cfg = SignalConfig(capacity=3)
    env = TrafficGrid(1, 1, uniform_demand(5.0), cfg, seed=1)
    env.phase[0] = 1
    env.step([0])
    assert env.rejected > 0
    assert env.queue.max() <= 3
    assert env.arrived == env.exited + env.in_network


def test_action_shape_validation():
    env = TrafficGrid(2, 2)
    with pytest.raises(ValueError):
        env.step([0, 1])


def test_state_encoder_roundtrip():
    enc = StateEncoder()
    seen = set()
    for phase in (0, 1):
        for ok in (0, 1):
            for ns in (0, 2, 5, 9, 15, 30):
                for ew in (0, 2, 5, 9, 15, 30):
                    s = enc.encode((phase, ok, ns, ew))
                    assert 0 <= s < enc.n_states
                    seen.add(s)
                    assert enc.decode(s) == (phase, ok, enc.bin(ns), enc.bin(ew))
    assert len(seen) == enc.n_states


def test_q_update_formula():
    ag = TabularAgent(4, 2, algo="q", alpha=0.5, gamma=0.9)
    ag.qa[1] = [1.0, 3.0]
    err = ag.update(0, 1, -2.0, 1)
    assert err == pytest.approx(-2.0 + 0.9 * 3.0)
    assert ag.qa[0, 1] == pytest.approx(0.5 * 0.7)


def test_terminal_update_ignores_bootstrap():
    ag = TabularAgent(2, 2, alpha=1.0)
    ag.qa[1] = [100, 100]
    ag.update(0, 0, 4.0, 1, done=True)
    assert ag.qa[0, 0] == 4.0


def test_double_q_updates_exactly_one_table():
    ag = TabularAgent(3, 2, algo="double_q", alpha=0.5, seed=2)
    before_a, before_b = ag.qa.copy(), ag.qb.copy()
    ag.update(0, 0, -1.0, 1)
    changed = [not np.array_equal(before_a, ag.qa), not np.array_equal(before_b, ag.qb)]
    assert sum(changed) == 1


def test_expected_sarsa_between_min_and_max():
    ag = TabularAgent(2, 2, algo="expected_sarsa", alpha=1.0, gamma=1.0, epsilon=0.2)
    ag.qa[1] = [0.0, 10.0]
    ag.update(0, 0, 0.0, 1)
    assert ag.qa[0, 0] == pytest.approx(0.8 * 10.0 + 0.2 * 0.5 * 10.0)


def test_epsilon_schedule_monotone_with_floor():
    sch = EpsilonSchedule(1.0, 0.1, 0.9)
    vals = [sch.value(i) for i in range(100)]
    assert vals[0] == 1.0 and vals[-1] == 0.1
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError):
        TabularAgent(2, 2, algo="nope")


def test_baselines_beat_fixed_time_on_asymmetric_demand():
    env = TrafficGrid(1, 1, asymmetric_demand())
    seeds = [900, 901, 902]
    fixed = evaluate(env, FixedTime(20), 200, seeds)["mean_queue"]
    lqf = evaluate(env, LongestQueueFirst(), 200, seeds)["mean_queue"]
    mp = evaluate(env, MaxPressure(), 200, seeds)["mean_queue"]
    assert lqf < fixed and mp < fixed


def test_trained_agent_beats_fixed_time_on_rush_hour():
    env = TrafficGrid(1, 1, rush_hour_demand(period=300))
    agent, hist = train(env, steps=200, episodes=100, seed=1)
    seeds = [900, 901, 902]
    learned = evaluate(env, make_policy(agent), 200, seeds)["mean_queue"]
    fixed = evaluate(env, FixedTime(20), 200, seeds)["mean_queue"]
    assert learned < fixed
    assert hist.epsilons[0] > hist.epsilons[-1]
    assert np.mean(hist.mean_queue[-10:]) < np.mean(hist.mean_queue[:10])


def test_training_is_reproducible():
    def run():
        env = TrafficGrid(1, 1, asymmetric_demand())
        ag, _ = train(env, steps=50, episodes=8, seed=4)
        return ag.q
    assert np.array_equal(run(), run())


def test_trained_policy_switches_towards_longer_queue():
    env = TrafficGrid(1, 1, rush_hour_demand(period=300))
    agent, _ = train(env, steps=200, episodes=150, seed=1)
    enc = StateEncoder()
    table = agent.policy_table()
    # switching allowed, served group empty and the other group long -> should switch
    for phase, ns, ew in [(0, 0, 14), (1, 14, 0)]:
        assert table[enc.encode((phase, 1, ns, ew))] == 1
