import dataclasses

import pytest

from zero_trust import (AuditLog, Effect, ZeroTrustEngine, evaluate_scenario,
                        generate_scenario)
from zero_trust.simulation import Event

DAY = 86400.0


def make_event(ts, **kw):
    base = dict(ts=ts, user="alice", role="engineer", device="laptop", lat=51.5, lon=-0.1,
                resource="docs/wiki", action="read", nbytes=8000.0, login_ok=True,
                mfa=False, posture=0.95, network="corp")
    base.update(kw)
    return Event(**base)


def warmed_engine(n=30):
    eng = ZeroTrustEngine()
    for i in range(n):
        eng.process(make_event(i * 3600.0 + 10 * 3600 % DAY, nbytes=8000 + 50 * (i % 7)))
    return eng


def test_scenario_is_deterministic():
    a, b = generate_scenario(5), generate_scenario(5)
    assert a.events == b.events and a.attackers == b.attackers
    assert generate_scenario(6).events != a.events
    assert all(x.ts <= y.ts for x, y in zip(a.events, a.events[1:]))


def test_scenario_validation():
    with pytest.raises(ValueError):
        generate_scenario(0, n_users=3, n_stolen=2, n_insider=2)


def test_benign_user_is_permitted():
    eng = ZeroTrustEngine()
    res = [eng.process(make_event(i * 3600.0 + 36000)) for i in range(5)]
    assert all(r.decision.allowed for r in res)
    assert res[-1].trust > 0.8


def test_new_device_and_impossible_travel_collapse_trust():
    eng = warmed_engine()
    t = 30 * 3600.0 + 36000
    res = eng.process(make_event(t + 60, device="rogue", lat=1.35, lon=103.8))
    assert "new_device" in res.signals and "impossible_travel" in res.signals
    assert res.trust < 0.3 and res.decision.effect is Effect.DENY


def test_large_transfer_triggers_behaviour_anomaly():
    eng = warmed_engine()
    res = eng.process(make_event(31 * 3600.0 + 36000, nbytes=8000 * 500))
    assert "behaviour_anomaly" in res.signals and res.anomaly_z > 4


def test_poor_posture_denied_by_policy():
    eng = warmed_engine()
    res = eng.process(make_event(31 * 3600.0 + 36000, posture=0.2))
    assert res.decision.effect is Effect.DENY and "deny-bad-posture" in res.decision.matched


def test_non_admin_cannot_touch_admin_resources():
    eng = warmed_engine()
    res = eng.process(make_event(31 * 3600.0 + 36000, resource="admin/keys"))
    assert not res.decision.allowed


def test_trust_recovers_over_time():
    eng = warmed_engine()
    t = 31 * 3600.0 + 36000
    low = eng.process(make_event(t, login_ok=False, posture=0.4)).trust
    assert eng.trust.score("alice", t + 3 * DAY) > low


def test_gradual_exfiltration_raises_drift_alert():
    eng = warmed_engine(40)
    alerts, t = 0, 41 * 3600.0
    for k in range(60):
        t += 3600.0
        nbytes = 8000 * (1.0 + 0.08 * k)
        alerts += eng.process(make_event(t, nbytes=nbytes)).drift_alert
    assert alerts >= 1
    psi_v, ks_d, ks_p = eng.distribution_shift("alice")
    assert psi_v > 0.25 and ks_p < 0.01


def test_distribution_shift_needs_data():
    with pytest.raises(ValueError):
        ZeroTrustEngine().distribution_shift("nobody")


def test_audit_log_detects_tampering():
    eng = warmed_engine(10)
    assert eng.audit.verify() and len(eng.audit) == 10
    eng.audit.records[3]["body"]["effect"] = "permit-forged"
    assert not eng.audit.verify()
    log = AuditLog()
    log.append({"a": 1}); log.append({"a": 2})
    log.records[0]["body"]["a"] = 9
    assert not log.verify()


def test_full_scenario_detects_all_attackers_with_low_fpr():
    sc = generate_scenario(0)
    out = evaluate_scenario(sc)
    assert out["missed_users"] == []
    assert out["false_positive_rate"] < 0.06
    assert out["audit_ok"]
    stolen = [u for u, k in sc.attackers.items() if k == "stolen"]
    assert all(out["detection_delay_hours"][u] < 12 for u in stolen)


def test_event_is_frozen():
    ev = make_event(0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.user = "bob"  # type: ignore[misc]
