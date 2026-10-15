import pytest

from zero_trust import (And, Between, Decision, Effect, Eq, Gte, In, Lte, Not, Or,
                        PolicySet, Request, Rule)


def req(resource="db/orders", action="read", role="engineer", **attrs):
    return Request("u1", role, resource, action, attrs)


def test_conditions_and_operators():
    ctx = {"a": 3, "b": "x"}
    assert Eq("b", "x").evaluate(ctx)
    assert not Eq("missing", 1).evaluate(ctx)
    assert In("b", ["x", "y"]).evaluate(ctx)
    assert Gte("a", 3).evaluate(ctx) and Lte("a", 3).evaluate(ctx)
    assert (Gte("a", 5) | Eq("b", "x")).evaluate(ctx)
    assert not (Gte("a", 5) & Eq("b", "x")).evaluate(ctx)
    assert (~Eq("b", "z")).evaluate(ctx)
    assert isinstance(And(Eq("a", 1), Eq("b", 2)), And) and isinstance(Or(Eq("a", 1), Eq("b", 2)), Or)
    assert isinstance(Not(Eq("a", 1)), Not)


def test_between_wraps_midnight():
    night = Between("hour", 23, 5)
    assert night.evaluate({"hour": 2}) and night.evaluate({"hour": 23.5})
    assert not night.evaluate({"hour": 12})
    assert Between("hour", 9, 17).evaluate({"hour": 10})


def test_default_deny():
    ps = PolicySet([Rule("r", Effect.PERMIT, "docs/*", ("read",))])
    d = ps.evaluate(req("db/x"), 0.99)
    assert d.effect is Effect.DENY and "default deny" in d.reason


def test_deny_overrides_permit():
    ps = PolicySet([
        Rule("p", Effect.PERMIT, "db/*"),
        Rule("d", Effect.DENY, "db/*", condition=Lte("posture", 0.3)),
    ])
    assert ps.evaluate(req(posture=0.9), 0.9).allowed
    d = ps.evaluate(req(posture=0.1), 0.99)
    assert d.effect is Effect.DENY and "d" in d.matched


def test_trust_thresholds_and_step_up():
    ps = PolicySet([Rule("p", Effect.PERMIT, "db/*", min_trust=0.7)], step_up_floor=0.3)
    assert ps.evaluate(req(), 0.8).effect is Effect.PERMIT
    assert ps.evaluate(req(), 0.5).effect is Effect.STEP_UP
    assert ps.evaluate(req(), 0.1).effect is Effect.DENY


def test_most_permissive_permit_wins():
    ps = PolicySet([
        Rule("strict", Effect.PERMIT, "db/*", min_trust=0.9),
        Rule("lax", Effect.PERMIT, "db/*", ("read",), min_trust=0.4),
    ])
    d = ps.evaluate(req(), 0.5)
    assert d.allowed and "lax" in d.reason


def test_rule_validation_and_duplicates():
    with pytest.raises(ValueError):
        Rule("x", Effect.STEP_UP)
    with pytest.raises(ValueError):
        Rule("x", Effect.PERMIT, min_trust=1.5)
    with pytest.raises(ValueError):
        PolicySet([Rule("a", Effect.PERMIT), Rule("a", Effect.DENY)])
    ps = PolicySet([Rule("a", Effect.PERMIT)])
    with pytest.raises(ValueError):
        ps.add(Rule("a", Effect.DENY))


def test_action_glob_and_role_condition():
    ps = PolicySet([Rule("w", Effect.PERMIT, "db/*", ("wri*",), condition=Eq("role", "admin"))])
    assert ps.evaluate(req(action="write", role="admin"), 0.5).allowed
    assert not ps.evaluate(req(action="write", role="engineer"), 0.5).allowed
    assert isinstance(ps.evaluate(req(action="read", role="admin"), 0.5), Decision)
