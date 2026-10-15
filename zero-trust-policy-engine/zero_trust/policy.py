"""Attribute-based access policies with deny-overrides and trust thresholds.

A :class:`Rule` couples a resource/action pattern, an attribute
:class:`Condition` and a minimum trust level.  A :class:`PolicySet` evaluates
all matching rules with the usual combining algorithm:

1. any matching ``DENY`` rule wins;
2. otherwise a matching ``PERMIT`` rule grants access if the current trust
   score reaches its ``min_trust``;
3. if trust is below ``min_trust`` but above ``step_up_floor`` the caller is
   asked for step-up authentication;
4. everything else is denied (default deny).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any, Iterable, Mapping, Sequence


class Effect(str, Enum):
    PERMIT = "permit"
    DENY = "deny"
    STEP_UP = "step_up"


@dataclass(frozen=True)
class Request:
    """An access request with its contextual attributes."""

    user: str
    role: str
    resource: str
    action: str
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def context(self) -> dict[str, Any]:
        ctx = dict(self.attrs)
        ctx.update(user=self.user, role=self.role, resource=self.resource, action=self.action)
        return ctx


# ---------------------------------------------------------------- conditions
class Condition:
    """Boolean predicate over a request context."""

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:  # pragma: no cover
        raise NotImplementedError

    def __and__(self, other: "Condition") -> "Condition":
        return And(self, other)

    def __or__(self, other: "Condition") -> "Condition":
        return Or(self, other)

    def __invert__(self) -> "Condition":
        return Not(self)


@dataclass(frozen=True)
class Eq(Condition):
    key: str
    value: Any

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return self.key in ctx and ctx[self.key] == self.value


@dataclass(frozen=True)
class In(Condition):
    key: str
    values: tuple

    def __init__(self, key: str, values: Iterable[Any]):
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "values", tuple(values))

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return self.key in ctx and ctx[self.key] in self.values


@dataclass(frozen=True)
class Gte(Condition):
    key: str
    bound: float

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return self.key in ctx and ctx[self.key] >= self.bound


@dataclass(frozen=True)
class Lte(Condition):
    key: str
    bound: float

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return self.key in ctx and ctx[self.key] <= self.bound


@dataclass(frozen=True)
class Between(Condition):
    """``lo <= ctx[key] <= hi``; if ``lo > hi`` the interval wraps (e.g. 22..6)."""

    key: str
    lo: float
    hi: float

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        if self.key not in ctx:
            return False
        v = ctx[self.key]
        if self.lo <= self.hi:
            return self.lo <= v <= self.hi
        return v >= self.lo or v <= self.hi


@dataclass(frozen=True)
class And(Condition):
    left: Condition
    right: Condition

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return self.left.evaluate(ctx) and self.right.evaluate(ctx)


@dataclass(frozen=True)
class Or(Condition):
    left: Condition
    right: Condition

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return self.left.evaluate(ctx) or self.right.evaluate(ctx)


@dataclass(frozen=True)
class Not(Condition):
    inner: Condition

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return not self.inner.evaluate(ctx)


class _Always(Condition):
    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        return True


ALWAYS: Condition = _Always()


# --------------------------------------------------------------------- rules
@dataclass(frozen=True)
class Rule:
    """A single policy rule.  ``effect`` must be PERMIT or DENY."""

    rule_id: str
    effect: Effect
    resource: str = "*"
    actions: tuple[str, ...] = ("*",)
    condition: Condition = ALWAYS
    min_trust: float = 0.0

    def __post_init__(self) -> None:
        if self.effect not in (Effect.PERMIT, Effect.DENY):
            raise ValueError("rule effect must be PERMIT or DENY")
        if not 0.0 <= self.min_trust <= 1.0:
            raise ValueError("min_trust must lie in [0, 1]")

    def matches(self, req: Request) -> bool:
        if not fnmatchcase(req.resource, self.resource):
            return False
        if not any(fnmatchcase(req.action, a) for a in self.actions):
            return False
        return self.condition.evaluate(req.context())


@dataclass(frozen=True)
class Decision:
    effect: Effect
    reason: str
    trust: float
    matched: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.effect is Effect.PERMIT


class PolicySet:
    """An ordered collection of rules evaluated with deny-overrides."""

    def __init__(self, rules: Sequence[Rule] = (), step_up_floor: float = 0.3):
        self.rules: list[Rule] = []
        self.step_up_floor = step_up_floor
        ids = set()
        for r in rules:
            if r.rule_id in ids:
                raise ValueError(f"duplicate rule id {r.rule_id}")
            ids.add(r.rule_id)
            self.rules.append(r)

    def add(self, rule: Rule) -> None:
        if any(r.rule_id == rule.rule_id for r in self.rules):
            raise ValueError(f"duplicate rule id {rule.rule_id}")
        self.rules.append(rule)

    def evaluate(self, req: Request, trust: float) -> Decision:
        matched = [r for r in self.rules if r.matches(req)]
        ids = tuple(r.rule_id for r in matched)
        denies = [r for r in matched if r.effect is Effect.DENY]
        if denies:
            return Decision(Effect.DENY, f"denied by {denies[0].rule_id}", trust, ids)
        permits = [r for r in matched if r.effect is Effect.PERMIT]
        if not permits:
            return Decision(Effect.DENY, "no matching permit rule (default deny)", trust, ids)
        # The most permissive matching rule decides: lowest trust requirement.
        best = min(permits, key=lambda r: r.min_trust)
        if trust >= best.min_trust:
            return Decision(Effect.PERMIT, f"permitted by {best.rule_id}", trust, ids)
        if trust >= self.step_up_floor:
            return Decision(
                Effect.STEP_UP,
                f"trust {trust:.2f} < {best.min_trust:.2f} required by {best.rule_id}",
                trust, ids,
            )
        return Decision(
            Effect.DENY,
            f"trust {trust:.2f} below step-up floor {self.step_up_floor:.2f}",
            trust, ids,
        )
