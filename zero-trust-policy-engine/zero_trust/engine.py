"""The zero-trust decision engine: per-event scoring, drift monitoring,
policy evaluation and a tamper-evident audit log."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field

from .drift import PageHinkley, RunningStats, ks_statistic, psi
from .policy import (
    And, Between, Decision, Effect, Eq, In, Lte, Not, PolicySet, Request, Rule,
)
from .simulation import Event, circular_features, hour_of_day
from .trust import Signal, TrustScorer, haversine_km



@dataclass
class EngineConfig:
    warmup: int = 20
    update_z: float = 2.5          # only "normal" events update the profile
    anomaly_z: float = 4.0
    max_speed_kmh: float = 900.0
    ph_delta: float = 0.3
    ph_threshold: float = 8.0
    z_clip: float = 10.0
    half_life: float = 6 * 3600.0
    prior: float = 0.8
    window: int = 40


def default_policy() -> PolicySet:
    """A small but realistic role/resource policy with trust thresholds."""
    return PolicySet([
        Rule("deny-bad-posture", Effect.DENY, condition=Lte("posture", 0.3)),
        Rule("deny-admin-non-admin", Effect.DENY, "admin/*", condition=Not(Eq("role", "admin"))),
        Rule("deny-night-finance", Effect.DENY, "finance/payroll",
             condition=Between("hour", 23, 5) & Not(Eq("network", "corp"))),
        Rule("docs-read", Effect.PERMIT, "docs/*", ("read",), min_trust=0.4),
        Rule("db-read", Effect.PERMIT, "db/*", ("read",), min_trust=0.6),
        Rule("db-write", Effect.PERMIT, "db/*", ("write",), min_trust=0.8,
             condition=In("role", ("engineer", "admin"))),
        Rule("finance-read", Effect.PERMIT, "finance/*", ("read",), min_trust=0.7,
             condition=In("role", ("finance", "admin"))),
        Rule("admin-all", Effect.PERMIT, "admin/*", min_trust=0.85,
             condition=Eq("role", "admin")),
    ], step_up_floor=0.3)


class AuditLog:
    """Append-only hash chain; any edit to a past record breaks :meth:`verify`."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self.records: list[dict] = []

    @staticmethod
    def _digest(prev: str, body: dict) -> str:
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((prev + blob).encode()).hexdigest()

    def append(self, body: dict) -> str:
        prev = self.records[-1]["hash"] if self.records else self.GENESIS
        h = self._digest(prev, body)
        self.records.append({"body": body, "prev": prev, "hash": h})
        return h

    def verify(self) -> bool:
        prev = self.GENESIS
        for rec in self.records:
            if rec["prev"] != prev or rec["hash"] != self._digest(prev, rec["body"]):
                return False
            prev = rec["hash"]
        return True

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class _UserState:
    stats: dict[str, RunningStats] = field(default_factory=lambda: {
        k: RunningStats() for k in ("log_bytes", "cos", "sin")})
    detector: PageHinkley | None = None
    devices: set[str] = field(default_factory=set)
    last: tuple[float, float, float] | None = None  # ts, lat, lon
    baseline: list[float] = field(default_factory=list)
    recent: deque = field(default_factory=deque)
    n: int = 0


@dataclass(frozen=True)
class EngineResult:
    decision: Decision
    trust: float
    anomaly_z: float
    signals: tuple[str, ...]
    drift_alert: bool

    @property
    def flagged(self) -> bool:
        """True when the engine did anything other than plainly permit."""
        return self.decision.effect is not Effect.PERMIT or self.drift_alert


class ZeroTrustEngine:
    """Scores every request, updates trust, and consults the policy set."""

    def __init__(self, policy: PolicySet | None = None, config: EngineConfig | None = None):
        self.policy = policy or default_policy()
        self.cfg = config or EngineConfig()
        self.trust = TrustScorer(self.cfg.prior, self.cfg.half_life)
        self.audit = AuditLog()
        self._users: dict[str, _UserState] = {}

    # ---------------------------------------------------------- internals
    def _state(self, user: str) -> _UserState:
        st = self._users.get(user)
        if st is None:
            st = _UserState(detector=PageHinkley(self.cfg.ph_delta, self.cfg.ph_threshold),
                            recent=deque(maxlen=self.cfg.window))
            self._users[user] = st
        return st

    def _anomaly(self, st: _UserState, feats: dict[str, float]) -> float:
        if st.n < self.cfg.warmup:
            return 0.0
        floors = {"log_bytes": 0.15, "cos": 0.1, "sin": 0.1}
        return max(abs(st.stats[k].zscore(x, floors[k])) for k, x in feats.items())

    # -------------------------------------------------------------- public
    def process(self, ev: Event) -> EngineResult:
        cfg, st = self.cfg, self._state(ev.user)
        hour = hour_of_day(ev.ts)
        c, s = circular_features(hour)
        feats = {"log_bytes": math.log1p(ev.nbytes), "cos": c, "sin": s}
        z = self._anomaly(st, feats)
        signals: list[Signal] = []

        if not ev.login_ok:
            signals.append(Signal("login_failure", -0.8))
        elif ev.mfa:
            signals.append(Signal("mfa_success", 0.3))
        if st.devices and ev.device not in st.devices:
            signals.append(Signal("new_device", -1.0))
        if ev.posture < 0.5:
            signals.append(Signal("poor_posture", -0.8))
        if st.last is not None:
            pts, plat, plon = st.last
            km = haversine_km(plat, plon, ev.lat, ev.lon)
            hrs = max(ev.ts - pts, 1.0) / 3600.0
            if km > 100.0 and km / hrs > cfg.max_speed_kmh:
                signals.append(Signal("impossible_travel", -3.0))
        if z > cfg.anomaly_z:
            signals.append(Signal("behaviour_anomaly", -0.4 * min(z - 3.0, 6.0)))

        drift = False
        if st.n >= cfg.warmup:
            zb = min(max(st.stats["log_bytes"].zscore(feats["log_bytes"], 0.15),
                         -cfg.z_clip), cfg.z_clip)
            drift = bool(st.detector.update(zb))  # type: ignore[union-attr]
            if drift:
                signals.append(Signal("behavioural_drift", -2.0))
        if not any(sg.weight < 0 for sg in signals):
            signals.append(Signal("clean_event", 0.15))

        trust = self.trust.update(ev.user, ev.ts, signals)
        req = Request(ev.user, ev.role, ev.resource, ev.action, {
            "hour": hour, "network": ev.network, "posture": ev.posture,
            "device": ev.device, "trust": trust,
        })
        decision = self.policy.evaluate(req, trust)

        # Profile learning: only from events that looked normal and were allowed.
        if decision.allowed and (st.n < cfg.warmup or z < cfg.update_z):
            for k, x in feats.items():
                st.stats[k].push(x)
            st.devices.add(ev.device)
            if len(st.baseline) < cfg.window:
                st.baseline.append(feats["log_bytes"])
            else:
                st.recent.append(feats["log_bytes"])
        elif st.n >= cfg.warmup:
            st.recent.append(feats["log_bytes"])
        st.n += 1
        st.last = (ev.ts, ev.lat, ev.lon)

        self.audit.append({
            "ts": round(ev.ts, 3), "user": ev.user, "resource": ev.resource,
            "action": ev.action, "effect": decision.effect.value,
            "trust": round(trust, 4), "signals": [sg.name for sg in signals],
        })
        return EngineResult(decision, trust, z, tuple(sg.name for sg in signals), drift)

    def distribution_shift(self, user: str) -> tuple[float, float, float]:
        """(PSI, KS statistic, KS p-value) of recent vs baseline log-bytes."""
        st = self._users.get(user)
        if st is None or len(st.baseline) < 10 or len(st.recent) < 10:
            raise ValueError("not enough data for a distribution comparison")
        d, p = ks_statistic(st.baseline, list(st.recent))
        return psi(st.baseline, list(st.recent), bins=5), d, p


def evaluate_scenario(scenario, engine: ZeroTrustEngine | None = None) -> dict:
    """Replay a scenario and report detection quality."""
    eng = engine or ZeroTrustEngine()
    first_flag: dict[str, float] = {}
    benign_total = benign_flagged = 0
    attack_total = attack_flagged = 0
    drift_alerts_benign = 0
    for ev in scenario.events:
        res = eng.process(ev)
        if ev.label == "benign":
            benign_total += 1
            benign_flagged += int(not res.decision.allowed)
            drift_alerts_benign += int(res.drift_alert)
        else:
            attack_total += 1
            attack_flagged += int(res.flagged)
            if res.flagged and ev.user not in first_flag:
                first_flag[ev.user] = ev.ts
    delays = {u: (first_flag[u] - scenario.attack_start[u]) / 3600.0
              for u in scenario.attackers if u in first_flag}
    return {
        "benign_events": benign_total,
        "false_positive_rate": benign_flagged / max(benign_total, 1),
        "attack_events": attack_total,
        "attack_flag_rate": attack_flagged / max(attack_total, 1),
        "detected_users": sorted(first_flag),
        "missed_users": sorted(set(scenario.attackers) - set(first_flag)),
        "detection_delay_hours": delays,
        "benign_drift_alerts": drift_alerts_benign,
        "audit_ok": eng.audit.verify(),
    }
