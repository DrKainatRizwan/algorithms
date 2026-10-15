"""Synthetic enterprise access logs with credential-theft and insider-drift attackers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CITIES = {
    "london": (51.51, -0.13), "berlin": (52.52, 13.40), "madrid": (40.42, -3.70),
    "boston": (42.36, -71.06), "austin": (30.27, -97.74), "lahore": (31.55, 74.34),
    "singapore": (1.35, 103.82), "lagos": (6.52, 3.38),
}
ROLE_RESOURCES = {
    "engineer": ["docs/wiki", "docs/runbook", "db/orders", "db/metrics"],
    "finance": ["docs/wiki", "finance/ledger", "finance/payroll", "db/orders"],
    "admin": ["docs/wiki", "db/orders", "admin/users", "admin/keys"],
}
DAY = 86400.0


@dataclass(frozen=True)
class Event:
    ts: float
    user: str
    role: str
    device: str
    lat: float
    lon: float
    resource: str
    action: str
    nbytes: float
    login_ok: bool
    mfa: bool
    posture: float
    network: str
    label: str = "benign"  # benign | stolen | insider


@dataclass
class Scenario:
    events: list[Event]
    users: list[str]
    attackers: dict[str, str]  # user -> kind
    attack_start: dict[str, float] = field(default_factory=dict)


def generate_scenario(
    seed: int = 0,
    n_users: int = 24,
    days: int = 14,
    n_stolen: int = 3,
    n_insider: int = 3,
    events_per_day: int = 8,
    attack_day: int = 9,
) -> Scenario:
    """Generate a reproducible log.  Attackers act from ``attack_day`` on.

    * ``stolen``: valid credentials used from a foreign city, a new device,
      at night, with bulk transfers and occasional failed logins.
    * ``insider``: same device, place and hours, but transfer volume grows
      gradually - invisible to simple threshold rules.
    """
    if n_stolen + n_insider > n_users:
        raise ValueError("more attackers than users")
    rng = np.random.default_rng(seed)
    city_names = list(CITIES)
    roles = list(ROLE_RESOURCES)
    users = [f"user{i:02d}" for i in range(n_users)]
    profile = {}
    for u in users:
        profile[u] = dict(
            role=roles[int(rng.integers(len(roles)))],
            city=city_names[int(rng.integers(len(city_names)))],
            hour=float(rng.uniform(9, 15)),
            mu=float(rng.uniform(8.0, 10.0)),
            sigma=float(rng.uniform(0.35, 0.6)),
        )
    chosen = rng.permutation(n_users)
    attackers: dict[str, str] = {}
    for i in chosen[:n_stolen]:
        attackers[users[int(i)]] = "stolen"
    for i in chosen[n_stolen:n_stolen + n_insider]:
        attackers[users[int(i)]] = "insider"

    events: list[Event] = []
    start = {u: attack_day * DAY for u in attackers}
    for u in users:
        p = profile[u]
        lat0, lon0 = CITIES[p["city"]]
        kind = attackers.get(u)
        for d in range(days):
            attacking = kind is not None and d >= attack_day
            for _ in range(events_per_day):
                hour = float(np.clip(rng.normal(p["hour"], 1.8), 0.5, 23.0))
                dev, lat, lon, label = f"dev-{u}", lat0, lon0, "benign"
                nbytes = float(rng.lognormal(p["mu"], p["sigma"]))
                login_ok, mfa, net = bool(rng.random() > 0.01), bool(rng.random() < 0.3), "corp"
                if rng.random() < 0.15:
                    net = "vpn"
                posture = float(np.clip(rng.normal(0.92, 0.04), 0, 1))
                if attacking and kind == "stolen":
                    far = [c for c in city_names if c != p["city"]]
                    lat, lon = CITIES[far[(int(chosen[0]) + d) % len(far)]]
                    dev, label, net = f"dev-rogue-{u}", "stolen", "public"
                    hour = float(rng.uniform(1.0, 4.5))
                    nbytes *= float(rng.uniform(15, 40))
                    login_ok = bool(rng.random() > 0.25)
                    mfa = False
                    posture = float(np.clip(rng.normal(0.55, 0.15), 0, 1))
                elif attacking and kind == "insider":
                    label = "insider"
                    nbytes *= 1.0 + 0.35 * (d - attack_day + 1)
                res = ROLE_RESOURCES[p["role"]][int(rng.integers(4))]
                act = "write" if (res.startswith("db/") and rng.random() < 0.1) else "read"
                events.append(Event(
                    d * DAY + hour * 3600.0 + float(rng.uniform(0, 60)), u, p["role"], dev,
                    lat, lon, res, act, nbytes, login_ok, mfa, posture, net, label,
                ))
    events.sort(key=lambda e: e.ts)
    return Scenario(events, users, attackers, start)


def hour_of_day(ts: float) -> float:
    return (ts % DAY) / 3600.0


def circular_features(hour: float) -> tuple[float, float]:
    ang = 2 * math.pi * hour / 24.0
    return math.cos(ang), math.sin(ang)
