# Zero-Trust Access Policy Engine with Continuous Trust Scoring and Drift Detection

## Overview

A from-scratch Python implementation of the decision core of a zero-trust access system. Every request is evaluated on its own merits: the engine keeps a continuously updated trust score for each identity, watches each user's behaviour for sudden anomalies and slow drift, and then asks an attribute-based policy set whether the request may proceed, needs step-up authentication, or must be denied. Every decision is written to a hash-chained audit log.

Only the standard library and NumPy are used.

| Module | Contents |
|---|---|
| `zero_trust/policy.py` | Attribute conditions (`Eq`, `In`, `Between`, `And`/`Or`/`Not`, ...), rules with trust thresholds, deny-overrides policy set |
| `zero_trust/trust.py` | Log-odds trust scorer with exponential decay, haversine distance |
| `zero_trust/drift.py` | Welford statistics, Page-Hinkley, CUSUM, PSI, two-sample KS test |
| `zero_trust/simulation.py` | Synthetic access logs with stolen-credential and insider-drift attackers |
| `zero_trust/engine.py` | Per-event pipeline, profile learning, audit chain, scenario evaluation |

## How it works

**Trust.** Each user has a log-odds value `l`. Evidence adds a signed weight (failed login −0.8, new device −1.0, impossible travel −3.0, MFA success +0.3, clean event +0.15, behavioural anomaly down to −2.4, drift alarm −2.0). Between events the value relaxes to the prior: `l(t) = l0 + (l − l0)·2^(−Δt/T½)`, and the score is `sigmoid(l)`. Distrust therefore fades with time, and so does accumulated goodwill.

**Anomaly and drift.** Each user has a Welford profile over log-bytes and the circular encoding (cos, sin) of the hour of day. The anomaly score is the largest absolute z-score. The profile is only updated from allowed, unremarkable events (|z| < 2.5), so an attacker cannot easily drag the baseline along. Because the baseline stays frozen while behaviour moves away from it, the z-score of a slowly growing transfer volume keeps rising, and a Page-Hinkley test on that z-score raises a drift alarm long before any fixed threshold would trip. PSI and a two-sample KS test compare the baseline window with the recent window for explanation.

**Policy.** Rules match a resource glob, an action glob and a condition. Evaluation is deny-overrides: any matching `DENY` wins; otherwise the most permissive matching `PERMIT` applies if trust ≥ its `min_trust`; trust between the step-up floor and `min_trust` yields `STEP_UP`; anything else is denied (default deny).

**Audit.** Each record stores `SHA-256(prev_hash ‖ canonical_json(body))`; editing any past record breaks `AuditLog.verify()`.

## Complexity

| Operation | Time | Space |
|---|---|---|
| Policy evaluation | O(R·C) for R rules of condition size C | O(R) |
| Trust update | O(S) for S signals | O(1) per user (+ history) |
| Profile / detector update | O(1) | O(1) per user |
| PSI / KS comparison | O(n log n) | O(n) |
| Audit append / verify | O(1) / O(N) | O(N) |

## Usage

```python
from zero_trust import ZeroTrustEngine, generate_scenario, evaluate_scenario

scenario = generate_scenario(seed=0)
engine = ZeroTrustEngine()
for ev in scenario.events:
    result = engine.process(ev)
    if result.flagged:
        print(ev.user, result.decision.effect.value, f"{result.trust:.2f}", result.signals)

print(evaluate_scenario(scenario)["false_positive_rate"])
```

Custom policies:

```python
from zero_trust import PolicySet, Rule, Effect, Eq, Between
policy = PolicySet([
    Rule("no-night-payroll", Effect.DENY, "finance/payroll", condition=Between("hour", 23, 5)),
    Rule("finance-read", Effect.PERMIT, "finance/*", ("read",), condition=Eq("role", "finance"), min_trust=0.7),
])
```

## Results

`python examples/demo.py` replays 24 users × 14 days (2,688 events) with three stolen-credential attackers and three gradual-exfiltration insiders starting on day 9:

| Seed | False-positive rate | Attack events flagged | Missed attackers | Mean time-to-flag, stolen | Mean time-to-flag, insider |
|---|---|---|---|---|---|
| 0 | 3.19% | 54.2% | 0 | 1.3 h | 37.9 h |
| 1 | 3.39% | 55.8% | 0 | 1.8 h | 27.2 h |
| 2 | 2.86% | 58.8% | 0 | 1.2 h | 27.7 h |

Every attacker is flagged in all three runs. For one insider the baseline and recent transfer-size distributions give PSI = 2.70 and KS D = 0.58 (p = 1.5e-06). The false-positive figure counts benign requests that were not plainly permitted (mostly step-ups for legitimate writes after a dip in trust). Not every attack event is flagged: once a stolen identity is within the trust range a rule permits, some individual requests still go through, which is why the per-event flag rate is lower than the per-attacker detection rate.

Run the tests with `python -m pytest -q` (30 tests).

## References

- NIST SP 800-207, *Zero Trust Architecture*, 2020.
- E. S. Page, "Continuous Inspection Schemes", *Biometrika* 41, 1954 (CUSUM and the Page-Hinkley test).
- D. V. Hinkley, "Inference about the change-point from cumulative sum tests", *Biometrika* 58, 1971.
- J. Gama et al., "A survey on concept drift adaptation", *ACM Computing Surveys* 46(4), 2014.
- B. P. Welford, "Note on a method for calculating corrected sums of squares and products", *Technometrics* 4, 1962.
- XACML 3.0, OASIS standard (deny-overrides combining algorithm).

---

**Author:** Dr. Kainat Rizwan
