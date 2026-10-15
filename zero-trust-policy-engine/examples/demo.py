"""Replay a synthetic enterprise log through the zero-trust engine."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zero_trust import Effect, ZeroTrustEngine, evaluate_scenario, generate_scenario  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    for seed in (0, 1, 2):
        sc = generate_scenario(seed)
        eng = ZeroTrustEngine()
        out = evaluate_scenario(sc, eng)
        print(f"== seed {seed}: {len(sc.events)} events, {len(sc.users)} users, "
              f"attackers: {dict(sorted(sc.attackers.items()))}")
        print(f"   false-positive rate : {out['false_positive_rate']:.3%} "
              f"({out['benign_events']} benign events, {out['benign_drift_alerts']} drift alerts)")
        print(f"   attack events flagged: {out['attack_flag_rate']:.1%} of {out['attack_events']}")
        print(f"   missed attackers    : {out['missed_users'] or 'none'}")
        for kind in ("stolen", "insider"):
            d = [out["detection_delay_hours"][u] for u, k in sc.attackers.items()
                 if k == kind and u in out["detection_delay_hours"]]
            if d:
                print(f"   mean time-to-flag ({kind:7s}): {sum(d) / len(d):6.1f} h")
        print(f"   audit chain valid   : {out['audit_ok']} ({len(eng.audit)} records)")

    # Drill into one insider: trust trajectory and distribution shift.
    sc = generate_scenario(0)
    eng = ZeroTrustEngine()
    target = next(u for u, k in sc.attackers.items() if k == "insider")
    counts = {e: 0 for e in Effect}
    for ev in sc.events:
        r = eng.process(ev)
        if ev.user == target:
            counts[r.decision.effect] += 1
    psi_v, ks_d, ks_p = eng.distribution_shift(target)
    print(f"\ninsider {target}: decisions {{{', '.join(f'{k.value}: {v}' for k, v in counts.items())}}}")
    print(f"   PSI={psi_v:.2f}  KS D={ks_d:.2f} (p={ks_p:.1e})  final trust={eng.trust.score(target):.3f}")
    print(f"\nelapsed {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
