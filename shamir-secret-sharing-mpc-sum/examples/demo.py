"""Demo: threshold sharing, verification, error correction and private sums."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from secret_sharing import (BGW, GroupParams, Share, PRIME_127, deal,
                            make_shares, reconstruct, reconstruct_bytes,
                            robust_reconstruct, secure_mean_variance,
                            share_bytes, shamir_secure_sum, verify_share,
                            additive_secure_sum, refresh_shares)
from secret_sharing.vss import find_cheaters


def main() -> None:
    rng = random.Random(2026)
    print("== (3,5) Shamir sharing ==")
    secret = 0xC0FFEE_DEADBEEF
    shares = make_shares(secret, 3, 5, rng=rng)
    print("recovered from shares 2,4,5:", hex(reconstruct([shares[1], shares[3], shares[4]])))
    print("recovered from only 2 shares:", hex(reconstruct(shares[:2])), "(wrong, as expected)")

    print("\n== Byte strings ==")
    parties = share_bytes(b"launch code: 0451", 3, 5, rng=rng)
    print(reconstruct_bytes([parties[0], parties[2], parties[3]]))

    print("\n== Proactive refresh ==")
    new = refresh_shares(shares, 3, rng)
    print("same secret:", reconstruct(new[:3]) == secret,
          "| mixed old/new shares:", reconstruct([shares[0], shares[1], new[2]]) == secret)

    print("\n== Feldman VSS ==")
    group = GroupParams.generate(64, seed=3)
    d = deal(424242, 4, 7, group, rng)
    forged = list(d.shares)
    forged[5] = Share(6, (forged[5].y + 1) % group.q, group.q)
    print("valid shares:", sum(verify_share(s, d.commitments, group) for s in d.shares), "/ 7",
          "| cheaters detected at x =", find_cheaters(forged, d.commitments, group))

    print("\n== Berlekamp-Welch error correction ==")
    shares = make_shares(31415, 3, 11, rng=rng)
    bad = list(shares)
    for i in (1, 5, 8, 10):
        bad[i] = Share(bad[i].x, rng.randrange(PRIME_127))
    value, wrong = robust_reconstruct(bad, 3)
    print("secret:", value, "corrupted x:", wrong)

    print("\n== Secure sum with dropouts ==")
    salaries = [52000, 61000, 48000, 75000, 58000, 90000, 43000]
    print("true sum:", sum(salaries), "| additive:", additive_secure_sum(salaries))
    r = shamir_secure_sum(salaries, 4, dropped_after=[0, 3, 6])
    print("shamir with 3 evaluators dropped:", r.total, f"({r.messages} messages)")
    r = shamir_secure_sum(salaries, 4, dropped_before=[2])
    print("party 2 never contributed:", r.total, "expected", sum(salaries) - salaries[2])

    print("\n== BGW mean/variance ==")
    t0 = time.perf_counter()
    mean, var, net = secure_mean_variance(salaries, t=3)
    print(f"mean={mean:.1f} var={var:.1f} messages={net.messages} rounds={net.rounds} "
          f"({time.perf_counter() - t0:.3f}s)")
    eng = BGW(n=7, t=3)
    prod = eng.mul(eng.input(1234), eng.input(5678))
    print("1234 * 5678 =", eng.open(prod))


if __name__ == "__main__":
    main()
