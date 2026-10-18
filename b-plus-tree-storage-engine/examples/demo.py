"""Demo: build, query, bulk-load, cache-profile and persist a B+ tree."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bplus_engine import (BPlusTree, BufferPool, KVStore, bulk_load, dumps, encode_key,
                          loads, max_keys_for_page, record_trace)


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0


def main() -> None:
    rng = random.Random(2024)
    n = 100_000
    keys = rng.sample(range(10**7), n)

    print("== Insertion vs bulk loading (max_keys=64, n=100000) ==")
    inc = BPlusTree(64)
    _, t_inc = timed(lambda: [inc.insert(k, k) for k in keys])
    srt = sorted(keys)
    bulk, t_bulk = timed(lambda: bulk_load(((k, k) for k in srt), 64, 0.9))
    for name, tr, t in (("insert", inc, t_inc), ("bulk", bulk, t_bulk)):
        tr.check()
        s = tr.stats()
        print(f"{name:7s} time={t:6.3f}s height={s['height']} leaves={s['leaves']:5d} fill={s['leaf_fill']:.2f}")

    print("\n== Range scan ==")
    lo, hi = 2_000_000, 2_050_000
    got, t_scan = timed(lambda: list(bulk.range(lo, hi)))
    assert [k for k, _ in got] == [k for k in srt if lo <= k < hi]
    print(f"range [{lo},{hi}) -> {len(got)} keys in {t_scan*1000:.2f} ms (verified against sorted list)")
    print("floor(5000000) =", bulk.floor(5_000_000)[0], " ceiling(5000000) =", bulk.ceiling(5_000_000)[0])

    print("\n== Buffer pool hit rate for 5000 random lookups ==")
    trace = record_trace(bulk, lambda tr: [tr.get(rng.choice(keys)) for _ in range(5000)])
    total_pages = bulk.stats()["leaves"] + bulk.stats()["internal"]
    for cap in (4, 16, 64, 256, 1024):
        pool = BufferPool(cap)
        pool.replay(trace)
        print(f"frames={cap:5d} ({cap/total_pages:6.1%} of {total_pages} pages) hit rate={pool.hit_rate:.3f} disk reads={pool.misses}")

    print("\n== Deletion keeps invariants ==")
    victims = rng.sample(keys, 60_000)
    _, t_del = timed(lambda: [bulk.delete(k) for k in victims])
    bulk.check()
    print(f"deleted 60000 keys in {t_del:.3f}s; size={len(bulk)} height={bulk.height()} fill={bulk.stats()['leaf_fill']:.2f}")

    print("\n== Page serialisation ==")
    m = max_keys_for_page(4096, 8)
    small = bulk_load(((k, k.to_bytes(8, 'big')) for k in srt[:20000]), m, 0.9)
    blob = dumps(small, 4096)
    back = loads(blob)
    assert list(back.items()) == list(small.items())
    print(f"max_keys for 4 KiB pages with 8-byte values: {m}")
    print(f"20000 entries -> {len(blob)//4096} pages, {len(blob)/1024:.0f} KiB; round trip OK")

    print("\n== Key-value store with prefix scan ==")
    st = KVStore(8)
    for w in ["alpha", "alphabet", "alpine", "beta", "betray", "gamma"]:
        st.put(encode_key(w), w.encode())
    print("prefix 'alp':", [v.decode() for _, v in st.scan_prefix("alp")])
    st.batch([("put", encode_key("delta"), b"delta"), ("del", encode_key("beta"), None)])
    print("all keys:", [v.decode() for _, v in st.scan()])


if __name__ == "__main__":
    main()
