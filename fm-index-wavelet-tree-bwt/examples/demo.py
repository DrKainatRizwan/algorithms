"""Demo: build an FM-index over a synthetic genome and compare with naive search."""
import os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compressed_index import FMIndex, bwt, inverse_bwt, naive_locate

r = random.Random(2024)
print("BWT(banana) =", bwt(b"banana"))
text = bytes(r.choice(b"ACGT") for _ in range(20000))
assert inverse_bwt(bwt(text[:2000])) == text[:2000]
print("inverse BWT round-trip ok")
for k in (1, 8, 32):
    t0 = time.perf_counter(); fm = FMIndex(text, sa_sample=k); bt = time.perf_counter() - t0
    pats = [text[s:s + 8] for s in (r.randrange(len(text) - 8) for _ in range(200))]
    t0 = time.perf_counter(); res = [fm.locate(p) for p in pats]; lt = time.perf_counter() - t0
    t0 = time.perf_counter(); ref = [naive_locate(text, p) for p in pats]; nt = time.perf_counter() - t0
    assert res == ref
    rep = fm.size_report()
    print(f"sample={k:>2} build={bt:.2f}s locate200={lt*1000:.1f}ms naive200={nt*1000:.1f}ms "
          f"sa_samples={rep['sa_samples']}B full_sa={rep['full_sa']}B")
print("count('ACGTACGT') =", fm.count(b"ACGTACGT"), "extract(100,20) =", fm.extract(100, 20))
