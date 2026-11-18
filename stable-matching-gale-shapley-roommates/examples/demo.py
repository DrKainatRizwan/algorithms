"""Demo: stable marriage, lattice, roommates and hospitals/residents."""
import math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_matching import *
from stable_matching.preferences import rank_table

t0 = time.time()
men, women = random_marriage_instance(8, seed=20)
mo = gale_shapley(men, women)
wo = gale_shapley(women, men)
print("Man-optimal  :", mo.match, "proposals", mo.proposals, "stable", is_stable(men, women, mo.match))
allm = enumerate_stable_matchings(men, women)
print("Stable matchings in instance:", len(allm), " rotations:", len(find_all_rotations(men, women)))
print("Egalitarian  :", egalitarian_matching(men, women))
print("Min regret   :", minimum_regret_matching(men, women))
print("Sex-equal    :", sex_equal_matching(men, women))
assert allm == sorted(brute_force_marriage(men, women))
print("Brute-force check passed")

print("\nProposals (men-proposing vs women-proposing), mean of 20 random instances:")
for n, (a, b) in proposal_statistics([50, 100, 200, 400], 20).items():
    print(f"  n={n:4d}  men {a:8.1f}  women {b:8.1f}  n*ln(n)={n*math.log(n):8.1f}")

print("\nRoommates: P(stable matching exists), 200 random instances")
for n in (6, 10, 20, 40):
    print(f"  n={n:3d}  {roommates_existence_rate(n, 200):.2f}")

print("\nMean number of stable matchings (random, 20 trials):")
for n in (6, 8, 10, 12):
    print(f"  n={n:3d}  {count_stable_matchings(n, 20):.2f}")

caps = [2, 1, 3, 2]
res, hosp = random_hr_instance(10, caps, 1)
a = hospitals_residents(res, hosp, caps, True)
b = hospitals_residents(res, hosp, caps, False)
print("\nHospitals/residents resident-optimal:", a)
print("Hospital-optimal                    :", b)
print("blocking pairs:", len(hr_blocking_pairs(res, hosp, caps, a)), len(hr_blocking_pairs(res, hosp, caps, b)))
print(f"\nElapsed {time.time()-t0:.1f}s")
