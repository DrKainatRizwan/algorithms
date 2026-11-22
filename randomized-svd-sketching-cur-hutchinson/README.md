# Randomized Numerical Linear Algebra: Sketching, Randomized SVD, CUR and Hutch++

## Overview
From-scratch NumPy implementations of the core tools of randomized numerical linear algebra:

- **Sketches** (`sketching.py`): Gaussian, subsampled randomized Hadamard (with a fast Walsh-Hadamard transform), CountSketch and sparse sign embeddings, plus the Johnson-Lindenstrauss dimension bound.
- **Low-rank approximation** (`rsvd.py`): randomized range finder with power iterations, randomized SVD, blocked adaptive range finder, single-pass (streaming) SVD, and a matrix-free spectral error estimator.
- **Column/row selection** (`cur.py`): column-pivoted QR, interpolative decomposition, CUR with leverage-score, uniform, pivoted and DEIM selection.
- **Least squares** (`regression.py`): sketch-and-solve and sketch-preconditioned CGLS (Blendenpik/LSRN style).
- **Trace estimation** (`trace.py`): Hutchinson and Hutch++.

## How it works
A random sketch `S` with `m = O(d)` rows is a subspace embedding: `||Sx|| ≈ ||x||` for every `x` in a fixed d-dimensional subspace. Randomized SVD multiplies `A` by a Gaussian test matrix to capture its range, orthonormalises (`Q`), and takes the SVD of the small matrix `QᵀA`. Power iterations `(AAᵀ)^q` sharpen slowly decaying spectra. The single-pass variant keeps only `AΩ` and `ΨA` and reconstructs `A ≈ Q (ΨQ)⁺ (ΨA)`. For least squares, a QR of `SA` yields `R` such that `AR⁻¹` has condition number O(1), so CGLS converges in a fixed number of iterations regardless of cond(A). Hutch++ spends a third of its matvecs on a low-rank sketch of the matrix, takes its trace exactly and applies Hutchinson only to the deflated residual.

## Complexity
| Method | Time | Extra space |
|---|---|---|
| Randomized SVD (rank k, oversampling p, q power its) | O(mn(k+p)(2q+1)) | O((m+n)(k+p)) |
| Gaussian / SRHT / CountSketch apply to n×d | O(mnd) / O(nd log n) / O(nnz) | O(mn) / O(n) / O(n) |
| Sketch-preconditioned CGLS | O(nd log n + d³ + nd·iters) | O(d²) |
| Hutchinson / Hutch++ (m matvecs) | m matvecs, error O(1/√m) / O(1/m) | O(nm) |

## Usage
```python
import numpy as np
from randomized_linalg import randomized_svd, decaying_spectrum
rng = np.random.default_rng(0)
A = decaying_spectrum(1500, 1000, "poly", rng)
U, s, Vt = randomized_svd(A, k=20, oversample=10, n_iter=2, rng=rng)
```
Run `python -m pytest -q` for the tests and `python examples/demo.py` for the benchmark.

## Results (from `examples/demo.py`)
Randomized SVD on a 1500×1000 matrix, rank 20 (optimal spectral error 4.762e-02; full SVD 296 ms):

| Power its q | Error | Time |
|---|---|---|
| 0 | 8.341e-02 | 6 ms |
| 1 | 4.990e-02 | 10 ms |
| 2 | 4.773e-02 | 14 ms |
| 4 | 4.762e-02 | 23 ms |

Single-pass SVD error: 1.406e-01. Subspace embeddings (m=400 of R^4096, 10-dim subspace) keep singular values within roughly [0.88, 1.13] for all four sketches; SRHT and CountSketch apply in under 1 ms against 17 ms for Gaussian.
CUR with 12 rows/columns on a noisy rank-8 matrix (best rank-8 error 16.9): uniform 22.0, leverage 21.4, DEIM 20.6, pivoted QR 19.5.
Least squares, 20000×30 with cond 1e8: preconditioned CGLS needs 25 iterations (cond(AR⁻¹)=2.92) versus 1139 for plain CGLS; sketch-and-solve residual is 1.03× optimal.
Trace estimation (n=500, 30 queries): Hutchinson mean relative error 1.09e-01 versus 5.4e-03 for Hutch++.

## References
- Halko, Martinsson, Tropp. *Finding Structure with Randomness*, SIAM Review, 2011.
- Tropp, Yurtsever, Udell, Cevher. *Practical Sketching Algorithms for Low-Rank Matrix Approximation*, SIMAX, 2017.
- Woodruff. *Sketching as a Tool for Numerical Linear Algebra*, 2014.
- Avron, Maymounkov, Toledo. *Blendenpik*, SISC, 2010.
- Mahoney, Drineas. *CUR matrix decompositions for improved data analysis*, PNAS, 2009.
- Meyer, Musco, Musco, Woodruff. *Hutch++: Optimal Stochastic Trace Estimation*, SOSA, 2021.

---

**Author:** Dr. Kainat Rizwan
