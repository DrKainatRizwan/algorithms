# Kalman Filter and Particle Filter Object Tracking Simulator

## Overview
From-scratch NumPy implementations of the main recursive Bayesian trackers: the linear Kalman filter (with Joseph-form update and RTS smoother), the extended Kalman filter, the unscented Kalman filter and a bootstrap particle filter with systematic/multinomial resampling. A simulator generates constant-velocity targets with position sensors and maneuvering (coordinated-turn) targets observed by a range/bearing radar, and NEES/NIS metrics check filter consistency.

## How it works
- **Kalman filter**: predict `x = Fx`, `P = FPFᵀ + Q`; update with gain `K = PHᵀS⁻¹`, `S = HPHᵀ + R`. The Joseph form keeps `P` symmetric positive definite.
- **RTS smoother**: backward pass `C_k = P_k Fᵀ P⁻¹_{k+1|k}` refining each estimate with future data.
- **EKF**: linearises `f` and `h` about the current estimate (central-difference Jacobians unless supplied); bearing innovations are angle-wrapped.
- **UKF**: scaled unscented transform with 2n+1 sigma points propagated through the nonlinear maps; circular mean for the bearing component.
- **Particle filter**: propagate N particles through the stochastic dynamics, weight by the measurement likelihood, resample (systematic) when ESS < N/2.
- **Consistency**: NEES (expected = state dim) and NIS (expected = measurement dim).

## Complexity
| Filter | Time per step | Space |
|---|---|---|
| KF / EKF | O(n³ + m³) | O(n²) |
| UKF | O(n·(n³ + cost of f,h)) | O(n²) |
| Particle filter | O(N·(cost of f,h)), resampling O(N) | O(N·n) |

## Usage
```python
import numpy as np
from tracking_filters import *

truth, zs, F, Q, H, R = simulate_cv(200, 1.0, 0.05, 2.0, seed=1)
kf = KalmanFilter(F, Q, H, R, np.zeros(4), np.eye(4) * 10)
xs, Ps, xp, Pp, ys, Ss = kf.run(zs)
smoothed, _ = rts_smoother(F, xs, Ps, xp, Pp)
print(rmse(xs[:, :2], truth[:, :2]))
```
Run `python examples/demo.py` for the full comparison and `python -m pytest -q` for the tests.

## Results (from `examples/demo.py`)
Linear CV target, 200 steps, measurement σ = 2:

| Method | Position RMSE |
|---|---|
| Raw measurements | 2.008 |
| Kalman filter | 1.322 |
| RTS smoother | 0.769 |

NEES/dim = 1.12 and NIS/dim = 0.98 (≈1 means consistent).

Maneuvering target with range/bearing radar, 150 steps, two turn-rate changes:

| Method | Position RMSE | Velocity RMSE | Time |
|---|---|---|---|
| Raw polar→xy | 2.549 | – | – |
| EKF | 1.758 | 0.886 | 0.03 s |
| UKF | 1.757 | 0.881 | 0.03 s |
| PF (1000 particles) | 1.753 | 0.941 | 0.45 s |

At this noise level the bearing nonlinearity is mild, so all three agree; the particle filter costs roughly 15× more.

## References
- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", 1960.
- H. E. Rauch, F. Tung, C. T. Striebel, "Maximum Likelihood Estimates of Linear Dynamic Systems", 1965.
- S. Julier, J. Uhlmann, "Unscented Filtering and Nonlinear Estimation", Proc. IEEE, 2004.
- N. Gordon, D. Salmond, A. Smith, "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", 1993.
- Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, 2001.

---

**Author:** Dr. Kainat Rizwan
