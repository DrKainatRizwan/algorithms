# Kernel SVM with SMO and Evasion Attacks

A from-scratch NumPy implementation of the soft-margin kernel support vector machine trained with Platt's Sequential Minimal Optimisation (SMO), together with multiclass wrappers, cross-validated grid search and gradient-based evasion attacks (DeepFool-style minimum-norm and L-infinity PGD) that work for any kernel through analytic kernel gradients.

## Package layout

| Module | Contents |
|---|---|
| `smo_svm/kernels.py` | Linear, polynomial and RBF kernels with gradients w.r.t. the input |
| `smo_svm/smo.py` | Binary `SVC`: SMO solver, primal/dual objectives, KKT check |
| `smo_svm/multiclass.py` | One-vs-rest and one-vs-one wrappers |
| `smo_svm/attacks.py` | `deepfool`, `pgd_linf`, `robust_accuracy`, `minimum_distance` |
| `smo_svm/tuning.py` | k-fold splits, cross-validation, grid search |
| `smo_svm/datasets.py` | Seeded blobs, moons, circles, XOR |

## How it works

The dual problem is

    max_a  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i, x_j)
    s.t.   0 <= a_i <= C,  sum_i a_i y_i = 0

SMO repeatedly picks two multipliers and solves the two-variable sub-problem in closed form, which keeps the equality constraint satisfied. The update is `a2' = clip(a2 + y2 (E1 - E2) / eta)` with `eta = K11 + K22 - 2 K12`, where `E` is the prediction error. The second index is chosen to maximise `|E1 - E2|` among non-bound points, falling back to a randomised sweep over non-bound and then all points. An error cache is updated incrementally after each step. The degenerate `eta <= 0` case evaluates the objective at both ends of the feasible segment.

The decision function is `f(x) = sum_i a_i y_i k(x_i, x) + b`, so its gradient is `sum_i a_i y_i grad_x k(x_i, x)`. This drives the attacks:

* **DeepFool** linearises `f` and steps `-f/||grad f||^2 * grad f` until the sign flips. For a linear kernel it lands exactly on the hyperplane (distance `|f|/||w||`).
* **PGD (L-inf)** takes signed-gradient steps on the margin and projects back onto the eps-ball.

## Complexity

* Gram matrix: O(n^2 d) time, O(n^2) memory.
* SMO: each step is O(n) (error-cache update); empirically between O(n) and O(n^2.2) steps in total.
* Prediction: O(|SV| d) per point. Attack iteration: O(|SV| d) per gradient.

## Usage

```python
from smo_svm import SVC, make_moons, train_test_split, deepfool

X, y = make_moons(300, seed=0)
Xa, Xb, ya, yb = train_test_split(X, y)
model = SVC(C=5.0, kernel="rbf", gamma=2.0).fit(Xa, ya)
print(model.score(Xb, yb), len(model.support_))
adv = deepfool(model, Xb[0])
```

## Results

From `python examples/demo.py` (seeded; a few seconds in total):

| Dataset | linear | poly (deg 3) | RBF (gamma 2) |
|---|---|---|---|
| moons | 0.889 | 1.000 | 1.000 |
| circles | 0.556 | 0.989 | 1.000 |
| xor | 0.722 | 1.000 | 0.989 |

Optimality on moons (RBF, C=5, tol 1e-4): dual 56.9240, primal 56.9251, maximum KKT violation 0.0001 after 1095 SMO steps.

Robust accuracy under L-inf PGD (blobs, std 1.2) at eps = 0 / 0.5 / 1 / 2:

* linear: 0.978 / 0.967 / 0.900 / 0.511
* RBF (gamma 0.2): 0.978 / 0.967 / 0.889 / 0.722

The median minimum L2 perturbation was 3.05 for the linear model; for the RBF model it was 15.5, because the kernel response decays away from the data and the boundary is reached only far out.

## Tests

`python -m pytest -q` runs 20 deterministic tests covering kernel PSD-ness and gradients, dual feasibility, KKT conditions, the primal-dual gap, non-linear separability, multiclass, attack correctness and cross-validation.

## References

* J. Platt, "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines", MSR-TR-98-14, 1998.
* C. Cortes and V. Vapnik, "Support-Vector Networks", Machine Learning 20, 1995.
* S. Moosavi-Dezfooli, A. Fawzi, P. Frossard, "DeepFool", CVPR 2016.
* A. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.
* B. Biggio et al., "Evasion Attacks against Machine Learning at Test Time", ECML PKDD 2013.

---

**Author:** Dr. Kainat Rizwan
