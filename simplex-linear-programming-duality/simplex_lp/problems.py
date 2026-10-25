"""Problem generators and classic applications of linear programming."""

from __future__ import annotations

import numpy as np

from .model import INF, LinearProgram
from .solver import solve


def beale_cycling_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Beale's degenerate LP on which Dantzig's rule cycles.

    Returned as standard-form data ``(c, A, b, starting_basis)`` with the
    three slack columns forming the initial (degenerate) basis.
    """
    c = np.array([-0.75, 20.0, -0.5, 6.0, 0.0, 0.0, 0.0])
    A = np.array([[0.25, -8.0, -1.0, 9.0, 1.0, 0.0, 0.0],
                  [0.5, -12.0, -0.5, 3.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    b = np.array([0.0, 0.0, 1.0])
    return c, A, b, [4, 5, 6]


def klee_minty(n: int) -> LinearProgram:
    """Klee-Minty cube: Dantzig's rule visits all 2^n vertices.

    ``max sum_j 10^(n-j) x_j`` subject to
    ``2 sum_{j<i} 10^(i-j) x_j + x_i <= 100^(i-1)``.
    """
    c = np.array([10.0 ** (n - 1 - j) for j in range(n)])
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            A[i, j] = 2.0 * 10.0 ** (i - j)
        A[i, i] = 1.0
    b = np.array([100.0 ** i for i in range(n)])
    return LinearProgram(c, A, b, maximize=True, name=f"klee-minty-{n}")


def production_planning() -> LinearProgram:
    """Textbook product-mix LP: maximise profit under three resource limits."""
    return LinearProgram([3.0, 5.0], [[1, 0], [0, 2], [3, 2]], [4, 12, 18],
                         maximize=True, name="product-mix")


def diet_problem() -> LinearProgram:
    """Stigler-style diet: minimise cost subject to nutrient minimums."""
    cost = [0.18, 0.23, 0.05, 0.40]  # bread, milk, rice, cheese
    # rows: calories, protein, calcium (each as >=, written as <= with negation)
    nutr = np.array([[300, 150, 350, 400],
                     [8, 8, 7, 25],
                     [20, 280, 10, 700]], dtype=float)
    need = np.array([2000, 55, 800], dtype=float)
    return LinearProgram(cost, -nutr, -need, bounds=[(0, 8)] * 4, name="diet")


def transportation(supply, demand, cost) -> LinearProgram:
    """Balanced transportation LP with variables ordered row-major."""
    supply, demand, cost = (np.asarray(v, float) for v in (supply, demand, cost))
    s, d = len(supply), len(demand)
    if abs(supply.sum() - demand.sum()) > 1e-9:
        raise ValueError("transportation problem must be balanced")
    A = np.zeros((s + d, s * d))
    for i in range(s):
        A[i, i * d:(i + 1) * d] = 1.0
    for j in range(d):
        A[s + j, j::d] = 1.0
    return LinearProgram(cost.ravel(), A_eq=A, b_eq=np.concatenate([supply, demand]),
                         name="transportation")


def max_flow_lp(n: int, edges: list[tuple[int, int, float]], s: int, t: int) -> LinearProgram:
    """Max-flow as an LP: variables are edge flows, conservation at inner nodes."""
    m = len(edges)
    c = np.zeros(m)
    A = np.zeros((n, m))
    for k, (u, v, cap) in enumerate(edges):
        A[u, k] -= 1.0
        A[v, k] += 1.0
        if u == s:
            c[k] += 1.0
        if v == s:
            c[k] -= 1.0
    inner = [v for v in range(n) if v not in (s, t)]
    return LinearProgram(c, A_eq=A[inner], b_eq=np.zeros(len(inner)),
                         bounds=[(0.0, cap) for _, _, cap in edges], maximize=True,
                         name="max-flow")


def assignment_lp(cost) -> LinearProgram:
    """Assignment problem; the LP has integral vertices so simplex solves it exactly."""
    cost = np.asarray(cost, float)
    k = cost.shape[0]
    return transportation(np.ones(k), np.ones(k), cost)


def random_feasible_lp(m: int, n: int, seed: int = 0, equality_fraction: float = 0.0) -> LinearProgram:
    """Random bounded feasible LP: ``min c^T x, A x <= b, 0 <= x <= 10``.

    Feasibility is guaranteed by building ``b`` from a hidden interior point.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    x0 = rng.uniform(0.5, 5.0, size=n)
    n_eq = int(equality_fraction * m)
    b = A @ x0
    b[n_eq:] += rng.uniform(0.1, 2.0, size=m - n_eq)
    c = rng.normal(size=n)
    return LinearProgram(c, A_ub=A[n_eq:], b_ub=b[n_eq:], A_eq=A[:n_eq] if n_eq else None,
                         b_eq=b[:n_eq] if n_eq else None, bounds=[(0.0, 10.0)] * n,
                         name=f"random-{m}x{n}")


def solve_matrix_game(payoff) -> tuple[float, np.ndarray, np.ndarray]:
    """Value and optimal mixed strategies of a zero-sum game (row player maximises).

    The row player solves ``max v`` s.t. ``P^T p >= v, sum p = 1, p >= 0``; the
    column player's strategy is read off the dual multipliers.
    """
    P = np.asarray(payoff, float)
    shift = P.min() - 1.0  # make all entries positive
    Q = P - shift
    r, k = Q.shape
    # column player: max sum w  s.t. Q w <= 1, w >= 0 ; value = 1 / sum w
    lp = LinearProgram(np.ones(k), Q, np.ones(r), maximize=True)
    sol = solve(lp)
    v = 1.0 / sol.objective
    col = sol.x * v
    row = -sol.duals * v  # shadow prices of the <= rows are >= 0 for max problems
    row = np.abs(row)
    return float(v + shift), row / row.sum(), col / col.sum()


def l1_regression(X, y) -> np.ndarray:
    """Least-absolute-deviations fit ``min sum |y - X w|`` via LP."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    m, d = X.shape
    c = np.concatenate([np.zeros(d), np.ones(m)])
    A = np.block([[X, -np.eye(m)], [-X, -np.eye(m)]])
    b = np.concatenate([y, -y])
    bounds = [(-INF, INF)] * d + [(0.0, INF)] * m
    sol = solve(LinearProgram(c, A, b, bounds=bounds))
    return sol.x[:d]


def chebyshev_fit(X, y) -> tuple[np.ndarray, float]:
    """Minimax fit ``min max_i |y_i - X_i w|``; returns weights and max error."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    m, d = X.shape
    c = np.concatenate([np.zeros(d), [1.0]])
    ones = np.ones((m, 1))
    A = np.block([[X, -ones], [-X, -ones]])
    b = np.concatenate([y, -y])
    bounds = [(-INF, INF)] * d + [(0.0, INF)]
    sol = solve(LinearProgram(c, A, b, bounds=bounds))
    return sol.x[:d], float(sol.x[d])
