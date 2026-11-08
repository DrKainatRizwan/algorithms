"""Quasi-Newton, nonlinear CG and trust-region optimisers implemented from scratch."""
from .functions import Problem, rosenbrock, quadratic, powell_singular, extended_rosenbrock, beta_function, logistic_regression, numerical_gradient
from .linesearch import backtracking, strong_wolfe
from .quasi_newton import Result, bfgs, dfp, sr1, lbfgs, LBFGSMemory
from .nonlinear_cg import nonlinear_cg
from .trust_region import steihaug_cg, dogleg, trust_region
from .benchmark import compare

__all__ = [
    "Problem", "rosenbrock", "quadratic", "powell_singular", "extended_rosenbrock",
    "beta_function", "logistic_regression", "numerical_gradient", "backtracking",
    "strong_wolfe", "Result", "bfgs", "dfp", "sr1", "lbfgs", "LBFGSMemory",
    "nonlinear_cg", "steihaug_cg", "dogleg", "trust_region", "compare",
]
