"""Proximal and coordinate methods for sparse regression, written from scratch."""
from .problems import (make_sparse_regression, lasso_objective, lambda_max,
                       kkt_violation, lipschitz_constant, SolverResult)
from .proximal import (soft_threshold, prox_group_l2, prox_elastic_net,
                       project_l1_ball, prox_box)
from .first_order import proximal_gradient, ista, fista, group_lasso
from .coordinate import coordinate_descent
from .admm import generalized_lasso_admm, lasso_admm, difference_matrix
from .path import lasso_path, cross_validate, support_metrics

__all__ = [
    "make_sparse_regression", "lasso_objective", "lambda_max", "kkt_violation",
    "lipschitz_constant", "SolverResult", "soft_threshold", "prox_group_l2",
    "prox_elastic_net", "project_l1_ball", "prox_box", "proximal_gradient",
    "ista", "fista", "group_lasso", "coordinate_descent",
    "generalized_lasso_admm", "lasso_admm", "difference_matrix", "lasso_path",
    "cross_validate", "support_metrics",
]
