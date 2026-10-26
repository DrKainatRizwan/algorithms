"""Influence functions and implicit gradients for logistic regression.

For a model ``w*`` minimising a twice-differentiable objective with Hessian
``H`` the first-order response to a perturbation of the training set is
``dw = -H^{-1} dg`` where ``dg`` is the change in the training gradient.  We
use this to (a) rank training points by their effect on validation loss and
(b) differentiate the validation loss with respect to the features of an
injected poison point (Koh & Liang 2017; Biggio et al. 2012).
"""

from __future__ import annotations

import numpy as np

from .model import LogisticRegression, sigmoid


def validation_gradient(model: LogisticRegression, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Gradient of the summed validation loss at the fitted parameters."""
    return model.grad(X_val, y_val, reg=False)


def removal_influence(model: LogisticRegression, X: np.ndarray, y: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """First-order change in validation loss if training point ``i`` were removed.

    ``dL_i ~= g_val^T H^{-1} g_i``; positive values mean removal hurts the model.
    """
    H = model.hessian(X)
    g_val = validation_gradient(model, X_val, y_val)
    v = np.linalg.solve(H, g_val)
    return model.per_sample_grad(X, y) @ v


def flip_influence(model: LogisticRegression, X: np.ndarray, y: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """First-order change in validation loss if the label of point ``i`` were flipped.

    Positive values mean the flip increases validation loss (good for an attacker).
    """
    H = model.hessian(X)
    g_val = validation_gradient(model, X_val, y_val)
    v = np.linalg.solve(H, g_val)
    g_orig = model.per_sample_grad(X, y)
    g_flip = model.per_sample_grad(X, 1 - y)
    # w_new - w = -H^{-1}(g_flip - g_orig);  dL = g_val . (w_new - w)
    return -(g_flip - g_orig) @ v


def poison_input_gradient(model: LogisticRegression, X_all: np.ndarray, x_c: np.ndarray, y_c: float,
                          X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Gradient of validation loss w.r.t. the features of a poison point.

    ``model`` must be fitted on the poisoned set whose design matrix (including
    the poison point) is ``X_all``.  By implicit differentiation::

        dL_val/dx_c = - J^T H^{-1} g_val,   J = d(grad_w l(x_c, y_c)) / dx_c

    where ``grad_w l = (sigma(w.x) - y) x`` gives
    ``J = (sigma - y) [I; 0] + sigma' * x_b w_x^T``.
    """
    w = model.w
    d = x_c.shape[0]
    xb = np.append(x_c, 1.0)
    s = float(sigmoid(xb @ w))
    J = np.zeros((d + 1, d))
    J[:d, :d] = (s - y_c) * np.eye(d)
    J += s * (1.0 - s) * np.outer(xb, w[:d])
    H = model.hessian(X_all)
    g_val = validation_gradient(model, X_val, y_val)
    return -J.T @ np.linalg.solve(H, g_val)
