"""xai_toolkit: Shapley-value and LIME style explanations implemented from scratch with NumPy."""
from .data import make_tabular, standardize, train_test_split
from .models import LogisticModel, MLPClassifier, RegressionTree, RandomForest
from .shapley import (ValueFunction, exact_shapley, permutation_shapley,
                      kernel_shap, linear_shap, shapley_kernel_weight)
from .lime import LimeTabular, LimeExplanation
from .gradients import integrated_gradients
from .metrics import (efficiency_gap, deletion_curve, deletion_auc,
                      explanation_stability, rank_correlation, topk_overlap)

__all__ = [n for n in dir() if not n.startswith("_")]
