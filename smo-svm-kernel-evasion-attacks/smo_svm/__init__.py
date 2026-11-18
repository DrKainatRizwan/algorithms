"""Kernel support vector machines trained with SMO, plus evasion attacks."""

from .kernels import Kernel, linear_kernel, polynomial_kernel, rbf_kernel, make_kernel
from .smo import SVC
from .multiclass import OneVsRestSVC, OneVsOneSVC
from .attacks import deepfool, pgd_linf, robust_accuracy, minimum_distance
from .datasets import make_blobs, make_moons, make_circles, make_xor, train_test_split
from .tuning import kfold_indices, cross_val_score, grid_search

__all__ = [
    "Kernel", "linear_kernel", "polynomial_kernel", "rbf_kernel", "make_kernel",
    "SVC", "OneVsRestSVC", "OneVsOneSVC",
    "deepfool", "pgd_linf", "robust_accuracy", "minimum_distance",
    "make_blobs", "make_moons", "make_circles", "make_xor", "train_test_split",
    "kfold_indices", "cross_val_score", "grid_search",
]
