"""Optimal transport toolkit: exact solver, Sinkhorn, sliced and barycenters."""
from .barycenter import barycentric_map, displacement_interpolation_1d, entropic_barycenter
from .cost import as_histogram, grid_cost, pairwise_cost, transport_cost
from .exact import emd, plan_is_feasible, wasserstein_exact
from .one_dimensional import (monotone_plan, quantile_function, sliced_wasserstein,
                              wasserstein_1d, wasserstein_1d_barycenter)
from .sinkhorn import (SinkhornResult, regularised_ot_value, sinkhorn, sinkhorn_classic,
                       sinkhorn_divergence, sinkhorn_eps_scaling)

__all__ = [n for n in dir() if not n.startswith("_")]
