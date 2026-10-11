"""Graph neural network message passing implemented from scratch with NumPy."""
from .analysis import (dirichlet_energy, label_agreement_after_hops,
                       mean_pairwise_distance, smoothing_curve)
from .graph import (Graph, add_self_loops, edge_homophily, gcn_normalize,
                    mean_normalize, split_masks, stochastic_block_model)
from .layers import GATLayer, GCNLayer, SAGELayer
from .model import GNN, Adam, cross_entropy, softmax
from .train import History, accuracy, evaluate, gradient_check, train

__all__ = [
    "Graph", "stochastic_block_model", "gcn_normalize", "mean_normalize",
    "add_self_loops", "split_masks", "edge_homophily", "GCNLayer", "SAGELayer",
    "GATLayer", "GNN", "Adam", "cross_entropy", "softmax", "History", "train",
    "evaluate", "accuracy", "gradient_check", "dirichlet_energy",
    "mean_pairwise_distance", "smoothing_curve", "label_agreement_after_hops",
]
