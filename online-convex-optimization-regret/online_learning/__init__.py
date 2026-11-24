"""Online convex optimisation: learners, domains, losses and regret analysis."""
from .domains import Ball, Box, Domain, L1Ball, Simplex, project_simplex
from .environments import (alternating_experts, expert_losses, ftl_adversary,
                           logistic_stream, regression_stream, stochastic_linear)
from .learners import (AdaGrad, ExponentiatedGradient, FollowThePerturbedLeader,
                       FollowTheLeader, FTRL, Hedge, Learner, OnlineGradientDescent,
                       OnlineNewtonStep)
from .losses import LinearLoss, LogisticLoss, Loss, SquaredLoss, aggregate
from .regret import (OnlineRun, best_in_hindsight, growth_exponent, hedge_eta,
                     hedge_regret_bound, minimise_over_domain, ogd_regret_bound,
                     regret, regret_curve_linear, run_online,
                     strongly_convex_regret_bound)

__all__ = [n for n in dir() if not n.startswith("_")]
