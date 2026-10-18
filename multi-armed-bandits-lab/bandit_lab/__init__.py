"""Multi-armed bandit laboratory: environments, policies, simulation and theory."""
from .analysis import (bernoulli_kl, exp3_regret_bound, lai_robbins_constant,
                       log_fit_slope, ucb1_regret_bound)
from .environments import (AdversarialBandit, BanditEnvironment, BernoulliBandit,
                           GaussianBandit, SwitchingBandit, alternating_adversary)
from .policies import (EXP3, EpsilonGreedy, Policy, Softmax, ThompsonBernoulli,
                       ThompsonGaussian, UCB1, UCBTuned, SlidingWindowUCB)
from .simulation import RunResult, average_regret, best_arm_rate, run, run_adversarial

__all__ = [
    "AdversarialBandit", "BanditEnvironment", "BernoulliBandit", "GaussianBandit",
    "SwitchingBandit", "alternating_adversary", "EXP3", "EpsilonGreedy", "Policy",
    "Softmax", "SlidingWindowUCB", "ThompsonBernoulli", "ThompsonGaussian", "UCB1", "UCBTuned",
    "RunResult", "average_regret", "best_arm_rate", "run", "run_adversarial",
    "bernoulli_kl", "exp3_regret_bound", "lai_robbins_constant", "log_fit_slope",
    "ucb1_regret_bound",
]
