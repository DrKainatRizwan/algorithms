"""shor_sim: Shor's algorithm, phase estimation and the QFT, simulated from scratch."""
from .analysis import (order_recovery_probability, peak_mass, resource_estimate,
                       split_probability, theoretical_success_floor)
from .modular import (continued_fraction, convergents, gcd, modinv, modpow,
                      multiplicative_order, recover_denominator)
from .period import counting_qubits, outcome_distribution, sample_order
from .qft import qft_fft, qft_gates, qft_matrix
from .qpe import phase_estimation_distribution
from .shor import ShorResult, factors_from_order, shor_factor

__all__ = [
    "ShorResult", "continued_fraction", "convergents", "counting_qubits",
    "factors_from_order", "gcd", "modinv", "modpow", "multiplicative_order",
    "order_recovery_probability", "outcome_distribution", "peak_mass",
    "phase_estimation_distribution", "qft_fft", "qft_gates", "qft_matrix",
    "recover_denominator", "resource_estimate", "sample_order", "shor_factor",
    "split_probability", "theoretical_success_floor",
]
