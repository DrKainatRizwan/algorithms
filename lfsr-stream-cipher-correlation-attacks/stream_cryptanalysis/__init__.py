"""LFSR-based stream cipher cryptanalysis: correlation, fast correlation and algebraic attacks."""
from .algebraic import algebraic_attack, unknown_count
from .correlation import (
    correlation_attack,
    correlation_scores,
    geffe_attack,
    required_length,
    walsh_hadamard,
)
from .fast_correlation import fast_correlation_attack, parity_checks
from .generators import FilterGenerator, GeffeGenerator, ThresholdGenerator, add_noise
from .lfsr import LFSR, berlekamp_massey, lfsr_from_connection, recover_state

__all__ = [
    "LFSR", "berlekamp_massey", "lfsr_from_connection", "recover_state",
    "GeffeGenerator", "ThresholdGenerator", "FilterGenerator", "add_noise",
    "walsh_hadamard", "correlation_scores", "correlation_attack", "geffe_attack",
    "required_length", "parity_checks", "fast_correlation_attack",
    "algebraic_attack", "unknown_count",
]
