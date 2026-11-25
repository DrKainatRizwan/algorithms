"""PRNG internals: reference generators, state-recovery attacks and statistical tests."""

from .generators import MT19937, Xorshift128Plus, LCG, JavaRandom, lsb_stream, bit_matrix
from .mt_attacks import (
    temper,
    untemper,
    clone_mt19937,
    rewind_twist,
    recover_seed,
)
from .lcg_attacks import (
    recover_modulus,
    recover_multiplier_increment,
    recover_lcg_parameters,
    recover_truncated_state,
    recover_java_random_state,
)
from .gf2 import solve_gf2, gf2_rank
from .xorshift_attacks import recover_xorshift128plus
from .stats import (
    berlekamp_massey,
    monobit_test,
    runs_test,
    block_frequency_test,
    byte_chi_square_test,
    serial_test,
    autocorrelation_test,
    spectral_test,
    linear_complexity_test,
    run_battery,
)

__all__ = [
    "MT19937", "Xorshift128Plus", "LCG", "JavaRandom", "lsb_stream", "bit_matrix",
    "temper", "untemper", "clone_mt19937", "rewind_twist", "recover_seed",
    "recover_modulus", "recover_multiplier_increment", "recover_lcg_parameters",
    "recover_truncated_state", "recover_java_random_state",
    "solve_gf2", "gf2_rank", "recover_xorshift128plus",
    "berlekamp_massey", "monobit_test", "runs_test", "block_frequency_test",
    "byte_chi_square_test", "serial_test", "autocorrelation_test", "spectral_test",
    "linear_complexity_test", "run_battery",
]
