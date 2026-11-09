"""Number-theoretic cryptanalysis: factoring, discrete logarithms and RSA attacks."""

from .primes import (
    crt, egcd, is_prime, jacobi, modinv, next_prime, random_prime, sieve, tonelli_shanks,
)
from .factoring import (
    factorize, fermat_factor, pollard_p_minus_1, pollard_rho_brent, trial_division,
)
from .quadratic_sieve import quadratic_sieve
from .discrete_log import baby_step_giant_step, pohlig_hellman, pollard_rho_dlog
from .index_calculus import index_calculus
from .rsa_attacks import (
    RSAKey, common_modulus_attack, generate_rsa, hastad_broadcast, integer_root,
    shared_prime_attack, wiener_attack,
)

__all__ = [
    "crt", "egcd", "is_prime", "jacobi", "modinv", "next_prime", "random_prime", "sieve",
    "tonelli_shanks", "factorize", "fermat_factor", "pollard_p_minus_1", "pollard_rho_brent",
    "trial_division", "quadratic_sieve", "baby_step_giant_step", "pohlig_hellman",
    "pollard_rho_dlog", "index_calculus", "RSAKey", "common_modulus_attack", "generate_rsa",
    "hastad_broadcast", "integer_root", "shared_prime_attack", "wiener_attack",
]
