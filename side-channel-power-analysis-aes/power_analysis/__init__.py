"""Side-channel power analysis of AES-128: DPA, CPA and masking."""
from .aes import SBOX, encrypt_block, expand_key, invert_key_schedule
from .attacks import (cpa_attack, dpa_attack, dpa_attack_multibit,
                      recover_key, second_order_cpa)
from .leakage import TraceSet, simulate_masked_traces, simulate_traces
from .metrics import key_rank, success_curve, traces_to_success

__all__ = [
    "SBOX", "encrypt_block", "expand_key", "invert_key_schedule",
    "cpa_attack", "dpa_attack", "dpa_attack_multibit", "recover_key",
    "second_order_cpa", "TraceSet", "simulate_masked_traces",
    "simulate_traces", "key_rank", "success_curve", "traces_to_success",
]
