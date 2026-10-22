"""BB84 quantum key distribution simulator with eavesdropper detection."""

from .photons import encode, measure, bits_to_angles
from .channel import Channel, InterceptResend
from .protocol import BB84Config, BB84Result, run_bb84, sift, estimate_qber
from .reconciliation import cascade, parity
from .privacy import (
    binary_entropy,
    secret_fraction,
    secure_key_length,
    toeplitz_hash,
    privacy_amplify,
)
from .analysis import (
    detection_probability,
    expected_qber_intercept,
    qber_threshold,
    hoeffding_margin,
    sweep_eavesdropping,
)

__all__ = [
    "encode", "measure", "bits_to_angles", "Channel", "InterceptResend",
    "BB84Config", "BB84Result", "run_bb84", "sift", "estimate_qber",
    "cascade", "parity", "binary_entropy", "secret_fraction",
    "secure_key_length", "toeplitz_hash", "privacy_amplify",
    "detection_probability", "expected_qber_intercept", "qber_threshold",
    "hoeffding_margin", "sweep_eavesdropping",
]
