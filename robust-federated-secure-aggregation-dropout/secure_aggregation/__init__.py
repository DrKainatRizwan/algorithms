"""Secure aggregation with dropout recovery and robust federated learning."""

from .crypto import AuthenticationError
from .federated import FLConfig, Dataset, make_dataset, train
from .field import shamir_reconstruct, shamir_share
from .protocol import AbortError, AggregationResult, Client, run_protocol
from .quantize import clip_l2, decode, encode

__all__ = [
    "AbortError", "AggregationResult", "AuthenticationError", "Client",
    "Dataset", "FLConfig", "clip_l2", "decode", "encode", "make_dataset",
    "run_protocol", "shamir_reconstruct", "shamir_share", "train",
]
