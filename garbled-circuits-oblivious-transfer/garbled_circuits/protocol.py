"""End-to-end semi-honest two-party computation (Yao's protocol)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .circuit import Circuit
from .garble import GarbledCircuit, evaluate, garble
from .ot import OTReceiver, OTSender
from .primitives import LABEL_BYTES


@dataclass
class ProtocolResult:
    """Outputs and communication costs of one protocol run."""
    outputs: list[int]
    scheme: str
    garbled_bytes: int
    ot_bytes: int
    n_and: int
    n_ot: int

    @property
    def total_bytes(self) -> int:
        return self.garbled_bytes + self.ot_bytes


def run_two_party(circuit: Circuit, a_bits: list[int], b_bits: list[int],
                  scheme: str = "half_gates", seed: int = 0) -> ProtocolResult:
    """Alice garbles; Bob obtains his input labels by OT and evaluates.

    Alice's labels are sent directly; Bob's are delivered via one OT per
    input bit, so Alice never sees ``b_bits`` and Bob only sees one label
    per wire.
    """
    if len(a_bits) != len(circuit.inputs_a) or len(b_bits) != len(circuit.inputs_b):
        raise ValueError("wrong number of input bits")
    gc: GarbledCircuit = garble(circuit, scheme, seed)
    labels = {w: gc.encode(w, bit) for w, bit in zip(circuit.inputs_a, a_bits)}

    rng = random.Random(seed ^ 0x5EED)
    sender = OTSender(rng)
    receiver = OTReceiver(list(b_bits), rng)
    b_msgs = receiver.choose(sender.public)
    pairs = [gc.input_labels[w] for w in circuit.inputs_b]
    got = receiver.recover(sender.respond(b_msgs, pairs))
    labels.update(zip(circuit.inputs_b, got))

    n = len(circuit.inputs_b)
    ot_bytes = 32 + n * 32 + n * 2 * LABEL_BYTES
    garbled_bytes = gc.size_bytes + len(circuit.inputs_a) * LABEL_BYTES
    return ProtocolResult(evaluate(gc, labels), scheme, garbled_bytes, ot_bytes,
                          circuit.count("AND"), n)
