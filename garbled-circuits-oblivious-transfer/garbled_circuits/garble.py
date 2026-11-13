"""Garbling schemes: classic point-and-permute, free-XOR and half-gates.

* ``classic``   - every XOR/AND gate has a 4-row table (point-and-permute).
* ``free_xor``  - XOR gates cost nothing (Kolesnikov-Schneider); AND has 4 rows.
* ``half_gates``- XOR free and AND costs only 2 ciphertexts (Zahur-Rosulek-Evans).

NOT gates are free in every scheme: the garbler swaps the meaning of the
labels and the evaluator forwards the label unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .circuit import AND, NOT, XOR, Circuit
from .primitives import (LABEL_BYTES, hash_labels, hash_single, lsb,
                         random_label, random_offset)

SCHEMES = ("classic", "free_xor", "half_gates")


@dataclass
class GarbledCircuit:
    """Everything the evaluator receives, plus garbler-only secrets."""
    circuit: Circuit
    scheme: str
    tables: dict[int, tuple[int, ...]]            # gate index -> ciphertexts
    decode: list[int]                              # colour bit of each output's 0-label
    input_labels: dict[int, tuple[int, int]] = field(repr=False, default_factory=dict)

    @property
    def n_ciphertexts(self) -> int:
        return sum(len(t) for t in self.tables.values())

    @property
    def size_bytes(self) -> int:
        """Bytes the garbler must transmit (tables + decoding bits, rounded up)."""
        return self.n_ciphertexts * LABEL_BYTES + (len(self.decode) + 7) // 8

    def encode(self, wire: int, bit: int) -> int:
        """Garbler-side: the label encoding ``bit`` on an input wire."""
        return self.input_labels[wire][bit]


def garble(circuit: Circuit, scheme: str = "half_gates", seed: int = 0) -> GarbledCircuit:
    """Garble ``circuit`` under ``scheme`` using a seeded generator."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    rng = random.Random(seed)
    free = scheme != "classic"
    R = random_offset(rng) if free else 0
    zero: dict[int, int] = {}                      # wire -> label of value 0 (free modes)
    pair: dict[int, tuple[int, int]] = {}          # wire -> (L0, L1)

    def fresh_pair() -> tuple[int, int]:
        l0 = random_label(rng)
        l1 = random_label(rng)
        l1 = (l1 & ~1) | (1 - lsb(l0))             # opposite colour bits
        return l0, l1

    def get(w: int) -> tuple[int, int]:
        return (zero[w], zero[w] ^ R) if free else pair[w]

    def put(w: int, l0: int, l1: int | None = None) -> None:
        if free:
            zero[w] = l0
        else:
            pair[w] = (l0, l1)  # type: ignore[assignment]

    input_labels: dict[int, tuple[int, int]] = {}
    for w in circuit.inputs_a + circuit.inputs_b:
        if free:
            put(w, random_label(rng))
        else:
            put(w, *fresh_pair())
        input_labels[w] = get(w)

    tables: dict[int, tuple[int, ...]] = {}
    for gi, g in enumerate(circuit.gates):
        if g.op == NOT:
            l0, l1 = get(g.a)
            put(g.out, l1, l0) if not free else put(g.out, l1)
        elif g.op == XOR and free:
            put(g.out, zero[g.a] ^ zero[g.b])
        elif scheme == "half_gates":                # AND
            a0, b0 = zero[g.a], zero[g.b]
            pa, pb = lsb(a0), lsb(b0)
            j, j2 = 2 * gi, 2 * gi + 1
            ha0, ha1 = hash_single(a0, j), hash_single(a0 ^ R, j)
            hb0, hb1 = hash_single(b0, j2), hash_single(b0 ^ R, j2)
            tg = ha0 ^ ha1 ^ (R if pb else 0)
            wg = ha0 ^ (tg if pa else 0)
            te = hb0 ^ hb1 ^ a0
            we = hb0 ^ ((te ^ a0) if pb else 0)
            put(g.out, wg ^ we)
            tables[gi] = (tg, te)
        else:                                       # 4-row table (classic any gate / free-xor AND)
            la, lb_ = get(g.a), get(g.b)
            if free:
                o0 = random_label(rng)
                outs = (o0, o0 ^ R)
            else:
                outs = fresh_pair()
            rows = [0] * 4
            for va in (0, 1):
                for vb in (0, 1):
                    v = (va ^ vb) if g.op == XOR else (va & vb)
                    idx = (lsb(la[va]) << 1) | lsb(lb_[vb])
                    rows[idx] = hash_labels(la[va], lb_[vb], gi) ^ outs[v]
            tables[gi] = tuple(rows)
            put(g.out, outs[0], outs[1])
    decode = [lsb(get(o)[0]) for o in circuit.outputs]
    return GarbledCircuit(circuit, scheme, tables, decode, input_labels)


def evaluate(gc: GarbledCircuit, labels: dict[int, int]) -> list[int]:
    """Evaluate with one label per input wire; returns the output bits.

    The evaluator never learns which logical value any label represents,
    except at the outputs where the decoding bits are applied.
    """
    c = gc.circuit
    free = gc.scheme != "classic"
    w: dict[int, int] = dict(labels)
    for gi, g in enumerate(c.gates):
        if g.op == NOT:
            w[g.out] = w[g.a]
        elif g.op == XOR and free:
            w[g.out] = w[g.a] ^ w[g.b]
        elif gc.scheme == "half_gates":
            a, b = w[g.a], w[g.b]
            tg, te = gc.tables[gi]
            wg = hash_single(a, 2 * gi) ^ (tg if lsb(a) else 0)
            we = hash_single(b, 2 * gi + 1) ^ ((te ^ a) if lsb(b) else 0)
            w[g.out] = wg ^ we
        else:
            a, b = w[g.a], w[g.b]
            idx = (lsb(a) << 1) | lsb(b)
            w[g.out] = gc.tables[gi][idx] ^ hash_labels(a, b, gi)
    return [lsb(w[o]) ^ d for o, d in zip(c.outputs, gc.decode)]


__all__ = ["SCHEMES", "GarbledCircuit", "garble", "evaluate", "AND", "XOR"]
