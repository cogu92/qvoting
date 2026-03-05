"""
qvoting.holographic.encoder
----------------------------
Encoding and decoding circuits for the holographic voter.

PHYSICAL PICTURE
----------------
The 9→3→1 majority voter implements a [[9,1,3]] distance-3 repetition code.
We encode the logical qubit as:

    |0⟩_L  →  |000 000 000⟩   (all boundary qubits in |0⟩)
    |1⟩_L  →  |111 111 111⟩   (all boundary qubits in |1⟩)
    |+⟩_L  →  (|000000000⟩ + |111111111⟩) / √2   ← GHZ-like boundary state

Encoding uses a single Hadamard + 8 CNOTs from the logical qubit to the rest.
Decoding is the 9→3→1 majority voter from voters.hierarchical.

HOLOGRAPHIC INTERPRETATION
--------------------------
The boundary (9 inp qubits) encodes the bulk (1 logical qubit).
The voter network defines which boundary subsets can reconstruct the bulk:

    Group 0: inp[0,1,2]  →  v1[0]
    Group 1: inp[3,4,5]  →  v1[1]   →  v2[0]  (bulk)
    Group 2: inp[6,7,8]  →  v1[2]

A boundary region A can reconstruct the bulk iff A contains a
majority of inputs in at least 2 out of 3 groups (min-cut ≥ 2).
This is the discrete Ryu-Takayanagi condition.
"""
from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# ── Group structure of the voter network ─────────────────────────────────────
VOTER_GROUPS = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
}


def encode_logical_qubit(logical_state: str = "plus") -> QuantumCircuit:
    """
    Build the encoding circuit for the holographic voter.

    Maps one logical qubit into 9 boundary qubits using a repetition code:
        |0⟩_L → |000 000 000⟩
        |1⟩_L → |111 111 111⟩
        |+⟩_L → (|000000000⟩ + |111111111⟩)/√2

    Parameters
    ----------
    logical_state : str
        Initial state of the logical qubit.
        "zero"  → encodes |0⟩_L
        "one"   → encodes |1⟩_L
        "plus"  → encodes |+⟩_L  (default, most interesting for holography)
        "minus" → encodes |−⟩_L

    Returns
    -------
    QuantumCircuit
        9-qubit circuit with boundary qubits in the encoded logical state.
        No measurements — the state remains quantum coherent.

    Notes
    -----
    The logical qubit is qubit 0. After encoding, all 9 qubits are entangled.
    The circuit has depth 2 (H + CNOT layer).
    """
    boundary = QuantumRegister(9, "b")
    qc = QuantumCircuit(boundary, name=f"encode_{logical_state}")

    # ── Step 1: Prepare the logical qubit (qubit 0) ─────────────────────────
    if logical_state == "zero":
        pass                        # |0⟩ by default
    elif logical_state == "one":
        qc.x(boundary[0])           # |1⟩
    elif logical_state == "plus":
        qc.h(boundary[0])           # (|0⟩ + |1⟩)/√2
    elif logical_state == "minus":
        qc.x(boundary[0])
        qc.h(boundary[0])           # (|0⟩ − |1⟩)/√2
    else:
        raise ValueError(f"Unknown logical_state: {logical_state!r}. "
                         f"Choose from 'zero', 'one', 'plus', 'minus'.")

    # ── Step 2: Spread to all 8 remaining boundary qubits via CNOT ──────────
    # Result: α|000000000⟩ + β|111111111⟩
    for i in range(1, 9):
        qc.cx(boundary[0], boundary[i])

    return qc


def build_voter_decoder() -> QuantumCircuit:
    """
    Build the decoding circuit: 9 boundary qubits → 1 bulk qubit.

    This is the full 9→3→1 hierarchical majority voter operating as a
    quantum decoder. The circuit does NOT measure between levels —
    level-1 ancillas remain coherent during level-2 computation.

    Returns
    -------
    QuantumCircuit
        13-qubit circuit (9 boundary + 3 intermediate + 1 bulk).
        Single measurement on the bulk qubit at the end.
    """
    boundary = QuantumRegister(9, "b")
    ancilla_l1 = QuantumRegister(3, "v1")  # Level-1 intermediates (coherent)
    bulk = QuantumRegister(1, "v2")        # Bulk (logical) output
    creg = ClassicalRegister(1, "bulk")

    qc = QuantumCircuit(boundary, ancilla_l1, bulk, creg, name="voter_decoder")

    # ── Level 1: three parallel 3-input majority voters ──────────────────────
    # maj(a,b,c) = (a∧b) ⊕ (b∧c) ⊕ (a∧c)  implemented via 3 Toffoli gates
    for group_id, (a, b, c) in enumerate(
        [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    ):
        v = ancilla_l1[group_id]
        qc.ccx(boundary[a], boundary[b], v)
        qc.ccx(boundary[b], boundary[c], v)
        qc.ccx(boundary[a], boundary[c], v)

    # ── Level 2: final majority vote on the three l1 results ─────────────────
    qc.ccx(ancilla_l1[0], ancilla_l1[1], bulk[0])
    qc.ccx(ancilla_l1[1], ancilla_l1[2], bulk[0])
    qc.ccx(ancilla_l1[0], ancilla_l1[2], bulk[0])

    qc.measure(bulk[0], creg[0])
    return qc


def build_decoder_only() -> QuantumCircuit:
    """
    Build only the decoder (voter) part — no encoding, no erasure.

    This unitary-only circuit is safe to fold for ZNE because it
    contains no reset or measurement operations until the final measure.
    Use it to calibrate the decoder noise level independently.

    Returns
    -------
    QuantumCircuit
        13-qubit decoder circuit (9 boundary + 3 v1 + 1 bulk).
    """
    boundary = QuantumRegister(9, "b")
    anc = QuantumRegister(3, "v1")
    bulk = QuantumRegister(1, "v2")
    creg = ClassicalRegister(1, "bulk")
    qc = QuantumCircuit(boundary, anc, bulk, creg, name="decoder_only")

    for group_id, (a, b, c) in enumerate([(0, 1, 2), (3, 4, 5), (6, 7, 8)]):
        v = anc[group_id]
        qc.ccx(boundary[a], boundary[b], v)
        qc.ccx(boundary[b], boundary[c], v)
        qc.ccx(boundary[a], boundary[c], v)

    qc.ccx(anc[0], anc[1], bulk[0])
    qc.ccx(anc[1], anc[2], bulk[0])
    qc.ccx(anc[0], anc[2], bulk[0])

    qc.measure(bulk[0], creg[0])
    return qc


def build_holographic_circuit_zne(
    logical_state: str = "plus",
    qubits_to_erase: list[int] | None = None,
    scale_factor: int = 1,
) -> QuantumCircuit:
    """
    Holographic circuit with ZNE folding applied ONLY to the decoder.

    The encoder (H + CNOTs) and reset operations are NOT folded —
    they cannot be inverted. Only the 12 CCX gates of the voter decoder
    are folded at the requested noise scale.

    For scale_factor = 2k+1:
        decoder → decoder · (decoder_inv · decoder)^k

    Since CCX† = CCX, decoder_inv is the same 12 CCX gates in reverse
    order. The ideal output is preserved while gate noise accumulates
    proportionally to scale_factor.

    Parameters
    ----------
    logical_state : str
        Logical state to encode ('plus', 'zero', 'one', 'minus').
    qubits_to_erase : list[int] | None
        Boundary qubit indices [0..8] to erase before decoding.
    scale_factor : int
        Odd integer >= 1. Decoder is folded to this noise level.

    Returns
    -------
    QuantumCircuit
        13-qubit circuit ready for ZNE execution.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(
            f"scale_factor must be an odd integer >= 1, got {scale_factor}"
        )

    boundary = QuantumRegister(9, "b")
    anc = QuantumRegister(3, "v1")
    bulk = QuantumRegister(1, "v2")
    creg = ClassicalRegister(1, "bulk")
    qc = QuantumCircuit(boundary, anc, bulk, creg)

    # ── Part 1: Encoding (never folded) ─────────────────────────────────
    if logical_state == "zero":
        pass
    elif logical_state == "one":
        qc.x(boundary[0])
    elif logical_state == "plus":
        qc.h(boundary[0])
    elif logical_state == "minus":
        qc.x(boundary[0])
        qc.h(boundary[0])
    else:
        raise ValueError(f"Unknown logical_state: {logical_state!r}")
    for i in range(1, 9):
        qc.cx(boundary[0], boundary[i])
    qc.barrier(label="encoded")

    # ── Part 2: Erasure (never folded — not unitary) ─────────────────────
    if qubits_to_erase:
        for idx in qubits_to_erase:
            if not 0 <= idx <= 8:
                raise ValueError(
                    f"Boundary qubit index must be in [0,8], got {idx}"
                )
            qc.reset(boundary[idx])
        qc.barrier(label="erased")

    # ── Part 3: Decoder — built separately, then folded ──────────────────
    # Build the decoder as a sub-circuit (no measurements, all unitary)
    dec = QuantumCircuit(boundary, anc, bulk, name="decoder")
    for group_id, (a, b, c) in enumerate([(0, 1, 2), (3, 4, 5), (6, 7, 8)]):
        v = anc[group_id]
        dec.ccx(boundary[a], boundary[b], v)
        dec.ccx(boundary[b], boundary[c], v)
        dec.ccx(boundary[a], boundary[c], v)
    dec.ccx(anc[0], anc[1], bulk[0])
    dec.ccx(anc[1], anc[2], bulk[0])
    dec.ccx(anc[0], anc[2], bulk[0])

    # Fold decoder: dec -> dec · (dec_inv · dec)^k
    k = (scale_factor - 1) // 2
    dec_inv = dec.inverse()
    folded = dec.copy()
    for _ in range(k):
        folded.compose(dec_inv, inplace=True)
        folded.compose(dec, inplace=True)

    qc.compose(folded, inplace=True)
    qc.measure(bulk[0], creg[0])
    return qc


def build_holographic_tomography_circuit(
    logical_state: str = "plus",
    qubits_to_erase: list[int] | None = None,
    basis: str = "Z",
) -> QuantumCircuit:
    """
    Holographic circuit with basis rotation before measurement.

    Identical to build_holographic_circuit() but adds a single-qubit
    rotation to the bulk qubit before the final measurement, enabling
    state tomography in the X, Y, or Z basis.

    Parameters
    ----------
    logical_state : str
        Logical state to encode. See encode_logical_qubit().
    qubits_to_erase : list[int] | None
        Boundary qubit indices to erase before decoding.
    basis : str
        Measurement basis for the bulk qubit.
        'Z' — measure directly (default, same as build_holographic_circuit)
        'X' — apply H before measurement  (measures ⟨X⟩)
        'Y' — apply Sdg+H before measurement (measures ⟨Y⟩)

    Returns
    -------
    QuantumCircuit
        Complete 13-qubit circuit with basis rotation on the bulk qubit.
    """
    if basis not in ("Z", "X", "Y"):
        raise ValueError(f"basis must be 'Z', 'X', or 'Y', got {basis!r}")

    boundary = QuantumRegister(9, "b")
    ancilla_l1 = QuantumRegister(3, "v1")
    bulk = QuantumRegister(1, "v2")
    creg = ClassicalRegister(1, "bulk")

    qc = QuantumCircuit(boundary, ancilla_l1, bulk, creg)

    # Encoding
    enc = encode_logical_qubit(logical_state)
    qc.compose(enc, qubits=list(range(9)), inplace=True)
    qc.barrier(label="encoded")

    # Erasure
    if qubits_to_erase:
        for idx in qubits_to_erase:
            if not 0 <= idx <= 8:
                raise ValueError(
                    f"Boundary qubit index must be in [0,8], got {idx}"
                )
            qc.reset(boundary[idx])
        qc.barrier(label=f"erased_{qubits_to_erase}")

    # Decoding (majority voter — same as build_holographic_circuit)
    for group_id, (a, b, c) in enumerate([(0, 1, 2), (3, 4, 5), (6, 7, 8)]):
        v = ancilla_l1[group_id]
        qc.ccx(boundary[a], boundary[b], v)
        qc.ccx(boundary[b], boundary[c], v)
        qc.ccx(boundary[a], boundary[c], v)

    qc.ccx(ancilla_l1[0], ancilla_l1[1], bulk[0])
    qc.ccx(ancilla_l1[1], ancilla_l1[2], bulk[0])
    qc.ccx(ancilla_l1[0], ancilla_l1[2], bulk[0])

    # Basis rotation before measurement
    if basis == "X":
        qc.h(bulk[0])
    elif basis == "Y":
        qc.sdg(bulk[0])
        qc.h(bulk[0])

    qc.measure(bulk[0], creg[0])
    return qc


def build_holographic_circuit(
    logical_state: str = "plus",
    qubits_to_erase: list[int] | None = None,
) -> QuantumCircuit:
    """
    Full holographic circuit: encode → (optional erasure) → decode.

    Parameters
    ----------
    logical_state : str
        Logical state to encode. See encode_logical_qubit().
    qubits_to_erase : list[int] | None
        Indices in [0, 8] of boundary qubits to erase before decoding.
        Erasure is simulated by resetting the qubit to |0⟩.

    Returns
    -------
    QuantumCircuit
        Complete circuit with 13 qubits and 1 classical bit.
    """
    boundary = QuantumRegister(9, "b")
    ancilla_l1 = QuantumRegister(3, "v1")
    bulk = QuantumRegister(1, "v2")
    creg = ClassicalRegister(1, "bulk")

    qc = QuantumCircuit(boundary, ancilla_l1, bulk, creg)

    # ── Encoding ──────────────────────────────────────────────────────────────
    enc = encode_logical_qubit(logical_state)
    qc.compose(enc, qubits=list(range(9)), inplace=True)
    qc.barrier(label="encoded")

    # ── Erasure ───────────────────────────────────────────────────────────────
    if qubits_to_erase:
        for idx in qubits_to_erase:
            if not 0 <= idx <= 8:
                raise ValueError(
                    f"Boundary qubit index must be in [0,8], got {idx}"
                )
            qc.reset(boundary[idx])  # Quantum erasure: collapse to |0⟩
        qc.barrier(label=f"erased_{qubits_to_erase}")

    # ── Decoding (majority voter) ──────────────────────────────────────────────
    # Level 1
    for group_id, (a, b, c) in enumerate([(0, 1, 2), (3, 4, 5), (6, 7, 8)]):
        v = ancilla_l1[group_id]
        qc.ccx(boundary[a], boundary[b], v)
        qc.ccx(boundary[b], boundary[c], v)
        qc.ccx(boundary[a], boundary[c], v)

    # Level 2
    qc.ccx(ancilla_l1[0], ancilla_l1[1], bulk[0])
    qc.ccx(ancilla_l1[1], ancilla_l1[2], bulk[0])
    qc.ccx(ancilla_l1[0], ancilla_l1[2], bulk[0])

    qc.measure(bulk[0], creg[0])
    return qc
