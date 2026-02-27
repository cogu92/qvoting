"""
qvoting.mitigation.zne
-----------------------
Zero-Noise Extrapolation (ZNE) via digital gate folding.

Two modes:
  • apply_zne()     — simulation-only, takes a NoiseModel (original API)
  • zne_hardware()  — works on any backend (Aer or IBM hardware) using
                      circuit-level folding G→G(G†G)^k for scale factors
                      λ = 1, 3, 5, ...
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction


# ══════════════════════════════════════════════════════════════════════════
# Public API — hardware-compatible ZNE
# ══════════════════════════════════════════════════════════════════════════

def fold_circuit(qc: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Scale noise in *qc* by *scale_factor* using circuit-level folding.

    For scale_factor = 2k+1 (must be odd ≥ 1), the circuit is transformed:
      C  →  C · (C† · C)^k

    This leaves the ideal output unchanged (C†C = I) while accumulating
    noise proportional to λ = scale_factor.

    Parameters
    ----------
    qc : QuantumCircuit
        Original circuit (must not contain mid-circuit measurements
        except at the very end).
    scale_factor : int
        Odd integer ≥ 1.  scale_factor=1 → original circuit (no folding).

    Returns
    -------
    QuantumCircuit
        Folded circuit with same register structure as *qc*.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(f"scale_factor must be an odd integer ≥ 1, got {scale_factor}")
    if scale_factor == 1:
        return qc.copy()

    # Strip measurements for the body (we'll re-add at the end)
    qc_body = _strip_measurements(qc)
    qc_inv  = qc_body.inverse()

    k = (scale_factor - 1) // 2
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    folded.compose(qc_body, inplace=True)
    for _ in range(k):
        folded.compose(qc_inv,  inplace=True)
        folded.compose(qc_body, inplace=True)

    # Re-append original measurements
    for instr in qc.data:
        if instr.operation.name == "measure":
            q_idx = [qc.find_bit(q).index for q in instr.qubits]
            c_idx = [qc.find_bit(c).index for c in instr.clbits]
            folded.measure(q_idx, c_idx)

    return folded


def zne_hardware(
    qc: QuantumCircuit,
    backend,
    shots: int = 1024,
    scale_factors: Optional[List[int]] = None,
    observable_fn: Optional[Callable[[Dict[str, int], int], float]] = None,
    extrapolation: str = "linear",
) -> Tuple[float, List[float], List[int]]:
    """
    Zero-Noise Extrapolation on any backend (Aer or IBM hardware).

    Runs *qc* folded at each scale factor in *scale_factors*, evaluates
    an observable at each noise level, then extrapolates to λ=0.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to execute. Must end with measurements.
    backend : AerSimulator | IBMBackend
        Target backend. For IBM hardware, SamplerV2 is used automatically
        via ``execute_circuit``.
    shots : int
        Shots per noise level.
    scale_factors : list[int], optional
        Odd integers for folding: default [1, 3, 5].
    observable_fn : callable, optional
        fn(counts, shots) → float in [-1, +1].
        Default: ⟨Z⟩ on the first classical bit
        (maps P(0)→+1, P(1)→-1, i.e. 2·P(0)−1).
    extrapolation : {"linear", "richardson"}
        Extrapolation method. ``"linear"`` fits a degree-1 polynomial;
        ``"richardson"`` uses Richardson extrapolation with the first
        two scale factors.

    Returns
    -------
    (E_zne, expectation_values, scale_factors_used)
        E_zne : float — extrapolated zero-noise expectation value.
        expectation_values : list[float] — ⟨O⟩ at each scale factor.
        scale_factors_used : list[int] — the actual scale factors run.
    """
    from qvoting.core.execution import execute_circuit

    if scale_factors is None:
        scale_factors = [1, 3, 5]
    for sf in scale_factors:
        if sf % 2 == 0:
            raise ValueError(f"All scale_factors must be odd; got {sf}")

    if observable_fn is None:
        def observable_fn(counts: Dict[str, int], n_shots: int) -> float:
            # ⟨Z⟩ on first (rightmost) classical bit
            p0 = sum(v for k, v in counts.items() if k[-1] == "0") / n_shots
            return 2 * p0 - 1

    exp_vals: List[float] = []
    for sf in scale_factors:
        qc_folded = fold_circuit(qc, sf)
        counts = execute_circuit(qc_folded, backend, shots=shots)
        exp_vals.append(observable_fn(counts, shots))

    lambdas = np.array(scale_factors, dtype=float)

    if extrapolation == "richardson" and len(scale_factors) >= 2:
        # Richardson: E_zne = (λ2·E1 − λ1·E2) / (λ2 − λ1)  at λ→0
        l1, l2 = lambdas[0], lambdas[1]
        e1, e2 = exp_vals[0], exp_vals[1]
        E_zne = float((l2 * e1 - l1 * e2) / (l2 - l1))
    else:
        poly  = np.polyfit(lambdas, exp_vals, deg=1)
        E_zne = float(poly[1])   # intercept at λ=0

    return E_zne, exp_vals, scale_factors


def tvd_from_counts(
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
    shots_a: int,
    shots_b: int,
) -> float:
    """Compute Total Variation Distance between two count distributions."""
    all_keys = set(counts_a) | set(counts_b)
    return 0.5 * sum(
        abs(counts_a.get(k, 0) / shots_a - counts_b.get(k, 0) / shots_b)
        for k in all_keys
    )


# ══════════════════════════════════════════════════════════════════════════
# Original simulation-only API (kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════

def apply_zne(
    qc: QuantumCircuit,
    noise_model,
    shots: int = 1024,
    amplification_factors: List[int] = None,
    target_gate: str = "cx",
) -> Tuple[float, List[float], np.ndarray]:
    """
    Zero-Noise Extrapolation using digital gate repetition (simulation only).

    .. deprecated::
        Use ``zne_hardware()`` for both simulation and real hardware.
        This function requires a ``NoiseModel`` and only works with Aer.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    if amplification_factors is None:
        amplification_factors = [1, 2, 3]

    sim = AerSimulator(noise_model=noise_model)
    expectation_values: List[float] = []

    for factor in amplification_factors:
        qc_amp = _amplify_gates(qc, factor, target_gate)
        qc_t = transpile(qc_amp, sim)
        job = sim.run(qc_t, shots=shots)
        counts = job.result().get_counts()

        total = sum(counts.values())
        p0 = counts.get("0", 0) / total
        exp_val = 2 * p0 - 1
        expectation_values.append(exp_val)

    lambdas = np.array(amplification_factors, dtype=float) - 1
    poly = np.polyfit(lambdas, expectation_values, deg=1)
    E_zne = float(poly[1])

    return E_zne, expectation_values, lambdas


# ── Internal helpers ───────────────────────────────────────────────────────

def _strip_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of *qc* with all measurement instructions removed."""
    qc_no_meas = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instr in qc.data:
        if instr.operation.name not in ("measure", "barrier"):
            q_idx = [qc.find_bit(q).index for q in instr.qubits]
            c_idx = [qc.find_bit(c).index for c in instr.clbits]
            qc_no_meas.append(instr.operation, q_idx, c_idx)
    return qc_no_meas


def _amplify_gates(
    qc: QuantumCircuit,
    factor: int,
    target_gate: str = "cx",
) -> QuantumCircuit:
    """Amplify a specific gate by repeating it `factor` times (legacy helper)."""
    qc_amp = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for instr in qc.data:
        q_indices = [qc.find_bit(q).index for q in instr.qubits]
        c_indices = [qc.find_bit(c).index for c in instr.clbits]
        if instr.operation.name == target_gate:
            for _ in range(factor):
                qc_amp.append(instr.operation, q_indices, c_indices)
        else:
            qc_amp.append(instr.operation, q_indices, c_indices)
    return qc_amp



def apply_zne(
    qc: QuantumCircuit,
    noise_model: NoiseModel,
    shots: int = 1024,
    amplification_factors: List[int] = None,
    target_gate: str = "cx",
) -> Tuple[float, List[float], np.ndarray]:
    """
    Zero-Noise Extrapolation using digital gate repetition.

    Amplifies noise by repeating the target gate (default: CNOT) an odd
    number of times, then extrapolates to λ=0 (zero-noise limit).

    Parameters
    ----------
    qc : QuantumCircuit
        Base circuit.
    noise_model : NoiseModel
        Noise model for simulation.
    shots : int
        Shots per noise level.
    amplification_factors : list[int]
        Noise amplification factors (e.g., [1, 2, 3]).
    target_gate : str
        Gate name to amplify (default 'cx').

    Returns
    -------
    (E_zne, expectation_values, lambdas)
        E_zne : extrapolated expectation value at λ=0
        expectation_values : measured ⟨Z⟩ at each factor
        lambdas : shifted factors (factor - 1)
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    if amplification_factors is None:
        amplification_factors = [1, 2, 3]

    sim = AerSimulator(noise_model=noise_model)
    expectation_values: List[float] = []

    for factor in amplification_factors:
        qc_amp = _amplify_gates(qc, factor, target_gate)
        qc_t = transpile(qc_amp, sim)
        job = sim.run(qc_t, shots=shots)
        counts = job.result().get_counts()

        total = sum(counts.values())
        p0 = counts.get("0", 0) / total
        exp_val = 2 * p0 - 1   # maps [0,1] → [-1,+1]
        expectation_values.append(exp_val)

    lambdas = np.array(amplification_factors, dtype=float) - 1
    poly = np.polyfit(lambdas, expectation_values, deg=1)
    E_zne = float(poly[1])  # Intercept at λ=0

    return E_zne, expectation_values, lambdas


# ── Helpers ────────────────────────────────────────────────────────────────

def _amplify_gates(
    qc: QuantumCircuit,
    factor: int,
    target_gate: str = "cx",
) -> QuantumCircuit:
    """
    Amplify a specific gate by repeating it `factor` times.

    Gate * factor preserves logical identity when factor is odd.
    """
    qc_amp = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for instr in qc.data:
        # Map bit objects → integer indices for the new circuit
        q_indices = [qc.find_bit(q).index for q in instr.qubits]
        c_indices = [qc.find_bit(c).index for c in instr.clbits]
        if instr.operation.name == target_gate:
            for _ in range(factor):
                qc_amp.append(instr.operation, q_indices, c_indices)
        else:
            qc_amp.append(instr.operation, q_indices, c_indices)
    return qc_amp
