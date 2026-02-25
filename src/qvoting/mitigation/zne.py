"""
qvoting.mitigation.zne
-----------------------
Zero-Noise Extrapolation (ZNE) via digital gate amplification.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel


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
        if instr.operation.name == target_gate:
            for _ in range(factor):
                qc_amp.append(instr.operation, instr.qubits, instr.clbits)
        else:
            qc_amp.append(instr.operation, instr.qubits, instr.clbits)
    return qc_amp
