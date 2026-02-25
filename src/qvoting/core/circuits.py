"""
qvoting.core.circuits
---------------------
Quantum circuit primitives: subcircuits and multi-circuit balancers.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def parity_subcircuit(
    name: str,
    num_qubits: int,
    active_qubits: List[int],
    target_idx: Optional[int] = None,
) -> Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    """
    Create a parity sub-circuit without global state dependencies.

    Parameters
    ----------
    name : str
        Register name.
    num_qubits : int
        Number of qubits in the register.
    active_qubits : list[int]
        Indices of qubits to flip (set to |1⟩) before the parity calculation.
    target_idx : int, optional
        Qubit index to accumulate the parity (default: last qubit).

    Returns
    -------
    (QuantumCircuit, QuantumRegister, ClassicalRegister)
    """
    if target_idx is None:
        target_idx = num_qubits - 1

    qreg = QuantumRegister(num_qubits, name)
    creg = ClassicalRegister(1, f"c_{name}")
    qc = QuantumCircuit(qreg, creg)

    # Prepare active qubits
    for idx in active_qubits:
        qc.x(qreg[idx])

    # XOR chain → parity
    for i in range(num_qubits - 1):
        if i != target_idx:
            qc.cx(qreg[i], qreg[target_idx])

    qc.measure(qreg[target_idx], creg[0])
    return qc, qreg, creg


def multi_circuit_balancer(
    subcircuits: List[Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]],
    logic_mode: str = "xor",
) -> QuantumCircuit:
    """
    Combine multiple subcircuits into a single multi-register circuit.

    Parameters
    ----------
    subcircuits : list
        List of (QuantumCircuit, QuantumRegister, ClassicalRegister) tuples.
    logic_mode : str
        Combination logic: 'xor', 'majority', 'and', 'or'.

    Returns
    -------
    QuantumCircuit
    """
    if not subcircuits:
        raise ValueError("subcircuits cannot be empty")

    all_qregs = [sc[1] for sc in subcircuits]
    all_cregs = [sc[2] for sc in subcircuits]

    qc_combined = QuantumCircuit(*all_qregs, *all_cregs)

    for sub_qc, sub_qreg, sub_creg in subcircuits:
        qc_combined.compose(
            sub_qc,
            qubits=list(sub_qreg),
            clbits=list(sub_creg),
            inplace=True,
        )

    return qc_combined
