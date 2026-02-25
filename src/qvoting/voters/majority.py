"""
qvoting.voters.majority
-----------------------
Toffoli-based majority voters for 3 and 5 inputs.
"""
from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def majority_voter(num_inputs: int = 3) -> QuantumCircuit:
    """
    Build a quantum majority voter circuit.

    The circuit computes majority(q_0, ..., q_{n-1}) into a vote qubit.

    Parameters
    ----------
    num_inputs : int
        Number of input qubits. Supported: 3 or 5.

    Returns
    -------
    QuantumCircuit
        Circuit with input qubits initialized to |1⟩ for demonstration.

    Raises
    ------
    ValueError
        If num_inputs is not 3 or 5.

    Examples
    --------
    >>> from qvoting.voters import majority_voter
    >>> qc = majority_voter(num_inputs=3)
    >>> qc.num_qubits
    4
    """
    if num_inputs == 3:
        return _voter_3()
    elif num_inputs == 5:
        return _voter_5()
    else:
        raise ValueError(f"num_inputs must be 3 or 5, got {num_inputs}")


def _voter_3() -> QuantumCircuit:
    """3-input Toffoli majority voter."""
    inputs = QuantumRegister(3, "input")
    vote = QuantumRegister(1, "vote")
    creg = ClassicalRegister(1, "result")
    qc = QuantumCircuit(inputs, vote, creg)

    # Initialize all inputs to |1⟩ (for testing)
    for i in range(3):
        qc.x(inputs[i])

    qc.barrier()

    # Majority via 3 Toffoli gates (correct majority function):
    # vote = (a∧b) ⊕ (b∧c) ⊕ (a∧c)  →  1 iff ≥2 inputs are 1
    qc.ccx(inputs[0], inputs[1], vote[0])
    qc.ccx(inputs[1], inputs[2], vote[0])
    qc.ccx(inputs[0], inputs[2], vote[0])

    qc.barrier()
    qc.measure(vote[0], creg[0])
    return qc


def _voter_5() -> QuantumCircuit:
    """5-input majority voter using a sum register."""
    inputs = QuantumRegister(5, "input")
    sum_reg = QuantumRegister(3, "sum")   # enough bits to count 0-5
    creg = ClassicalRegister(3, "result")
    qc = QuantumCircuit(inputs, sum_reg, creg)

    # Initialize all inputs to |1⟩
    for i in range(5):
        qc.x(inputs[i])

    qc.barrier()

    # Count inputs into sum_reg (popcount)
    for i in range(5):
        qc.cx(inputs[i], sum_reg[0])
    for i in range(5):
        qc.ccx(inputs[i], sum_reg[0], sum_reg[1])

    qc.barrier()
    qc.measure(sum_reg, creg)
    return qc
