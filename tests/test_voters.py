"""
Tests for qvoting.voters.majority
"""
import pytest
from qiskit import QuantumCircuit
from qvoting.voters.majority import majority_voter


def test_voter_3_returns_circuit():
    qc = majority_voter(num_inputs=3)
    assert isinstance(qc, QuantumCircuit)


def test_voter_3_qubit_count():
    qc = majority_voter(num_inputs=3)
    # 3 input + 1 vote
    assert qc.num_qubits == 4
    assert qc.num_clbits == 1


def test_voter_5_returns_circuit():
    qc = majority_voter(num_inputs=5)
    assert isinstance(qc, QuantumCircuit)


def test_voter_5_qubit_count():
    qc = majority_voter(num_inputs=5)
    # 5 input + 3 sum
    assert qc.num_qubits == 8
    assert qc.num_clbits == 3


def test_voter_invalid_raises():
    with pytest.raises(ValueError):
        majority_voter(num_inputs=2)


def test_voter_3_simulation():
    """With all inputs=|1⟩, majority should return |1⟩."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    qc = majority_voter(num_inputs=3)
    sim = AerSimulator()
    qc_t = transpile(qc, sim)
    result = sim.run(qc_t, shots=1024).result()
    counts = result.get_counts()
    # Majority of (1,1,1) = 1
    assert counts.get("1", 0) > 900, f"Expected mostly '1', got {counts}"
