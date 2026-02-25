"""
Tests for qvoting.core.circuits
"""
import pytest
from qiskit import QuantumCircuit
from qvoting.core.circuits import parity_subcircuit, multi_circuit_balancer


def test_parity_subcircuit_returns_tuple():
    qc, qreg, creg = parity_subcircuit("test", 3, [0, 1])
    assert isinstance(qc, QuantumCircuit)


def test_parity_subcircuit_num_qubits():
    qc, qreg, creg = parity_subcircuit("test", 4, [0, 2])
    assert qc.num_qubits == 4
    assert qc.num_clbits == 1


def test_parity_subcircuit_default_target():
    qc, qreg, creg = parity_subcircuit("test", 3, [0])
    # Default target = last qubit (index 2)
    assert len(qreg) == 3


def test_multi_circuit_balancer_empty_raises():
    with pytest.raises(ValueError):
        multi_circuit_balancer([])


def test_multi_circuit_balancer_combines():
    sc1 = parity_subcircuit("sc1", 2, [0])
    sc2 = parity_subcircuit("sc2", 2, [1])
    qc = multi_circuit_balancer([sc1, sc2])
    assert qc.num_qubits == 4
    assert qc.num_clbits == 2
