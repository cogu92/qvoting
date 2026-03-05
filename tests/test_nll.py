"""
Tests for qvoting.nll — Neural Linked List module.
"""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qvoting.nll import NeuralLinkedList, SPSATrainer, pattern_recovery_rate
from qvoting.nll.node import NLLNode


# ── NLLNode tests ─────────────────────────────────────────────────────────

class TestNLLNode:
    def test_qubit_indices(self):
        node = NLLNode(node_id=0, theta=0.0)
        assert node.qubit_indices == (0, 1)

        node2 = NLLNode(node_id=2, theta=0.0)
        assert node2.qubit_indices == (4, 5)

    def test_apply_to_circuit_zero_theta(self):
        """θ=0 → |00⟩: RY(0) is identity, CNOT does nothing."""
        node = NLLNode(node_id=0, theta=0.0)
        qc = QuantumCircuit(2, 2)
        node.apply_to_circuit(qc)
        qc.measure([0, 1], [0, 1])

        sim = AerSimulator()
        counts = sim.run(qc, shots=256).result().get_counts()
        assert counts.get("00", 0) > 200  # almost all |00⟩

    def test_apply_to_circuit_pi_theta(self):
        """θ=π → |11⟩: RY(π) = X, CNOT flips target."""
        node = NLLNode(node_id=0, theta=np.pi)
        qc = QuantumCircuit(2, 2)
        node.apply_to_circuit(qc)
        qc.measure([0, 1], [0, 1])

        sim = AerSimulator()
        counts = sim.run(qc, shots=256).result().get_counts()
        assert counts.get("11", 0) > 200  # almost all |11⟩

    def test_set_theta(self):
        node = NLLNode(node_id=1, theta=0.5)
        node.set_theta(1.0)
        assert node.theta == 1.0

    def test_repr(self):
        node = NLLNode(node_id=0, theta=np.pi / 4)
        assert "NLLNode" in repr(node)
        assert "id=0" in repr(node)


# ── NeuralLinkedList tests ─────────────────────────────────────────────────

class TestNeuralLinkedList:
    def test_default_construction(self):
        nll = NeuralLinkedList(n_nodes=3)
        assert nll.n_nodes == 3
        assert nll.n_qubits == 6
        assert len(nll.phi) == 2

    def test_n_params(self):
        nll3 = NeuralLinkedList(n_nodes=3)
        assert nll3.n_params() == 5  # 3 theta + 2 phi

        nll10 = NeuralLinkedList(n_nodes=10)
        assert nll10.n_params() == 19  # 10 theta + 9 phi

    def test_wrong_theta_length_raises(self):
        with pytest.raises(ValueError, match="theta must have"):
            NeuralLinkedList(n_nodes=3, theta=[0.1, 0.2])  # only 2, need 3

    def test_wrong_phi_length_raises(self):
        with pytest.raises(ValueError, match="phi must have"):
            NeuralLinkedList(n_nodes=3, phi=[0.1, 0.2, 0.3])  # 3, need 2

    def test_build_circuit_qubits(self):
        nll = NeuralLinkedList(n_nodes=3)
        qc = nll.build_circuit(measure=True)
        assert qc.num_qubits == 6
        assert qc.num_clbits == 3

    def test_build_circuit_no_measure(self):
        nll = NeuralLinkedList(n_nodes=3)
        qc = nll.build_circuit(measure=False)
        assert qc.num_clbits == 0

    def test_build_circuit_depth_reasonable(self):
        """NLL-3 depth should be < 30 for NISQ hardware feasibility."""
        nll = NeuralLinkedList(n_nodes=3)
        qc = nll.build_circuit(measure=True)
        assert qc.depth() < 30

    def test_get_set_params_flat(self):
        nll = NeuralLinkedList(n_nodes=3)
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        nll.set_params_flat(params)
        retrieved = nll.get_params()
        np.testing.assert_allclose(retrieved, params)

    def test_set_params(self):
        nll = NeuralLinkedList(n_nodes=3)
        nll.set_params(theta=[0.1, 0.2, 0.3], phi=[0.4, 0.5])
        assert nll.theta == [0.1, 0.2, 0.3]
        assert nll.phi == [0.4, 0.5]

    def test_simulation_output_range(self):
        """Pattern recovery rate must be in [0, 1]."""
        nll = NeuralLinkedList(n_nodes=3, theta=[np.pi, 0.0, np.pi], phi=[0.0, 0.0])
        qc = nll.build_circuit(measure=True)

        sim = AerSimulator()
        counts = sim.run(qc, shots=512).result().get_counts()
        p = pattern_recovery_rate(counts, "101", shots=512)
        assert 0.0 <= p <= 1.0

    def test_all_ones_pattern_high_prob(self):
        """θ=π for all nodes → state |11⟩^n → measuring control qubits gives |1>^n."""
        n = 3
        nll = NeuralLinkedList(n_nodes=n, theta=[np.pi] * n, phi=[0.0] * (n - 1))
        qc = nll.build_circuit(measure=True)

        sim = AerSimulator()
        counts = sim.run(qc, shots=512).result().get_counts()
        p_all1 = pattern_recovery_rate(counts, "1" * n, shots=512)
        assert p_all1 > 0.8  # should be near 1.0 with ideal sim

    def test_repr(self):
        nll = NeuralLinkedList(n_nodes=5)
        r = repr(nll)
        assert "NeuralLinkedList" in r
        assert "n_nodes=5" in r


# ── pattern_recovery_rate tests ────────────────────────────────────────────

class TestPatternRecoveryRate:
    def test_exact_match(self):
        counts = {"101": 1024}
        assert pattern_recovery_rate(counts, "101", 1024) == 1.0

    def test_zero_match(self):
        counts = {"000": 512, "111": 512}
        assert pattern_recovery_rate(counts, "101", 1024) == 0.0

    def test_partial(self):
        counts = {"101": 512, "000": 512}
        assert pattern_recovery_rate(counts, "101", 1024) == pytest.approx(0.5)

    def test_random_baseline(self):
        """For n=3 random circuit, P(101) ≈ 1/8 = 12.5%."""
        # Not a strict statistical test, just verify output is in range
        counts = {"101": 16, "000": 112, "111": 100, "010": 28}
        p = pattern_recovery_rate(counts, "101", 256)
        assert 0.0 <= p <= 1.0
