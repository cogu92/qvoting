"""
qvoting.nll.node
----------------
NLLNode: a single 2-qubit trainable node of the Neural Linked List.

State encoding
--------------
Each node encodes a bit value b ∈ {0,1} using a trainable angle θ:

    |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩

Implemented with RY(θ) on the control qubit followed by CNOT(control, target).
For θ = 0  → |00⟩  (encodes bit 0)
For θ = π  → |11⟩  (encodes bit 1)
For θ = π/2 → (|00⟩+|11⟩)/√2 (superposition)
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit


class NLLNode:
    """
    Single 2-qubit node of a Neural Linked List.

    Parameters
    ----------
    node_id : int
        Index of this node within the NLL (0-indexed).
    theta : float
        Initial rotation angle in radians. θ=0 → |00⟩, θ=π → |11⟩.
    """

    def __init__(self, node_id: int, theta: float = np.pi / 4):
        self.node_id = node_id
        self.theta = float(theta)

    @property
    def qubit_indices(self) -> tuple[int, int]:
        """Control and target qubit indices in the full NLL circuit."""
        ctrl = 2 * self.node_id
        tgt  = 2 * self.node_id + 1
        return ctrl, tgt

    def apply_to_circuit(self, qc: QuantumCircuit) -> None:
        """
        Apply this node's encoding to *qc* in-place.

        Adds RY(θ) on the control qubit and CNOT(control, target).
        The resulting state is cos(θ/2)|00⟩ + sin(θ/2)|11⟩.
        """
        ctrl, tgt = self.qubit_indices
        qc.ry(self.theta, ctrl)
        qc.cx(ctrl, tgt)

    def set_theta(self, theta: float) -> None:
        """Update the trainable angle."""
        self.theta = float(theta)

    def __repr__(self) -> str:
        return f"NLLNode(id={self.node_id}, θ={self.theta:.4f})"
