"""
qvoting.nll.circuit
-------------------
NeuralLinkedList: builds the full NLL quantum circuit.

Architecture (v2 — multi-layer)
--------------------------------
For n_nodes nodes and n_layers layers:
  - 2*n_nodes qubits total (2 per node: control + target)
  - Each layer l:
      Node encoding : RY(θ_{l,i}) + CNOT(ctrl_i, tgt_i) per node i
      Inter-node    : CRY(φ_{l,i}) between adjacent control qubits
  - Measurement on all control qubits (qubit 0, 2, 4, ...)

Parameters (flat layout)
-------------------------
  theta : [θ_{0,0}, θ_{0,1}, ..., θ_{L-1, n-1}]   length = n_nodes * n_layers
  phi   : [φ_{0,0}, φ_{0,1}, ..., φ_{L-1, n-2}]   length = (n_nodes-1) * n_layers

Full state:
  |ψ_NLL(Θ, Φ)⟩ = U_L(Θ_L, Φ_L) · ... · U_1(Θ_1, Φ_1) |0⟩^{2n}

Pattern-based initialisation (from_pattern)
-------------------------------------------
  Layer 0: theta_{0,i} = π × bit_i   (encodes target directly)
  Layer 1+: theta ~ N(0, noise_scale)
  All phi   = 0.0                     (no entanglement bias)

This gives P(|target⟩) ≈ 1.0 in ideal simulation and provides a warm
start for SPSA on noisy hardware.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .node import NLLNode


class NeuralLinkedList:
    """
    Neural Linked List quantum circuit builder (multi-layer).

    Parameters
    ----------
    n_nodes : int
        Number of nodes (each occupies 2 qubits).
    theta : list[float] or None
        Flat list of θ values: [layer0_node0, layer0_node1, ..., layerL_nodeN].
        Length must be n_nodes * n_layers. Default: π/4 for all.
    phi : list[float] or None
        Flat list of φ values: [layer0_link0, ..., layerL_linkN-2].
        Length must be (n_nodes-1) * n_layers. Default: 0.0 for all.
    n_layers : int
        Number of encoding + entanglement layers. Default: 1.
    """

    def __init__(
        self,
        n_nodes: int,
        theta: Optional[List[float]] = None,
        phi: Optional[List[float]] = None,
        n_layers: int = 1,
    ):
        self.n_nodes = n_nodes
        self.n_qubits = 2 * n_nodes
        self.n_layers = n_layers

        n_theta = n_nodes * n_layers
        n_phi = (n_nodes - 1) * n_layers

        if theta is None:
            theta = [np.pi / 4] * n_theta
        if phi is None:
            phi = [0.0] * n_phi

        if len(theta) != n_theta:
            raise ValueError(
                f"theta must have {n_theta} elements "
                f"(n_nodes={n_nodes} x n_layers={n_layers}), got {len(theta)}"
            )
        if len(phi) != n_phi:
            raise ValueError(
                f"phi must have {n_phi} elements "
                f"((n_nodes-1)={n_nodes - 1} x n_layers={n_layers}), got {len(phi)}"
            )

        self._theta = list(theta)
        self._phi = list(phi)
        # NLLNode list kept for layer-0 legacy compatibility
        self.nodes = [NLLNode(i, theta[i]) for i in range(n_nodes)]

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_pattern(
        cls,
        target_pattern: str,
        n_layers: int = 2,
        noise_scale: float = 0.1,
        seed: Optional[int] = 42,
    ) -> "NeuralLinkedList":
        """
        Initialise NLL with theta encoding the target bitstring directly.

        Layer 0: theta_{0,i} = π × bit_i  → encodes |target⟩ exactly
        Layer 1+: theta ~ N(0, noise_scale) → small perturbations
        All phi = 0.0                       → no entanglement bias

        In ideal simulation this gives P(|target⟩) ≈ 1.0 before any training,
        providing SPSA with a warm start near the global optimum.

        Parameters
        ----------
        target_pattern : str
            Bitstring, e.g. "101". Length = n_nodes.
        n_layers : int
            Number of encoding layers (default 2 for noise robustness).
        noise_scale : float
            Std deviation for higher-layer theta noise.
        seed : int or None
            RNG seed.
        """
        n_nodes = len(target_pattern)
        rng = np.random.default_rng(seed)
        theta: List[float] = []
        for layer in range(n_layers):
            for bit in target_pattern:
                if layer == 0:
                    theta.append(np.pi * int(bit))
                else:
                    theta.append(float(rng.normal(0.0, noise_scale)))
        phi = [0.0] * ((n_nodes - 1) * n_layers)
        return cls(n_nodes=n_nodes, theta=theta, phi=phi, n_layers=n_layers)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def theta(self) -> List[float]:
        return list(self._theta)

    @property
    def phi(self) -> List[float]:
        return list(self._phi)

    # ── Parameter management ───────────────────────────────────────────────

    def set_params(self, theta: List[float], phi: List[float]) -> None:
        """Update all trainable parameters (theta and phi flat lists)."""
        n_theta = self.n_nodes * self.n_layers
        n_phi = (self.n_nodes - 1) * self.n_layers
        if len(theta) != n_theta:
            raise ValueError(f"Expected {n_theta} theta values, got {len(theta)}")
        if len(phi) != n_phi:
            raise ValueError(f"Expected {n_phi} phi values, got {len(phi)}")
        self._theta = list(theta)
        self._phi = list(phi)
        for i, node in enumerate(self.nodes):
            node.set_theta(theta[i])

    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return self.n_nodes * self.n_layers + (self.n_nodes - 1) * self.n_layers

    def get_params(self) -> np.ndarray:
        """Return all parameters as a flat array [theta..., phi...]."""
        return np.array(self._theta + self._phi)

    def set_params_flat(self, params: np.ndarray) -> None:
        """Set all parameters from a flat array (same order as get_params)."""
        n_theta = self.n_nodes * self.n_layers
        self.set_params(list(params[:n_theta]), list(params[n_theta:]))

    # ── Circuit builder ────────────────────────────────────────────────────

    def build_circuit(self, measure: bool = True) -> QuantumCircuit:
        """
        Build the multi-layer NLL quantum circuit.

        Each layer applies:
          1. RY(θ_{l,i}) + CNOT(ctrl_i, tgt_i) for each node i
          2. CRY(φ_{l,i}) between adjacent control qubits

        Parameters
        ----------
        measure : bool
            If True, append measurements on all control qubits.

        Returns
        -------
        QuantumCircuit
        """
        qreg = QuantumRegister(self.n_qubits, "nll")
        creg = ClassicalRegister(self.n_nodes, "c") if measure else None
        qc = QuantumCircuit(qreg, creg) if measure else QuantumCircuit(qreg)

        for layer in range(self.n_layers):
            # Node encoding: RY(θ_{l,i}) + CNOT per node
            for i in range(self.n_nodes):
                theta_li = self._theta[layer * self.n_nodes + i]
                ctrl = 2 * i
                tgt = 2 * i + 1
                qc.ry(theta_li, ctrl)
                qc.cx(ctrl, tgt)

            # Inter-node links: CRY(φ_{l,i}) between adjacent control qubits
            for i in range(self.n_nodes - 1):
                phi_li = self._phi[layer * (self.n_nodes - 1) + i]
                ctrl_a = 2 * i
                ctrl_b = 2 * (i + 1)
                qc.cry(phi_li, ctrl_a, ctrl_b)

            if layer < self.n_layers - 1:
                qc.barrier()

        qc.barrier()

        if measure:
            for i in range(self.n_nodes):
                qc.measure(2 * i, i)

        return qc

    def __repr__(self) -> str:
        return (
            f"NeuralLinkedList(n_nodes={self.n_nodes}, "
            f"n_layers={self.n_layers}, "
            f"n_qubits={self.n_qubits}, n_params={self.n_params()})"
        )


def pattern_recovery_rate(
    counts: Dict[str, int],
    target_pattern: str,
    shots: int,
) -> float:
    """
    Compute P(target_pattern) from measurement counts.

    Parameters
    ----------
    counts : dict[str, int]
        Measurement counts from circuit execution.
    target_pattern : str
        Bitstring to measure probability of (e.g., "101" for NLL-3).
    shots : int
        Total number of shots.

    Returns
    -------
    float in [0, 1]

    Examples
    --------
    >>> counts = {"101": 820, "000": 104, "111": 100}
    >>> pattern_recovery_rate(counts, "101", shots=1024)
    0.80078125
    """
    return counts.get(target_pattern, 0) / shots
