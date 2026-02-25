"""
qvoting.mitigation.readout
--------------------------
Readout error mitigation via confusion matrix inversion.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def confusion_matrix(prob_0_to_1: float = 0.02, prob_1_to_0: float = 0.05) -> np.ndarray:
    """
    Build a single-qubit readout confusion matrix.

    Parameters
    ----------
    prob_0_to_1 : float
        P(measure 1 | prepared 0).
    prob_1_to_0 : float
        P(measure 0 | prepared 1).

    Returns
    -------
    np.ndarray, shape (2, 2)
        Confusion matrix M where M[j, i] = P(measure j | prepared i).
    """
    return np.array([
        [1 - prob_1_to_0, prob_0_to_1],
        [prob_1_to_0,     1 - prob_0_to_1],
    ])


def apply_readout_mitigation(
    counts: Dict[str, int],
    calib_matrix: np.ndarray,
    num_qubits: int = 1,
) -> Dict[str, int]:
    """
    Apply readout error mitigation by inverting the calibration matrix.

    Parameters
    ----------
    counts : dict
        Raw measurement counts.
    calib_matrix : np.ndarray
        Calibration/confusion matrix for *one* qubit (2×2).
        For multi-qubit, a tensor product is computed automatically.
    num_qubits : int
        Number of qubits measured.

    Returns
    -------
    dict
        Mitigated counts (int-valued, non-negative).

    Notes
    -----
    Uses the single-qubit approximation M⊗n for the full confusion matrix.
    For full multi-qubit calibration use qiskit_experiments.
    """
    total = sum(counts.values())
    num_states = 2 ** num_qubits

    # Build full confusion matrix via tensor product
    M_full = calib_matrix.copy()
    for _ in range(num_qubits - 1):
        M_full = np.kron(M_full, calib_matrix)

    if M_full.shape[0] != num_states:
        raise ValueError(
            f"Confusion matrix shape {M_full.shape} does not match {num_states} states"
        )

    # Build probability vector from counts
    states = [format(i, f"0{num_qubits}b") for i in range(num_states)]
    prob_raw = np.array([counts.get(s, 0) / total for s in states])

    # Invert: M @ p_true ≈ p_raw  →  p_true ≈ M⁻¹ @ p_raw
    M_inv = np.linalg.inv(M_full)
    prob_true = M_inv @ prob_raw

    # Force non-negative and renormalize
    prob_true = np.maximum(prob_true, 0)
    prob_sum = prob_true.sum()
    if prob_sum > 0:
        prob_true /= prob_sum

    mitigated = {s: int(round(p * total)) for s, p in zip(states, prob_true)}
    return {k: v for k, v in mitigated.items() if v > 0}
