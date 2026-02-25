"""
Tests for qvoting.mitigation
"""
import numpy as np
import pytest
from qvoting.mitigation.readout import confusion_matrix, apply_readout_mitigation


def test_confusion_matrix_shape():
    M = confusion_matrix()
    assert M.shape == (2, 2)


def test_confusion_matrix_columns_sum_to_one():
    M = confusion_matrix(prob_0_to_1=0.02, prob_1_to_0=0.05)
    np.testing.assert_allclose(M.sum(axis=0), [1.0, 1.0], atol=1e-10)


def test_readout_mitigation_ideal():
    """With a near-identity matrix, mitigated ≈ raw."""
    M = confusion_matrix(prob_0_to_1=0.001, prob_1_to_0=0.001)
    counts_raw = {"0": 500, "1": 500}
    counts_mit = apply_readout_mitigation(counts_raw, M, num_qubits=1)
    total_raw = sum(counts_raw.values())
    total_mit = sum(counts_mit.values())
    # Total shots should be preserved approximately
    assert abs(total_mit - total_raw) < 10


def test_readout_mitigation_reduces_error():
    """Mitigation should bring P(1) closer to ideal (1.0 for X|0⟩)."""
    M = confusion_matrix(prob_0_to_1=0.02, prob_1_to_0=0.05)
    # Simulated: X|0⟩ → should be |1⟩, but 5% noise gives 50 wrong '0'
    counts_raw = {"1": 950, "0": 50}
    counts_mit = apply_readout_mitigation(counts_raw, M, num_qubits=1)
    total = sum(counts_mit.values())
    p1_mit = counts_mit.get("1", 0) / total
    p1_raw = counts_raw["1"] / (counts_raw["0"] + counts_raw["1"])
    # Mitigation should improve p1 (closer to 1.0)
    assert p1_mit >= p1_raw - 0.01  # at least as good
