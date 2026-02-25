"""
Pytest configuration and shared fixtures.
"""
import pytest
from qiskit_aer import AerSimulator


@pytest.fixture
def sim():
    """Clean AerSimulator for use in tests."""
    return AerSimulator()


@pytest.fixture
def noisy_sim():
    """AerSimulator with a basic depolarizing noise model."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    error = depolarizing_error(0.01, 1)
    nm.add_all_qubit_quantum_error(error, ["h", "x"])
    return AerSimulator(noise_model=nm)
