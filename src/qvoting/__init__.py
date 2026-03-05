"""
QVoting — Quantum Voting Framework
====================================

Quantum voting circuits with integrated error mitigation for NISQ hardware.

Quick start
-----------
>>> from qvoting.voters import majority_voter
>>> from qvoting.core import execute_circuit
>>> qc = majority_voter(num_inputs=3)
>>> counts = execute_circuit(qc, backend="aer_simulator", shots=1024)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qvoting")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__author__ = "Nicolas Yesid Corredor Guasca"
__license__ = "MIT"

# Importaciones principales para acceso directo
from qvoting.voters.majority import majority_voter
from qvoting.core.execution import execute_circuit
from qvoting.core.logging import JobLogger
from qvoting.mitigation.readout import apply_readout_mitigation
from qvoting.mitigation.zne import (
    apply_zne, zne_hardware, fold_circuit, tvd_from_counts,
)
from qvoting.nisq_selector import NISQSelector
from qvoting.nll import NeuralLinkedList, SPSATrainer, pattern_recovery_rate
from qvoting.holographic import (
    encode_logical_qubit,
    build_voter_decoder,
    erase_qubits,
    run_erasure_sweep,
    min_cut,
    rt_prediction,
    all_rt_predictions,
)

__all__ = [
    "majority_voter",
    "execute_circuit",
    "JobLogger",
    "apply_readout_mitigation",
    "apply_zne",
    "zne_hardware",
    "fold_circuit",
    "tvd_from_counts",
    "NISQSelector",
    # nll
    "NeuralLinkedList",
    "SPSATrainer",
    "pattern_recovery_rate",
    # holographic
    "encode_logical_qubit",
    "build_voter_decoder",
    "erase_qubits",
    "run_erasure_sweep",
    "min_cut",
    "rt_prediction",
    "all_rt_predictions",
]
