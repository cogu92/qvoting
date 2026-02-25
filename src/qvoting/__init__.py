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

__author__ = "Nicolas"
__license__ = "MIT"

# Importaciones principales para acceso directo
from qvoting.voters.majority import majority_voter
from qvoting.core.execution import execute_circuit
from qvoting.core.logging import JobLogger
from qvoting.mitigation.readout import apply_readout_mitigation
from qvoting.mitigation.zne import apply_zne

__all__ = [
    "majority_voter",
    "execute_circuit",
    "JobLogger",
    "apply_readout_mitigation",
    "apply_zne",
]
