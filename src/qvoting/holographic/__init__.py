"""
qvoting.holographic
-------------------
Holographic quantum error correction via hierarchical majority voting.

The 9→3→1 voter structure implements a [[9,1,3]] repetition code
with holographic properties: the logical (bulk) qubit can be
reconstructed from any boundary region that contains a majority
of the 3 voter groups.

This module exposes:
  - encode_logical_qubit              : encodes a|0>+b|1> into 9 boundary qubits
  - erase_qubits                      : simulates qubit erasure (quantum channel)
  - run_erasure_sweep                 : full experiment across all erasure patterns
  - min_cut                           : RT formula - min cut in voter network graph
  - rt_prediction                     : RT entanglement entropy prediction for region A
  - build_holographic_tomography_circuit : multi-basis circuits for state tomography
  - bloch_from_counts                 : reconstruct Bloch vector from 3-basis counts
  - state_fidelity                    : quantum state fidelity from Bloch vectors

References
----------
Almheiri, Dong, Harlow (2015) -- Bulk locality and QEC in AdS/CFT
Ryu, Takayanagi (2006)        -- Holographic derivation of entanglement entropy
"""

from qvoting.holographic.encoder import (
    encode_logical_qubit,
    build_voter_decoder,
    build_holographic_circuit,
    build_holographic_circuit_zne,
    build_holographic_tomography_circuit,
)
from qvoting.holographic.erasure import (
    erase_qubits,
    run_erasure_sweep,
    summarize_sweep,
    run_rt_curve_experiment,
)
from qvoting.holographic.rt_formula import (
    min_cut,
    rt_prediction,
    all_rt_predictions,
    erasure_threshold_table,
)
from qvoting.holographic.tomography import (
    ideal_bloch_vector,
    bloch_from_counts,
    state_fidelity,
    holographic_fidelity_proxy,
)

__all__ = [
    "encode_logical_qubit",
    "build_voter_decoder",
    "build_holographic_circuit",
    "build_holographic_circuit_zne",
    "build_holographic_tomography_circuit",
    "erase_qubits",
    "run_erasure_sweep",
    "summarize_sweep",
    "run_rt_curve_experiment",
    "min_cut",
    "rt_prediction",
    "all_rt_predictions",
    "erasure_threshold_table",
    "ideal_bloch_vector",
    "bloch_from_counts",
    "state_fidelity",
    "holographic_fidelity_proxy",
]
