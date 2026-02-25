"""
qvoting.voters.hierarchical
-----------------------------
Hierarchical majority voting: 9-input → 3-input → 1-output.
"""
from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qvoting.voters.majority import _voter_3


def hierarchical_voter_9to1() -> QuantumCircuit:
    """
    Build a 9-input hierarchical majority voter.

    Architecture:
        3 groups of 3 inputs → 3 local majority votes → 1 global vote.

        [in0, in1, in2] → vote0
        [in3, in4, in5] → vote1  →  global_vote
        [in6, in7, in8] → vote2

    Returns
    -------
    QuantumCircuit
        9 input qubits + 3 level-1 vote qubits + 1 level-2 vote qubit = 13 qubits total.
    """
    n_inputs = 9
    inputs = QuantumRegister(n_inputs, "inp")
    votes_l1 = QuantumRegister(3, "v1")   # Level-1 votes
    vote_l2 = QuantumRegister(1, "v2")    # Final vote
    creg = ClassicalRegister(1, "result")

    qc = QuantumCircuit(inputs, votes_l1, vote_l2, creg)

    # Initialize all inputs to |1⟩ (demo: majority of all ones = 1)
    for i in range(n_inputs):
        qc.x(inputs[i])

    qc.barrier(label="Level 1")

    # ── Level 1: three local 3-input majority voters ───────────────────
    # Correct majority: maj(a,b,c) = (a∧b) ⊕ (b∧c) ⊕ (a∧c)  via 3×CCX
    groups = [(0, 1, 2, 0), (3, 4, 5, 1), (6, 7, 8, 2)]
    for a_idx, b_idx, c_idx, v_idx in groups:
        qc.ccx(inputs[a_idx], inputs[b_idx], votes_l1[v_idx])
        qc.ccx(inputs[b_idx], inputs[c_idx], votes_l1[v_idx])
        qc.ccx(inputs[a_idx], inputs[c_idx], votes_l1[v_idx])

    qc.barrier(label="Level 2")

    # ── Level 2: global majority vote on the three l1 results ─────────
    # Same 3×CCX pattern for correctness
    qc.ccx(votes_l1[0], votes_l1[1], vote_l2[0])
    qc.ccx(votes_l1[1], votes_l1[2], vote_l2[0])
    qc.ccx(votes_l1[0], votes_l1[2], vote_l2[0])

    qc.barrier()
    qc.measure(vote_l2[0], creg[0])
    return qc
