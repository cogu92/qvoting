"""
qvoting.holographic.rt_formula
--------------------------------
Discrete Ryu-Takayanagi formula for the 9→3→1 voter network.

THEORY
------
In AdS/CFT holography, the entanglement entropy of a boundary region A is:

    S(A) = Area(γ_A) / 4G_N

where γ_A is the minimal surface separating A from its complement in the bulk.

For discrete tensor network codes (Pastawski, Yoshida, Harlow, Preskill 2015),
this becomes a MIN-CUT problem in the network graph:

    S(A) = min_cut(A, bulk_output)

VOTER NETWORK GRAPH
-------------------
Nodes:
    boundary : b[0..8]  (9 physical qubits — the AdS boundary)
    level-1  : v1[0..2] (3 intermediate nodes — bulk entanglement wedge)
    bulk     : v2[0]    (1 logical qubit — bulk point)

Edges (capacity 1 each):
    b[0] → v1[0],  b[1] → v1[0],  b[2] → v1[0]   (group 0)
    b[3] → v1[1],  b[4] → v1[1],  b[5] → v1[1]   (group 1)
    b[6] → v1[2],  b[7] → v1[2],  b[8] → v1[2]   (group 2)
    v1[0] → v2[0], v1[1] → v2[0], v1[2] → v2[0]

RT FORMULA RESULT
-----------------
For any boundary region A ⊆ {0,...,8}:

    S_RT(A) = |{groups touched by A}|
            = |{i ∈ {0,1,2} : A ∩ group_i ≠ ∅}|

This equals the number of voter groups that A "activates" — the number of
v1→v2 edges that must be cut to separate A from the bulk output.

HOLOGRAPHIC RECONSTRUCTION CONDITION
-------------------------------------
The complementary region Ā can reconstruct the bulk iff:

    min_cut(Ā, bulk) ≥ 2

i.e., Ā touches at least 2 groups with at least 2 qubits each
(so each touched group votes correctly via majority).

Equivalently: reconstruction from A fails only if A contains ≥ 2 complete
majority groups (groups where A holds ≥ 2 of 3 qubits).
"""
from __future__ import annotations

from itertools import combinations


# ── Voter network structure ──────────────────────────────────────────────────
GROUPS: dict[int, list[int]] = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
}

BOUNDARY_QUBITS = list(range(9))


def _groups_touched(region: set[int]) -> set[int]:
    """Return group indices that contain at least one qubit from region."""
    return {g for g, members in GROUPS.items() if region & set(members)}


def _majority_groups(region: set[int]) -> set[int]:
    """Return groups where region holds a majority (≥ 2 of 3) of the qubits."""
    return {
        g for g, members in GROUPS.items()
        if len(region & set(members)) >= 2
    }


def min_cut(boundary_region: set[int]) -> int:
    """
    Compute the min-cut between boundary_region and the bulk output (v2[0]).

    This is the discrete Ryu-Takayanagi area: the minimum number of edges
    that must be cut to disconnect the region from the bulk.

    In the voter network, the min-cut equals the number of voter groups
    that the region touches (has at least 1 qubit from).

    Parameters
    ----------
    boundary_region : set[int]
        Subset of boundary qubit indices {0,...,8}.

    Returns
    -------
    int
        Min-cut value = RT entropy prediction S_RT(A).

    Examples
    --------
    >>> min_cut({0, 1, 2})           # full group 0
    1
    >>> min_cut({0, 3})              # one qubit from groups 0 and 1
    2
    >>> min_cut({0, 3, 6})           # one qubit from each group
    3
    >>> min_cut(set())               # empty region
    0
    """
    return len(_groups_touched(boundary_region))


def rt_prediction(boundary_region: set[int]) -> dict:
    """
    Full Ryu-Takayanagi prediction for a boundary region A.

    Returns the RT entropy, the reconstruction condition, and the
    entanglement wedge (which part of the bulk is accessible from A).

    Parameters
    ----------
    boundary_region : set[int]
        Subset of boundary qubit indices {0,...,8}.

    Returns
    -------
    dict with keys:
        region         : the input boundary region A
        complement     : the complementary region Ā
        S_RT_A         : RT entropy of A (min-cut from A to bulk)
        S_RT_Abar      : RT entropy of Ā (min-cut from Ā to bulk)
        A_can_reconstruct   : True if A can reconstruct the bulk
        Abar_can_reconstruct: True if Ā can reconstruct the bulk
        maj_groups_A   : groups where A holds a majority
        maj_groups_Abar: groups where Ā holds a majority
        note           : human-readable interpretation
    """
    A = set(boundary_region)
    Abar = set(BOUNDARY_QUBITS) - A

    s_a = min_cut(A)
    s_abar = min_cut(Abar)

    maj_a = _majority_groups(A)
    maj_abar = _majority_groups(Abar)

    # Reconstruction requires a majority of the 3 voter groups to vote
    # correctly. A group votes correctly iff region holds >= 2 of its 3 qubits.
    a_can = len(maj_a) >= 2
    abar_can = len(maj_abar) >= 2

    if a_can and abar_can:
        note = "Complementary recovery: both A and Abar reconstruct the bulk."
    elif a_can:
        note = "Only A can reconstruct the bulk (Abar is too sparse)."
    elif abar_can:
        note = "Only Abar can reconstruct the bulk (A is too sparse)."
    else:
        note = "Neither A nor Abar can reconstruct the bulk — info is lost."

    return {
        "region":               sorted(A),
        "complement":           sorted(Abar),
        "S_RT_A":               s_a,
        "S_RT_Abar":            s_abar,
        "A_can_reconstruct":    a_can,
        "Abar_can_reconstruct": abar_can,
        "maj_groups_A":         sorted(maj_a),
        "maj_groups_Abar":      sorted(maj_abar),
        "note":                 note,
    }


def all_rt_predictions() -> list[dict]:
    """
    Compute RT predictions for ALL 2^9 = 512 possible boundary regions.

    Returns a list of dicts (one per region), sorted by region size.
    Useful for building the full RT entropy curve and verifying the
    holographic threshold experimentally.
    """
    results = []
    for size in range(10):
        for region in combinations(BOUNDARY_QUBITS, size):
            results.append(rt_prediction(set(region)))
    return results


def erasure_threshold_table() -> list[dict]:
    """
    For each possible erasure pattern (set of erased qubits), predict
    whether holographic reconstruction succeeds.

    An erasure of qubits E corresponds to removing those qubits from the
    boundary — the decoder must reconstruct from the surviving region A = Ā^E
    where Ā^E = {0,...,8} \\ E.

    Returns
    -------
    list[dict] with keys:
        erased_qubits         : set of erased qubit indices
        n_erased              : number of erased qubits
        surviving_region      : A = boundary - erased
        maj_groups_surviving  : groups where surviving region holds majority
        reconstruction_ok     : True if reconstruction is possible
    """
    results = []
    for size in range(10):
        for erased in combinations(BOUNDARY_QUBITS, size):
            erased_set = set(erased)
            surviving = set(BOUNDARY_QUBITS) - erased_set
            maj_surv = _majority_groups(surviving)
            ok = len(maj_surv) >= 2

            results.append({
                "erased_qubits":        sorted(erased_set),
                "n_erased":             size,
                "surviving_region":     sorted(surviving),
                "maj_groups_surviving": sorted(maj_surv),
                "reconstruction_ok":    ok,
            })
    return results


def rt_entropy_curve() -> dict[int, dict]:
    """
    Compute the RT entropy averaged over all boundary regions of each size.

    Returns
    -------
    dict mapping region_size → {mean_S_RT, min_S_RT, max_S_RT, n_regions}

    The curve shows how S_RT(A) scales with |A|, analogous to the
    entanglement entropy growth with subsystem size in holographic systems.
    """
    from collections import defaultdict

    buckets: dict[int, list[int]] = defaultdict(list)
    for size in range(10):
        for region in combinations(BOUNDARY_QUBITS, size):
            s = min_cut(set(region))
            buckets[size].append(s)

    curve = {}
    for size, vals in sorted(buckets.items()):
        curve[size] = {
            "mean_S_RT":  sum(vals) / len(vals),
            "min_S_RT":   min(vals),
            "max_S_RT":   max(vals),
            "n_regions":  len(vals),
        }
    return curve
