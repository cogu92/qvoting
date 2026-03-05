"""
qvoting.holographic.erasure
----------------------------
Quantum erasure experiment for the holographic voter.

EXPERIMENT DESIGN
-----------------
We test whether the 9→3→1 majority voter can reconstruct the logical
(bulk) qubit after k boundary qubits are erased.

Protocol for each erasure pattern E ⊆ {0,...,8}:
    1. Encode logical |+⟩_L into 9 boundary qubits
       → state: (|000000000⟩ + |111111111⟩) / √2
    2. Erase qubits in E by resetting them to |0⟩ (quantum erasure channel)
    3. Apply the 9→3→1 majority voter decoder
    4. Measure the bulk output qubit
    5. Record P(bulk=1) — for a correctly recovered |+⟩, P(bulk=1) ≈ 0.5

RECONSTRUCTION FIDELITY METRIC
-------------------------------
For logical |+⟩_L, the ideal output is:
    P(0) = 0.5,  P(1) = 0.5

After erasure and recovery:
    - Perfect recovery → P(1) ≈ 0.5
    - Failed recovery  → P(1) → 0 (decoder always outputs |0⟩)

We define the reconstruction fidelity as:
    F = 1 - |P(1) - 0.5| / 0.5  ∈ [0, 1]

F = 1 means perfect reconstruction; F = 0 means complete failure.

RT FORMULA COMPARISON
---------------------
The discrete RT formula predicts reconstruction success based on the
min-cut of the surviving region. We compare:
    - RT prediction: reconstruction_ok (True/False)
    - Experimental F: high (>0.8) or low (<0.3)

Agreement between RT and experiment validates the holographic interpretation.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from qvoting.holographic.encoder import build_holographic_circuit
from qvoting.holographic.rt_formula import erasure_threshold_table, rt_prediction


def erase_qubits(
    qc: QuantumCircuit,
    boundary_register_index: int,
    qubits_to_erase: list[int],
) -> QuantumCircuit:
    """
    Apply quantum erasure to specified boundary qubits in-place.

    Erasure is modeled as a quantum reset: the qubit's state is
    discarded (traced out) and replaced with |0⟩. This is the
    standard model for photon loss, qubit loss, or leakage errors.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to modify (will be modified in place via compose).
    boundary_register_index : int
        The index of the boundary register in qc.qregs.
    qubits_to_erase : list[int]
        Indices [0..8] of boundary qubits to erase.

    Returns
    -------
    QuantumCircuit
        Modified circuit (same object, modified in place).
    """
    breg = qc.qregs[boundary_register_index]
    for idx in qubits_to_erase:
        qc.reset(breg[idx])
    return qc


def run_erasure_sweep(
    shots: int = 2048,
    max_erased: int = 9,
    backend=None,
) -> list[dict]:
    """
    Run the full holographic erasure experiment across all erasure patterns.

    For each possible subset of boundary qubits to erase, we:
        1. Build the encode→erase→decode circuit
        2. Run it on the simulator (or IBM backend)
        3. Compute reconstruction fidelity
        4. Compare to RT formula prediction

    Parameters
    ----------
    shots : int
        Number of measurement shots per circuit.
    max_erased : int
        Maximum number of qubits to erase (default: all 9).
    backend : optional
        Qiskit backend. Defaults to AerSimulator with statevector method.

    Returns
    -------
    list[dict]
        One entry per erasure pattern, with keys:
            erased_qubits       : list of erased qubit indices
            n_erased            : number of erased qubits
            counts              : raw measurement counts {'0': n, '1': m}
            P1                  : P(bulk = 1)
            fidelity            : reconstruction fidelity F ∈ [0,1]
            rt_ok               : RT formula prediction (True/False)
            rt_agrees           : whether RT matches experiment
    """
    if backend is None:
        backend = AerSimulator(method="statevector")

    rt_table = {
        tuple(row["erased_qubits"]): row
        for row in erasure_threshold_table()
    }

    results = []
    total_patterns = sum(
        len(list(combinations(range(9), k)))
        for k in range(max_erased + 1)
    )
    print(f"Running {total_patterns} erasure patterns ({shots} shots each)...")

    pattern_count = 0
    for n_erased in range(max_erased + 1):
        for erased in combinations(range(9), n_erased):
            erased_list = list(erased)

            # Build circuit: encode |+⟩_L → erase → decode
            qc = build_holographic_circuit(
                logical_state="plus",
                qubits_to_erase=erased_list,
            )

            # Execute
            from qiskit import transpile
            qc_t = transpile(qc, backend, optimization_level=1)
            job = backend.run(qc_t, shots=shots)
            counts = job.result().get_counts()

            n0 = counts.get("0", 0)
            n1 = counts.get("1", 0)
            p1 = n1 / (n0 + n1)

            # Fidelity: how close is P(1) to the ideal 0.5?
            fidelity = 1.0 - abs(p1 - 0.5) / 0.5

            # RT prediction
            rt_row = rt_table.get(tuple(sorted(erased_list)), {})
            rt_ok  = rt_row.get("reconstruction_ok", False)

            # Agreement: RT predicts success ↔ F > 0.7 threshold
            rt_agrees = (rt_ok and fidelity > 0.7) or (not rt_ok and fidelity <= 0.7)

            results.append({
                "erased_qubits": erased_list,
                "n_erased":      n_erased,
                "counts":        dict(counts),
                "P1":            round(p1, 4),
                "fidelity":      round(fidelity, 4),
                "rt_ok":         rt_ok,
                "rt_agrees":     rt_agrees,
            })

            pattern_count += 1
            if pattern_count % 50 == 0:
                print(f"  {pattern_count}/{total_patterns} patterns done...")

    print(f"Done. {len(results)} patterns evaluated.")
    return results


def summarize_sweep(results: list[dict]) -> dict:
    """
    Aggregate erasure sweep results by number of erased qubits.

    Returns
    -------
    dict mapping n_erased → {
        mean_fidelity, std_fidelity,
        rt_success_rate,   (fraction RT predicts success)
        exp_success_rate,  (fraction F > 0.7 experimentally)
        rt_agreement_rate, (fraction where RT matches experiment)
        n_patterns
    }
    """
    import math
    from collections import defaultdict

    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in results:
        buckets[row["n_erased"]].append(row)

    summary = {}
    for k, rows in sorted(buckets.items()):
        fids    = [r["fidelity"] for r in rows]
        mean_f  = sum(fids) / len(fids)
        var_f   = sum((f - mean_f) ** 2 for f in fids) / len(fids)
        std_f   = math.sqrt(var_f)

        rt_ok_count  = sum(1 for r in rows if r["rt_ok"])
        exp_ok_count = sum(1 for r in rows if r["fidelity"] > 0.7)
        agree_count  = sum(1 for r in rows if r["rt_agrees"])

        summary[k] = {
            "mean_fidelity":      round(mean_f, 4),
            "std_fidelity":       round(std_f, 4),
            "rt_success_rate":    round(rt_ok_count / len(rows), 4),
            "exp_success_rate":   round(exp_ok_count / len(rows), 4),
            "rt_agreement_rate":  round(agree_count / len(rows), 4),
            "n_patterns":         len(rows),
        }

    return summary


def run_rt_curve_experiment(
    shots: int = 2048,
    backend=None,
) -> list[dict]:
    """
    Measure entanglement entropy curve by sweeping over boundary region sizes.

    For each boundary region A, we prepare the encoded boundary state and
    measure the marginal distribution on A, then estimate von Neumann entropy
    via the classical proxy: Shannon entropy of the marginal counts.

    This gives an experimental proxy S_exp(A) to compare with S_RT(A).

    Note: true von Neumann entropy requires state tomography. Here we use
    the Shannon entropy of the measurement distribution as a proxy.
    Classical proxy: H(P) where P is the distribution over 2^|A| outcomes.

    Parameters
    ----------
    shots : int
        Shots per circuit.
    backend : optional
        Qiskit backend.

    Returns
    -------
    list[dict] with keys:
        region, S_RT, S_exp_proxy, agreement
    """
    import math
    if backend is None:
        backend = AerSimulator(method="statevector")

    results = []
    boundary_qubits = list(range(9))

    # Sample representative regions of each size (not all 512)
    sample_regions = []
    for size in range(10):
        all_of_size = list(combinations(boundary_qubits, size))
        # Take up to 10 representatives per size
        step = max(1, len(all_of_size) // 10)
        sample_regions.extend(all_of_size[::step])

    print(f"Measuring RT entropy curve for {len(sample_regions)} boundary regions...")

    for region in sample_regions:
        region_set = set(region)
        s_rt       = len({g for g, m in [(0,[0,1,2]),(1,[3,4,5]),(2,[6,7,8])]
                          if region_set & set(m)})

        # Handle empty region: no qubits to measure → entropy = 0
        if len(region) == 0:
            s_exp = 0.0
        else:
            # Build circuit: just encode, measure the boundary region A
            boundary = QuantumRegister(9, "b")
            creg     = ClassicalRegister(len(region), "meas")
            qc       = QuantumCircuit(boundary, creg)
            qc.h(boundary[0])
            for i in range(1, 9):
                qc.cx(boundary[0], boundary[i])
            for k, q in enumerate(sorted(region)):
                qc.measure(boundary[q], creg[k])

            from qiskit import transpile
            qc_t   = transpile(qc, backend, optimization_level=1)
            job    = backend.run(qc_t, shots=shots)
            counts = job.result().get_counts()

            # Shannon entropy as proxy for von Neumann entropy
            total  = sum(counts.values())
            probs  = [c / total for c in counts.values()]
            s_exp  = -sum(p * math.log2(p) for p in probs if p > 0)

        # For GHZ state, ideal entropy of any non-empty proper subset = 1 bit
        results.append({
            "region":    sorted(region),
            "region_size": len(region),
            "S_RT":      s_rt,
            "S_exp_proxy": round(s_exp, 4),
            "S_ideal_GHZ": 1.0 if 0 < len(region) < 9 else 0.0,
        })

    print("RT entropy curve measurement complete.")
    return results
