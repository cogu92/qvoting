"""
qvoting.core.execution
----------------------
Unified execution layer for both Aer simulators and IBM hardware backends.
"""
from __future__ import annotations

from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.result import Counts


def execute_circuit(
    qc: QuantumCircuit,
    backend,
    shots: int = 1024,
    optimization_level: int = 1,
) -> Counts:
    """
    Execute a quantum circuit on any backend (AerSimulator or IBMBackend).

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to execute.
    backend : AerSimulator | IBMBackend
        Target backend.
    shots : int
        Number of measurement shots.
    optimization_level : int
        Transpiler optimization level (0-3).

    Returns
    -------
    Counts
        Measurement counts dictionary.
    """
    backend_type = type(backend).__name__

    if "Aer" in backend_type or "Simulator" in backend_type:
        # ── Aer simulator path ────────────────────────────────────────────
        from qiskit_aer import AerSimulator

        qc_t = transpile(qc, backend, optimization_level=optimization_level)
        job = backend.run(qc_t, shots=shots)
        result = job.result()
        return result.get_counts()

    else:
        # ── IBM hardware path ─────────────────────────────────────────────
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
        )
        isa_circuit = pm.run(qc)
        sampler = Sampler(backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]
        return pub_result.data.c.get_counts()
