"""
qvoting.holographic.tomography
-------------------------------
Single-qubit state tomography utilities for holographic QEC experiments.

Enables characterization of the bulk qubit state beyond just P(1):
- Measure in Z, X, Y bases to reconstruct the full Bloch vector
- Compute state fidelity F = (1 + n_est · n_ideal) / 2
- Compare classical vs quantum information recovery

Key finding: the standard metric F = 1 - |P(1) - 0.5| / 0.5 only captures
the Z component of the Bloch vector. For |+>_L inputs, P(1) = 0.5 regardless
of whether quantum coherence is preserved, giving misleading F=1 results.
Full state tomography reveals that the [[9,1,3]] repetition code does NOT
preserve quantum phase information through group erasure.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ── Ideal Bloch vectors for the supported logical states ─────────────────────

_IDEAL_BLOCH: Dict[str, Tuple[float, float, float]] = {
    "zero":  (0.0,  0.0, +1.0),   # |0>  → north pole
    "one":   (0.0,  0.0, -1.0),   # |1>  → south pole
    "plus":  (+1.0, 0.0,  0.0),   # |+>  → +X equator
    "minus": (-1.0, 0.0,  0.0),   # |->  → -X equator
    "iplus": (0.0, +1.0,  0.0),   # |i>  → +Y equator
    "iminus": (0.0, -1.0, 0.0),   # |-i> → -Y equator
}


def ideal_bloch_vector(logical_state: str) -> Dict[str, float]:
    """
    Ideal (noise-free) Bloch vector for the given logical input state.

    Parameters
    ----------
    logical_state : str
        One of 'zero', 'one', 'plus', 'minus', 'iplus', 'iminus'.

    Returns
    -------
    dict with keys 'X', 'Y', 'Z' — expectation values in [-1, +1].
    """
    if logical_state not in _IDEAL_BLOCH:
        raise ValueError(
            f"Unknown logical_state {logical_state!r}. "
            f"Choose from {list(_IDEAL_BLOCH)}"
        )
    x, y, z = _IDEAL_BLOCH[logical_state]
    return {"X": x, "Y": y, "Z": z}


def bloch_from_counts(
    counts_Z: Dict[str, int],
    counts_X: Dict[str, int],
    counts_Y: Dict[str, int],
    shots: int,
) -> Dict[str, float]:
    """
    Estimate the Bloch vector from three single-qubit measurement bases.

    Each measurement circuit ends with a classical bit 'bulk'. The
    convention is:
      Z basis : measure directly
      X basis : H before measure   (maps |+>→|0>, |->→|1>)
      Y basis : Sdg+H before measure (maps |i>→|0>, |-i>→|1>)

    Expectation values: <O> = 1 - 2·P(outcome="1")

    Parameters
    ----------
    counts_Z, counts_X, counts_Y : dict
        Raw counts dicts {bitstring: count} for each basis.
        The relevant bit is the last character of each bitstring.
    shots : int
        Total shots per basis measurement.

    Returns
    -------
    dict with keys 'X', 'Y', 'Z' — estimated Bloch vector components.
    """
    def _p1(counts: Dict[str, int]) -> float:
        n1 = sum(v for k, v in counts.items() if k[-1] == "1")
        return n1 / shots if shots > 0 else 0.5

    return {
        "Z": 1.0 - 2.0 * _p1(counts_Z),
        "X": 1.0 - 2.0 * _p1(counts_X),
        "Y": 1.0 - 2.0 * _p1(counts_Y),
    }


def state_fidelity(
    bloch_est: Dict[str, float],
    bloch_ideal: Dict[str, float],
) -> float:
    """
    State fidelity between an estimated and ideal pure qubit state.

    For a pure target state |psi> with Bloch vector n_ideal and an
    estimated state rho with Bloch vector n_est:

        F = <psi|rho|psi> = (1 + n_est · n_ideal) / 2

    Parameters
    ----------
    bloch_est : dict with keys 'X', 'Y', 'Z'
        Estimated Bloch vector from tomography measurements.
    bloch_ideal : dict with keys 'X', 'Y', 'Z'
        Ideal Bloch vector (from ideal_bloch_vector()).

    Returns
    -------
    float in [0, 1]. F=1 → perfect recovery; F=0.5 → completely mixed.
    """
    dot = sum(bloch_est[k] * bloch_ideal[k] for k in ("X", "Y", "Z"))
    return float(np.clip((1.0 + dot) / 2.0, 0.0, 1.0))


def holographic_fidelity_proxy(counts: Dict[str, int], shots: int) -> float:
    """
    Original holographic fidelity metric: F = 1 - |P(1) - 0.5| / 0.5.

    This is the metric used in the first hardware experiment. It only
    captures the Z component of the Bloch vector and gives F=1 whenever
    P(1)=0.5, even for a completely mixed state.

    Kept here for direct comparison with state_fidelity().
    """
    p1 = sum(v for k, v in counts.items() if k[-1] == "1") / shots
    return float(1.0 - abs(p1 - 0.5) / 0.5)
