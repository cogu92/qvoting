# QVoting — Quantum Voting Framework

[![PyPI version](https://badge.fury.io/py/qvoting.svg)](https://badge.fury.io/py/qvoting)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-15%2F15%20passing-brightgreen.svg)]()

**Quantum voting circuits with integrated readout error mitigation and ZNE for NISQ hardware.**

Validated on IBM Quantum hardware (ibm_torino Eagle-r3 and ibm_fez Heron-r1).
Bell state fidelity: **97.27%** on ibm_torino, **93.65%** on ibm_fez.

```bash
pip install qvoting
```

For IBM Quantum hardware execution:
```bash
pip install qvoting[ibm]
```

---

## Quick Start

```python
from qvoting.voters import majority_voter
from qvoting.mitigation import apply_readout_mitigation
from qvoting.core import execute_circuit

# Build a 3-input majority voter
voter = majority_voter(num_inputs=3)
print(voter.draw())

# Run on local Aer simulator
counts = execute_circuit(voter, backend="aer", shots=1024)
print(counts)  # {'1': 1024}  (all inputs |1> -> majority = |1>)

# Apply readout error mitigation
counts_mitigated = apply_readout_mitigation(counts, calibration_counts={'0': 50, '1': 974})
```

---

## Package Structure

```
qvoting/
+-- core/
|   +-- circuits.py       <- Parity sub-circuits & multi-circuit load balancer
|   +-- execution.py      <- Unified backend (Aer simulator + IBM Quantum)
|   +-- logging.py        <- JobLogger for persistent IBM job tracking
+-- voters/
|   +-- majority.py       <- Toffoli majority voters (3 and 5 inputs)
|   +-- hierarchical.py   <- Hierarchical voter (9->3->1, 13 qubits)
+-- mitigation/
    +-- readout.py        <- Confusion matrix readout error mitigation
    +-- zne.py            <- Zero-Noise Extrapolation via gate folding
```

---

## Features

- **Quantum majority voters** - 3-input and 5-input Toffoli-based circuits
- **Hierarchical voting** - 9->3->1 reduction (13 qubits total)
- **Quantum load balancer** - parity sub-circuit distributes depth across sub-circuits O(n/k)
- **Readout error mitigation** - confusion matrix inversion (M tensor-n approximation)
- **Zero-Noise Extrapolation** - gate folding with linear regression intercept
- **Unified execution** - same API for Aer simulator and IBM Quantum hardware

---

## Hardware Benchmark Results

| Backend | Bell Fidelity | TVD | Device |
|---------|--------------|-----|--------|
| ibm_torino | **97.27%** | 0.0557 | Eagle-r3 (133q) |
| ibm_fez | 93.65% | 0.0918 | Heron-r1 (156q) |
| Improvement | +3.87 pp | -39.3% | - |

GHZ 3-qubit state on ibm_torino (2048 shots): TVD = 0.062, spurious states < 5%.

---

## Module Status

| Module | Implemented | Tests |
|--------|------------|-------|
| `core.circuits` | Yes | 5/5 |
| `core.execution` | Yes | - |
| `core.logging` | Yes | - |
| `voters.majority` | Yes | 6/6 |
| `voters.hierarchical` | Yes | - |
| `mitigation.readout` | Yes | 4/4 |
| `mitigation.zne` | Yes | - |
| **Total** | **15 tests** | **15/15** |

---

## Citation

If you use QVoting in your research, please cite:

```bibtex
@article{qvoting2026,
  title   = {Quantum Voting Circuits with Integrated Error Mitigation on NISQ Hardware},
  author  = {Nicolas},
  year    = {2026},
  journal = {[under review]},
  url     = {https://arxiv.org/abs/[TODO]}
}
```

---

## License

MIT - see LICENSE.
