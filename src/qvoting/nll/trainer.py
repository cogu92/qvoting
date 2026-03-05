"""
qvoting.nll.trainer
-------------------
SPSATrainer: SPSA optimizer for NLL parameters on quantum hardware.

SPSA (Simultaneous Perturbation Stochastic Approximation) estimates
gradients using only 2 circuit evaluations per iteration, regardless
of the number of parameters — ideal for noisy quantum hardware.

Update rule
-----------
  Θ_{k+1} = Θ_k - a_k * ĝ_k

where:
  ĝ_k = [L(Θ_k + c_k·Δ_k) - L(Θ_k - c_k·Δ_k)] / (2·c_k·Δ_k)
  Δ_k  ~ Bernoulli(±1)  (random perturbation vector)
  a_k  = a / (k + 1 + A)^alpha   (gain sequence)
  c_k  = c / (k + 1)^gamma       (perturbation sequence)

Recommended hyperparameters (Spall 1998):
  alpha = 0.602, gamma = 0.101
  a, c chosen from calibration
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .circuit import NeuralLinkedList, pattern_recovery_rate


class SPSATrainer:
    """
    SPSA optimizer for NLL trainable parameters.

    Parameters
    ----------
    nll : NeuralLinkedList
        The NLL to optimize.
    target_pattern : str
        Bitstring pattern to maximize probability of (e.g., "101").
    backend : AerSimulator | IBMBackend
        Execution backend.
    shots : int
        Shots per circuit evaluation. Recommended ≥ 2048 for hardware.
    n_iter : int
        Number of SPSA iterations.
    a : float
        Step size scale (SPSA hyperparameter). Start with 0.1-0.5.
    c : float
        Perturbation scale. Start with 0.1-0.3.
    A : float
        Stability constant. Typically 0.1 * n_iter.
    alpha : float
        Step size decay exponent (Spall 1998: 0.602).
    gamma : float
        Perturbation decay exponent (Spall 1998: 0.101).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        nll: NeuralLinkedList,
        target_pattern: str,
        backend,
        shots: int = 2048,
        n_iter: int = 50,
        a: float = 0.3,
        c: float = 0.2,
        A: float = 5.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        seed: Optional[int] = None,
    ):
        self.nll = nll
        self.target_pattern = target_pattern
        self.backend = backend
        self.shots = shots
        self.n_iter = n_iter
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.history: List[Dict] = []  # {iter, loss, params}

    def _loss(self, params: np.ndarray) -> float:
        """
        Cost function: L(Θ) = 1 - P(target_pattern).
        Lower is better (minimization objective).
        """
        from qvoting.core.execution import execute_circuit

        self.nll.set_params_flat(params)
        qc = self.nll.build_circuit(measure=True)
        counts = execute_circuit(qc, self.backend, shots=self.shots)
        p = pattern_recovery_rate(counts, self.target_pattern, self.shots)
        return 1.0 - p

    def _a_k(self, k: int) -> float:
        return self.a / (k + 1 + self.A) ** self.alpha

    def _c_k(self, k: int) -> float:
        return self.c / (k + 1) ** self.gamma

    def run(self, verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        Run SPSA optimization.

        Parameters
        ----------
        verbose : bool
            Print progress every 10 iterations.

        Returns
        -------
        (best_params, loss_history)
            best_params  : np.ndarray — optimal parameter vector [θ..., φ...]
            loss_history : list[float] — loss at each iteration
        """
        n_params = self.nll.n_params()
        params = self.nll.get_params().copy()
        loss_history: List[float] = []
        best_params = params.copy()
        best_loss = float("inf")

        if verbose:
            print(f"SPSATrainer: {n_params} params, {self.n_iter} iterations, "
                  f"{self.shots} shots, target='{self.target_pattern}'")

        for k in range(self.n_iter):
            ak = self._a_k(k)
            ck = self._c_k(k)

            # Random Bernoulli perturbation vector
            delta = self.rng.choice([-1.0, 1.0], size=n_params)

            # Two-point gradient estimate
            loss_plus  = self._loss(params + ck * delta)
            loss_minus = self._loss(params - ck * delta)
            grad_hat   = (loss_plus - loss_minus) / (2 * ck * delta)

            # Parameter update
            params = params - ak * grad_hat

            # Clip to [0, 2π] to keep angles meaningful
            params = np.mod(params, 2 * np.pi)

            # Evaluate at current (non-perturbed) params
            current_loss = self._loss(params)
            loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()

            self.history.append({
                "iter": k,
                "loss": current_loss,
                "p_target": 1.0 - current_loss,
                "a_k": ak,
                "c_k": ck,
                "params": params.tolist(),
            })

            if verbose and (k % 10 == 0 or k == self.n_iter - 1):
                print(f"  iter {k:3d}: loss={current_loss:.4f}  "
                      f"P(target)={1-current_loss:.4f}  "
                      f"best_P={1-best_loss:.4f}")

        # Restore best parameters to the NLL
        self.nll.set_params_flat(best_params)

        if verbose:
            print(f"\nFinal best P('{self.target_pattern}') = {1-best_loss:.4f}")

        return best_params, loss_history
