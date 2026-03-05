"""
qvoting.nisq_selector
---------------------
Backend selector automático basado en el Quality Signal Q(t) del NISQ Health Monitor.

Uso rápido
----------
>>> from qvoting.nisq_selector import NISQSelector
>>> selector = NISQSelector(service, candidates=["ibm_torino","ibm_fez","ibm_marrakesh"])
>>> backend = selector.best()          # devuelve el backend con mayor Q(t) disponible
>>> counts  = execute_circuit(qc, backend, shots=1024)
"""
from __future__ import annotations

import json
import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Pesos y umbral por defecto ────────────────────────────────────────────
_W_T1, _W_T2, _W_RO, _W_GATE = 0.25, 0.25, 0.30, 0.20
_QT_THRESHOLD = 0.50


def _safe(fn, *args, scale: float = 1.0):
    """Llama fn(*args) devolviendo el valor escalado, o None si lanza excepción."""
    try:
        val = fn(*args)
        return val * scale if val is not None else None
    except Exception:
        return None


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)


def _compute_qt(
    df_qubits: pd.DataFrame,
    df_gates: pd.DataFrame,
    backend_names: list[str],
    w_t1: float = _W_T1,
    w_t2: float = _W_T2,
    w_ro: float = _W_RO,
    w_gate: float = _W_GATE,
    threshold: float = _QT_THRESHOLD,
) -> pd.DataFrame:
    """Calcula Q(t) y b(t) para cada backend."""
    rows = []
    for name in backend_names:
        dq = df_qubits[df_qubits["backend"] == name]
        dg = df_gates[df_gates["backend"] == name] if len(df_gates) > 0 else pd.DataFrame()
        rows.append({
            "backend":    name,
            "T1_med":     dq["T1_us"].median()        if len(dq) else np.nan,
            "T2_med":     dq["T2_us"].median()        if len(dq) else np.nan,
            "ro_mean":    dq["readout_error"].mean()  if len(dq) else np.nan,
            "gate_mean":  dg["error"].dropna().mean() if len(dg) > 0 else np.nan,
        })
    df = pd.DataFrame(rows)

    T1n = _minmax(df["T1_med"].fillna(df["T1_med"].min()))
    T2n = _minmax(df["T2_med"].fillna(df["T2_med"].min()))
    ron = 1 - _minmax(df["ro_mean"].fillna(df["ro_mean"].max()))

    if df["gate_mean"].notna().any():
        gn = 1 - _minmax(df["gate_mean"].fillna(df["gate_mean"].max()))
        Qt = w_t1 * T1n + w_t2 * T2n + w_ro * ron + w_gate * gn
    else:
        scale = 1.0 / (w_t1 + w_t2 + w_ro)
        Qt = (w_t1 * T1n + w_t2 * T2n + w_ro * ron) * scale

    df["Qt"]     = Qt.values
    df["binary"] = (df["Qt"] >= threshold).astype(int)
    return df


def _collect_properties(service, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrae propiedades físicas de los backends vía IBM Runtime API."""
    all_q, all_g = [], []
    for name in names:
        try:
            b, props = service.backend(name), service.backend(name).properties()
            for q in range(b.num_qubits):
                all_q.append({
                    "backend":       name,
                    "qubit":         q,
                    "T1_us":         _safe(props.t1,            q, scale=1e6),
                    "T2_us":         _safe(props.t2,            q, scale=1e6),
                    "readout_error": _safe(props.readout_error, q),
                })
            for gate in [g for g in props.gates if len(g.qubits) == 2]:
                all_g.append({
                    "backend": name,
                    "error":   _safe(props.gate_error, gate.gate, gate.qubits),
                })
        except Exception:
            pass   # backend no disponible → omitir
    return pd.DataFrame(all_q), pd.DataFrame(all_g)


class NISQSelector:
    """
    Selector automático de backend basado en el Quality Signal Q(t).

    Parameters
    ----------
    service : QiskitRuntimeService
        Servicio activo de IBM Quantum.
    candidates : list[str]
        Nombres de los backends candidatos a evaluar.
    threshold : float
        Umbral mínimo de Q(t) para considerar un backend "usable" (b=1).
    weights : tuple[float, float, float, float]
        Pesos (w_T1, w_T2, w_readout, w_gate). Se normalizan automáticamente.
    cache_minutes : float
        Si los datos tienen menos de `cache_minutes` minutos, no se reevalúan.
    snapshot_dir : str | Path, optional
        Directorio donde guardar snapshots JSON automáticamente.

    Examples
    --------
    >>> selector = NISQSelector(service, ["ibm_torino", "ibm_fez", "ibm_marrakesh"])
    >>> backend  = selector.best()
    >>> print(selector.report())
    """

    def __init__(
        self,
        service,
        candidates: list[str],
        threshold: float = _QT_THRESHOLD,
        weights: tuple[float, float, float, float] = (_W_T1, _W_T2, _W_RO, _W_GATE),
        cache_minutes: float = 30.0,
        snapshot_dir: Optional[str | Path] = None,
    ):
        self.service       = service
        self.candidates    = list(candidates)
        self.threshold     = threshold
        self._w            = weights
        self.cache_minutes = cache_minutes
        self.snapshot_dir  = Path(snapshot_dir) if snapshot_dir else None

        self._df_qt: Optional[pd.DataFrame] = None
        self._last_eval: Optional[datetime.datetime] = None

    # ── Evaluación ──────────────────────────────────────────────────────

    def evaluate(self, force: bool = False) -> pd.DataFrame:
        """
        Evalúa Q(t) para todos los candidatos.

        Parameters
        ----------
        force : bool
            Si True, ignora la caché y reevalúa aunque sea reciente.

        Returns
        -------
        pd.DataFrame con columnas: backend, Qt, binary, T1_med, T2_med, ro_mean, gate_mean.
        """
        # Comprobar caché
        if (
            not force
            and self._last_eval is not None
            and self._df_qt is not None
        ):
            elapsed = (datetime.datetime.now(timezone.utc).replace(tzinfo=None) - self._last_eval).total_seconds() / 60
            if elapsed < self.cache_minutes:
                return self._df_qt

        w_t1, w_t2, w_ro, w_gate = self._w
        dq, dg = _collect_properties(self.service, self.candidates)

        if len(dq) == 0:
            raise RuntimeError(
                "No se pudieron obtener propiedades de ningún backend candidato."
            )

        available = dq["backend"].unique().tolist()
        self._df_qt = _compute_qt(dq, dg, available, w_t1, w_t2, w_ro, w_gate, self.threshold)
        self._last_eval = datetime.datetime.now(timezone.utc).replace(tzinfo=None)

        # Guardar snapshot opcional
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = self._last_eval.strftime("%Y%m%d_%H%M%S")
            out = self.snapshot_dir / f"selector_snapshot_{ts}.json"
            with open(out, "w") as f:
                json.dump({
                    "timestamp_utc": self._last_eval.isoformat(),
                    "threshold":     self.threshold,
                    "results":       self._df_qt[["backend", "Qt", "binary"]].to_dict(orient="records"),
                }, f, indent=2)

        return self._df_qt

    # ── Selección ────────────────────────────────────────────────────────

    def best(self, force: bool = False):
        """
        Devuelve el objeto backend IBM con el Q(t) más alto entre los usables (b=1).

        Raises
        ------
        RuntimeError
            Si ningún backend candidato supera el umbral.
        """
        df = self.evaluate(force=force)
        usable = df[df["binary"] == 1].sort_values("Qt", ascending=False)

        if len(usable) == 0:
            degraded = df.sort_values("Qt", ascending=False)
            msg = (
                f"Ningún backend supera el umbral Q(t)={self.threshold}.\n"
                f"  Estado actual:\n"
            )
            for _, r in degraded.iterrows():
                msg += f"    {r['backend']}: Q(t)={r['Qt']:.4f}\n"
            raise RuntimeError(msg)

        best_name = usable.iloc[0]["backend"]
        return self.service.backend(best_name)

    def ranked(self, force: bool = False) -> list[tuple[str, float, int]]:
        """
        Devuelve la lista de (backend_name, Qt, binary) ordenada por Q(t) descendente.
        """
        df = self.evaluate(force=force)
        return [
            (r["backend"], float(r["Qt"]), int(r["binary"]))
            for _, r in df.sort_values("Qt", ascending=False).iterrows()
        ]

    def is_usable(self, backend_name: str, force: bool = False) -> bool:
        """Devuelve True si el backend tiene b(t)=1."""
        df = self.evaluate(force=force)
        row = df[df["backend"] == backend_name]
        if len(row) == 0:
            raise ValueError(f"Backend '{backend_name}' no encontrado en candidatos.")
        return bool(row.iloc[0]["binary"] == 1)

    # ── Reporte ──────────────────────────────────────────────────────────

    def report(self, force: bool = False) -> str:
        """Devuelve una cadena formateada con el estado de todos los candidatos."""
        df = self.evaluate(force=force)
        ts = self._last_eval.strftime("%Y-%m-%d %H:%M:%S UTC") if self._last_eval else "?"
        lines = [
            f"NISQ Health Monitor — {ts}",
            f"Umbral Q(t) = {self.threshold}",
            "-" * 55,
            f"  {'Backend':<22s}  {'Q(t)':>6s}  {'b(t)':>4s}  {'T1[µs]':>8s}  {'RO err':>7s}",
            "-" * 55,
        ]
        for _, r in df.sort_values("Qt", ascending=False).iterrows():
            badge = "OK" if r["binary"] == 1 else "--"
            t1 = f"{r['T1_med']:.1f}" if not np.isnan(r["T1_med"]) else "  N/A"
            ro = f"{r['ro_mean']:.4f}" if not np.isnan(r["ro_mean"]) else "  N/A"
            lines.append(
                f"  {r['backend']:<22s}  {r['Qt']:>6.4f}  {badge}     {t1:>8s}  {ro:>7s}"
            )
        lines.append("-" * 55)
        return "\n".join(lines)
