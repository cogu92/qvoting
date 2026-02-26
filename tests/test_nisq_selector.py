"""
Tests for qvoting.nisq_selector.NISQSelector

Uses a mock QiskitRuntimeService so no real IBM credentials are needed.
"""
from __future__ import annotations

import datetime
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from qvoting.nisq_selector import (
    NISQSelector,
    _safe,
    _minmax,
    _compute_qt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_props(t1=100e-6, t2=80e-6, ro=0.03, gate_err=0.02, n_qubits=5):
    """Build a minimal mock BackendProperties-like object."""
    props = MagicMock()
    props.t1.return_value          = t1
    props.t2.return_value          = t2
    props.readout_error.return_value = ro
    props.gate_error.return_value  = gate_err

    # Fake 2-qubit gates
    g = MagicMock()
    g.gate   = "cx"
    g.qubits = [0, 1]
    props.gates = [g]

    return props


def _make_backend(name, n_qubits=5, **prop_kwargs):
    b = MagicMock()
    b.name       = name
    b.num_qubits = n_qubits
    b.properties.return_value = _make_props(**prop_kwargs)
    return b


def _make_service(backends: list):
    """Service whose .backend(name) returns the matching mock."""
    svc = MagicMock()
    lookup = {b.name: b for b in backends}
    svc.backend.side_effect = lambda name: lookup[name]
    return svc


# Three distinct backends: A best, B mid, C worst
_B_BEST = _make_backend("b_best",  t1=200e-6, t2=150e-6, ro=0.01, gate_err=0.01)
_B_MID  = _make_backend("b_mid",   t1=150e-6, t2=100e-6, ro=0.03, gate_err=0.03)
_B_WORST= _make_backend("b_worst", t1=80e-6,  t2=60e-6,  ro=0.08, gate_err=0.08)
_SVC    = _make_service([_B_BEST, _B_MID, _B_WORST])
_CANDIDATES = ["b_best", "b_mid", "b_worst"]


# ── Unit tests: helper functions ──────────────────────────────────────────

class TestSafeHelper:
    def test_returns_scaled_value(self):
        assert _safe(lambda: 1.0, scale=1e6) == pytest.approx(1e6)

    def test_returns_none_on_exception(self):
        def boom(): raise RuntimeError("no data")
        assert _safe(boom) is None

    def test_returns_none_when_value_is_none(self):
        assert _safe(lambda: None) is None


class TestMinmax:
    def test_range_zero_to_one(self):
        s = pd.Series([10.0, 20.0, 30.0])
        n = _minmax(s)
        assert n.min() == pytest.approx(0.0)
        assert n.max() == pytest.approx(1.0)

    def test_constant_returns_half(self):
        s = pd.Series([5.0, 5.0, 5.0])
        n = _minmax(s)
        assert n.values == pytest.approx([0.5, 0.5, 0.5])


class TestComputeQt:
    def _make_dfs(self):
        dq = pd.DataFrame([
            {"backend": "A", "qubit": 0, "T1_us": 200.0, "T2_us": 150.0, "readout_error": 0.01},
            {"backend": "B", "qubit": 0, "T1_us": 100.0, "T2_us":  80.0, "readout_error": 0.05},
        ])
        dg = pd.DataFrame([
            {"backend": "A", "error": 0.01},
            {"backend": "B", "error": 0.05},
        ])
        return dq, dg

    def test_better_backend_has_higher_qt(self):
        dq, dg = self._make_dfs()
        df = _compute_qt(dq, dg, ["A", "B"])
        qt_A = df[df["backend"] == "A"]["Qt"].values[0]
        qt_B = df[df["backend"] == "B"]["Qt"].values[0]
        assert qt_A > qt_B

    def test_qt_in_zero_one(self):
        dq, dg = self._make_dfs()
        df = _compute_qt(dq, dg, ["A", "B"])
        assert df["Qt"].between(0.0, 1.0).all()

    def test_binary_threshold(self):
        dq, dg = self._make_dfs()
        df = _compute_qt(dq, dg, ["A", "B"], threshold=0.0)
        assert (df["binary"] == 1).all()

        df2 = _compute_qt(dq, dg, ["A", "B"], threshold=1.1)
        assert (df2["binary"] == 0).all()


# ── Integration tests: NISQSelector ──────────────────────────────────────

@pytest.fixture
def selector():
    return NISQSelector(_SVC, _CANDIDATES, threshold=0.5)


class TestNISQSelectorEvaluate:
    def test_returns_dataframe(self, selector):
        df = selector.evaluate(force=True)
        assert isinstance(df, pd.DataFrame)
        assert set(["backend", "Qt", "binary"]).issubset(df.columns)

    def test_all_candidates_present(self, selector):
        df = selector.evaluate(force=True)
        assert set(df["backend"].tolist()) == set(_CANDIDATES)

    def test_cache_reuses_result(self, selector):
        df1 = selector.evaluate(force=True)
        df2 = selector.evaluate(force=False)   # should hit cache
        pd.testing.assert_frame_equal(df1, df2)

    def test_force_refreshes_cache(self, selector):
        selector.evaluate(force=True)
        t1 = selector._last_eval
        selector.evaluate(force=True)
        t2 = selector._last_eval
        assert t2 >= t1


class TestNISQSelectorBest:
    def test_returns_backend_object(self, selector):
        backend = selector.best(force=True)
        assert backend.name in _CANDIDATES

    def test_returns_highest_qt_usable(self, selector):
        backend = selector.best(force=True)
        df = selector._df_qt
        usable = df[df["binary"] == 1]
        assert backend.name == usable.sort_values("Qt", ascending=False).iloc[0]["backend"]

    def test_raises_when_all_degraded(self):
        sel = NISQSelector(_SVC, _CANDIDATES, threshold=2.0)  # impossible threshold
        with pytest.raises(RuntimeError, match="Ningún backend"):
            sel.best(force=True)


class TestNISQSelectorRanked:
    def test_returns_sorted_list(self, selector):
        ranking = selector.ranked(force=True)
        qt_values = [r[1] for r in ranking]
        assert qt_values == sorted(qt_values, reverse=True)

    def test_tuple_structure(self, selector):
        ranking = selector.ranked(force=True)
        for name, qt, bit in ranking:
            assert isinstance(name, str)
            assert 0.0 <= qt <= 1.0
            assert bit in (0, 1)


class TestNISQSelectorIsUsable:
    def test_known_backend(self, selector):
        selector.evaluate(force=True)
        result = selector.is_usable("b_best")
        assert isinstance(result, bool)

    def test_unknown_backend_raises(self, selector):
        selector.evaluate(force=True)
        with pytest.raises(ValueError, match="no encontrado"):
            selector.is_usable("nonexistent_backend")


class TestNISQSelectorReport:
    def test_report_contains_all_backends(self, selector):
        report = selector.report(force=True)
        for name in _CANDIDATES:
            assert name in report

    def test_report_contains_threshold(self, selector):
        report = selector.report(force=True)
        assert "0.5" in report

    def test_report_is_string(self, selector):
        assert isinstance(selector.report(force=True), str)


class TestNISQSelectorSnapshot:
    def test_snapshot_saved_to_dir(self, tmp_path, selector):
        sel = NISQSelector(_SVC, _CANDIDATES, threshold=0.5, snapshot_dir=tmp_path)
        sel.evaluate(force=True)
        snapshots = list(tmp_path.glob("selector_snapshot_*.json"))
        assert len(snapshots) == 1

    def test_snapshot_json_structure(self, tmp_path):
        import json
        sel = NISQSelector(_SVC, _CANDIDATES, threshold=0.5, snapshot_dir=tmp_path)
        sel.evaluate(force=True)
        snap = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert "timestamp_utc" in snap
        assert "threshold" in snap
        assert "results" in snap
        assert len(snap["results"]) == len(_CANDIDATES)
