"""
qvoting.core.logging
---------------------
Persistent job logging for IBM Quantum jobs.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class JobLogger:
    """
    Persistent JSON-based logger for IBM Quantum jobs.

    Survives kernel restarts — jobs can always be retrieved by ID.

    Examples
    --------
    >>> logger = JobLogger("my_jobs.json")
    >>> logger.register(job_id="abc123", description="Bell state", backend="ibm_torino")
    >>> logger.update_status("abc123", "DONE")
    >>> pending = logger.pending()
    """

    TERMINAL_STATES = ("DONE", "CANCELLED", "ERROR", "FAILED")

    def __init__(self, log_file: str = "quantum_jobs_log.json") -> None:
        self.log_file = Path(log_file)
        self.jobs: Dict[str, dict] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────
    def _load(self) -> Dict[str, dict]:
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.jobs, f, indent=2, ensure_ascii=False)

    # ── CRUD ──────────────────────────────────────────────────────────────
    def register(
        self,
        job_id: str,
        description: str,
        backend: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Register a new job."""
        self.jobs[job_id] = {
            "description": description,
            "backend": backend,
            "timestamp": datetime.now().isoformat(),
            "status": "QUEUED",
            "metadata": metadata or {},
        }
        self._save()

    def update_status(self, job_id: str, status: str) -> None:
        """Update job status."""
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found in log")
        self.jobs[job_id]["status"] = str(status)
        self.jobs[job_id]["last_updated"] = datetime.now().isoformat()
        self._save()

    # ── Queries ───────────────────────────────────────────────────────────
    def pending(self) -> List[Tuple[str, dict]]:
        """Return all jobs that are not in a terminal state."""
        return [
            (jid, info)
            for jid, info in self.jobs.items()
            if info["status"] not in self.TERMINAL_STATES
        ]

    def completed(self) -> List[Tuple[str, dict]]:
        """Return all completed (DONE) jobs."""
        return [
            (jid, info)
            for jid, info in self.jobs.items()
            if info["status"] == "DONE"
        ]

    # ── Display ───────────────────────────────────────────────────────────
    def summary(self) -> None:
        """Print a formatted summary of all jobs."""
        print(f"\n{'─' * 72}")
        print(f"{'Job ID':<30} {'Backend':<15} {'Status':<12} {'Date'}")
        print(f"{'─' * 72}")
        for jid, info in self.jobs.items():
            date = info["timestamp"][:10]
            print(f"{jid:<30} {info['backend']:<15} {info['status']:<12} {date}")
        print(f"{'─' * 72}")
        print(f"Total: {len(self.jobs)} | Pending: {len(self.pending())} | Done: {len(self.completed())}\n")

    def __repr__(self) -> str:  # pragma: no cover
        return f"JobLogger(file={self.log_file}, jobs={len(self.jobs)})"
