"""Structured JSONL experiment logger with provenance metadata for full paper traceability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunLogger:
    """Writes per-run structured JSONL event logs and appends run summaries to a central registry."""

    def __init__(
        self,
        run_id: str,
        logs_dir: str | Path = "logs",
        git_commit: str = "unknown",
        config_hash: str = "unknown",
        evaluator_seal_hash: str = "unsealed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.logs_dir = Path(logs_dir)
        self.runs_dir = self.logs_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_file = self.runs_dir / f"{run_id}.jsonl"
        self.registry_file = self.logs_dir / "experiment_registry.jsonl"
        self.git_commit = git_commit
        self.config_hash = config_hash
        self.evaluator_seal_hash = evaluator_seal_hash
        self.metadata = metadata or {}
        self.log_event("run_started", {"metadata": self.metadata})

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "evaluator_seal_hash": self.evaluator_seal_hash,
            "payload": payload,
        }
        with self.run_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def finalize_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "evaluator_seal_hash": self.evaluator_seal_hash,
            **self.metadata,
            **summary,
        }
        self.log_event("run_finalized", entry)
        with self.registry_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return entry
