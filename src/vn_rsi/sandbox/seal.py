"""Evaluator SHA-256 tamper seal, 3-zone data boundary guard, and append-only Sealed Test Ledger."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vn_rsi.utils.repro import compute_tree_sha256


class TamperDetectedError(RuntimeError):
    """Raised if human-owned sandbox, evaluator, or config files change during a run."""


class DataZoneAccessError(PermissionError):
    """Raised when an agent attempts unauthorized direct access to validation or sealed_test bars."""


class SealedTestReuseError(PermissionError):
    """Raised when attempting to evaluate an already-scored candidate hash on the sealed test set."""


class EvaluatorSeal:
    """Locks SHA-256 hash of human-owned evaluation code and market/sandbox configs."""

    def __init__(self, protected_paths: list[str | Path]) -> None:
        self.protected_paths = [Path(p) for p in protected_paths]
        self.reference_hash = compute_tree_sha256(self.protected_paths)

    def verify_intact(self) -> str:
        current = compute_tree_sha256(self.protected_paths)
        if current != self.reference_hash:
            raise TamperDetectedError(
                f"Evaluator tamper check FAILED! Expected {self.reference_hash}, got {current}"
            )
        return current


class SealedTestLedger:
    """Append-only ledger refusing a second evaluation of the same frozen candidate config on sealed_test."""

    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _seen_hashes(self) -> set[str]:
        if not self.ledger_path.exists():
            return set()
        seen: set[str] = set()
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                seen.add(str(rec["candidate_hash"]))
        return seen

    def assert_and_record_access(
        self,
        candidate_hash: str,
        candidate_label: str,
        run_id: str,
        evaluator_seal_hash: str,
    ) -> dict[str, Any]:
        seen = self._seen_hashes()
        if candidate_hash in seen:
            raise SealedTestReuseError(
                f"Sealed test set access DENIED: candidate_hash '{candidate_hash}' ({candidate_label}) "
                f"was already scored in {self.ledger_path}. Sealed test can only be scored ONCE per frozen config."
            )
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "candidate_hash": candidate_hash,
            "candidate_label": candidate_label,
            "run_id": run_id,
            "evaluator_seal_hash": evaluator_seal_hash,
        }
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record


class DataZoneGuard:
    """Enforces strict three-zone partitioning (`search`, `validation`, `sealed_test`)."""

    def __init__(self, zones_cfg: dict[str, dict[str, str]]) -> None:
        self.zones_cfg = zones_cfg
        self._validate_non_overlapping()

    def _validate_non_overlapping(self) -> None:
        s_end = pd.Timestamp(self.zones_cfg["search"]["end"])
        v_start = pd.Timestamp(self.zones_cfg["validation"]["start"])
        v_end = pd.Timestamp(self.zones_cfg["validation"]["end"])
        t_start = pd.Timestamp(self.zones_cfg["sealed_test"]["start"])
        if not (s_end < v_start and v_end < t_start):
            raise ValueError(
                f"Data zones must be strictly chronological and non-overlapping: "
                f"search_end={s_end}, val=[{v_start}, {v_end}], test_start={t_start}"
            )

    def get_agent_visible_bars(self, bars: pd.DataFrame, requested_zone: str = "search") -> pd.DataFrame:
        """Agents may ONLY inspect raw bars in the `search` zone."""
        if requested_zone != "search":
            raise DataZoneAccessError(
                f"Agent requested direct bar access to '{requested_zone}'. "
                "Direct data inspection is restricted to 'search'; validation/test return scalar scores via Evaluator only."
            )
        return self._slice_zone(bars, "search")

    def _slice_zone(self, df: pd.DataFrame, zone: str) -> pd.DataFrame:
        z = self.zones_cfg[zone]
        start = pd.Timestamp(z["start"])
        end = pd.Timestamp(z["end"])
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[mask].copy()
