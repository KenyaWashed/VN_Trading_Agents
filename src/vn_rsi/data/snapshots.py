"""Immutable Parquet snapshot writer and checksum-verified reader.

Experiments MUST read snapshots only — never live APIs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vn_rsi.data.providers import RawMarketBundle


def _binary_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class MarketSnapshot:
    snapshot_id: str
    snapshot_dir: Path
    bars: pd.DataFrame
    fundamentals_pit: pd.DataFrame
    index_membership: pd.DataFrame
    benchmark: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def is_synthetic(self) -> bool:
        return bool(self.manifest.get("is_synthetic", False))

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest.get("manifest_content_sha256", ""))


class SnapshotManager:
    """Manages immutable versioned Parquet snapshots with SHA-256 integrity checks."""

    FILE_NAMES = {
        "bars": "bars.parquet",
        "fundamentals_pit": "fundamentals_pit.parquet",
        "index_membership": "index_membership.parquet",
        "benchmark": "benchmark.parquet",
    }

    def __init__(self, snapshot_root: str | Path = "data/snapshots") -> None:
        self.snapshot_root = Path(snapshot_root)

    def write_snapshot(
        self,
        snapshot_id: str,
        bundle: RawMarketBundle,
        config_metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> MarketSnapshot:
        snap_dir = self.snapshot_root / snapshot_id
        manifest_path = snap_dir / "manifest.json"
        if manifest_path.exists() and not overwrite:
            return self.load_snapshot(snapshot_id)

        snap_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "bars": bundle.bars,
            "fundamentals_pit": bundle.fundamentals_pit,
            "index_membership": bundle.index_membership,
            "benchmark": bundle.benchmark,
        }
        checksums: dict[str, str] = {}
        row_counts: dict[str, int] = {}

        for key, fname in self.FILE_NAMES.items():
            fpath = snap_dir / fname
            tables[key].to_parquet(fpath, index=False)
            checksums[fname] = _binary_sha256(fpath)
            row_counts[key] = int(len(tables[key]))

        core_manifest: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "provider_name": bundle.provider_name,
            "is_synthetic": bundle.is_synthetic,
            "row_counts": row_counts,
            "symbols_count": int(bundle.bars["symbol"].nunique()),
            "date_min": str(bundle.bars["date"].min())[:10],
            "date_max": str(bundle.bars["date"].max())[:10],
            "checksums_sha256": checksums,
            "reconciliation_summary": bundle.reconciliation_summary,
            "intraday_coverage_report": bundle.intraday_coverage_report,
            "config_metadata": config_metadata or {},
        }
        digest = hashlib.sha256(
            json.dumps(core_manifest["checksums_sha256"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        core_manifest["manifest_content_sha256"] = digest
        manifest_path.write_text(json.dumps(core_manifest, indent=2, sort_keys=True), encoding="utf-8")
        return self.load_snapshot(snapshot_id)

    def load_snapshot(self, snapshot_id: str) -> MarketSnapshot:
        snap_dir = self.snapshot_root / snapshot_id
        manifest_path = snap_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Snapshot manifest not found at {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_checksums: dict[str, str] = manifest.get("checksums_sha256", {})

        loaded: dict[str, pd.DataFrame] = {}
        for key, fname in self.FILE_NAMES.items():
            fpath = snap_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing snapshot table {fpath}")
            actual = _binary_sha256(fpath)
            expected = expected_checksums.get(fname)
            if actual != expected:
                raise RuntimeError(
                    f"Immutable snapshot checksum mismatch for {fpath}: expected {expected}, got {actual}"
                )
            df = pd.read_parquet(fpath)
            for dt_col in ("date", "fiscal_period_end", "announcement_date", "as_of_date"):
                if dt_col in df.columns and len(df) > 0:
                    df[dt_col] = pd.to_datetime(df[dt_col])
            loaded[key] = df

        return MarketSnapshot(
            snapshot_id=snapshot_id,
            snapshot_dir=snap_dir,
            bars=loaded["bars"],
            fundamentals_pit=loaded["fundamentals_pit"],
            index_membership=loaded["index_membership"],
            benchmark=loaded["benchmark"],
            manifest=manifest,
        )
