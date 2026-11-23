"""Mandatory Data QA, Survivorship, Point-in-Time Fundamentals, and Shift-Leakage Tests (Phase 1 CI)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vn_rsi.data import (
    DataQASuite,
    SnapshotManager,
    SyntheticVNMarketProvider,
    load_market_rules,
)
from vn_rsi.data.qa import join_pit_fundamentals_causal


@pytest.fixture(scope="module")
def sample_env(tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp("vn_data_test")
    repo_root = Path(__file__).resolve().parents[1]
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    provider = SyntheticVNMarketProvider()
    bundle = provider.fetch_bundle(
        universe_tier="vn30",
        start_date="2021-01-04",
        end_date="2022-12-30",
        market_rules=rules,
        seed=12345,
    )
    mgr = SnapshotManager(tmp_dir / "snapshots")
    snap = mgr.write_snapshot("test_snap_v1", bundle)
    return rules, mgr, snap


def test_snapshot_manifest_and_tamper_detection(sample_env, tmp_path: Path) -> None:
    _, _, snap = sample_env
    assert snap.is_synthetic is True
    assert snap.manifest["symbols_count"] == 30
    assert len(snap.bars) > 1000

    # Copy snapshot to temp dir and corrupt bars.parquet to ensure checksum mismatch is caught
    bad_mgr = SnapshotManager(tmp_path / "corrupt_snaps")
    bad_dir = tmp_path / "corrupt_snaps" / "bad_v1"
    bad_dir.mkdir(parents=True)
    for item in snap.snapshot_dir.iterdir():
        (bad_dir / item.name).write_bytes(item.read_bytes())
    (bad_dir / "bars.parquet").write_bytes(b"corrupted_parquet_bytes")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bad_mgr.load_snapshot("bad_v1")


def test_qa_suite_survivorship_bands_and_pit_lag(sample_env) -> None:
    rules, _, snap = sample_env
    qa = DataQASuite(market_rules=rules, max_source_diff_pct=0.02)
    report = qa.run_all(snap, require_delisted=True)
    assert report.passed is True
    assert "ROS_DELIST" in report.checks["survivorship_coverage"]["delisted_symbols"]
    assert report.checks["pit_fundamentals_causality"]["min_announcement_lag_days"] >= 20

    # Verify causal PIT join never leaks a quarter before its announcement_date
    joined = join_pit_fundamentals_causal(snap.bars, snap.fundamentals_pit)
    non_null = joined.dropna(subset=["announcement_date"])
    assert (non_null["announcement_date"] <= non_null["date"]).all()


def test_shift_detector_passes_causal_and_catches_future_leak(sample_env) -> None:
    _, _, snap = sample_env
    mini = snap.bars[snap.bars["symbol"].isin(["VCB", "FPT"])].copy().reset_index(drop=True)

    def causal_ma10(df: pd.DataFrame) -> pd.Series:
        return df.groupby("symbol")["close"].transform(lambda s: s.rolling(10, min_periods=1).mean())

    def leaky_future_return(df: pd.DataFrame) -> pd.Series:
        # Deliberate 1-bar lookahead leak!
        return df.groupby("symbol")["close"].transform(lambda s: s.shift(-1) / s - 1.0)

    assert DataQASuite.verify_feature_causality_shift_test(mini, causal_ma10) is True
    assert DataQASuite.verify_feature_causality_shift_test(mini, leaky_future_return) is False
