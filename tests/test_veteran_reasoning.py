"""Unit tests for 10-Year Veteran Cognitive Reasoning, SFT/DPO Preference Builder, & 32-Scenario Benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.veteran import (
    MarketContextSnapshot,
    VeteranReasoningEngine,
    build_30_plus_veteran_scenarios,
    build_veteran_sft_and_dpo_datasets,
    evaluate_veteran_benchmark,
    train_veteran_two_stage_qlora_dpo,
)


def test_veteran_5step_reasoning_and_pit_guard() -> None:
    eng = VeteranReasoningEngine()
    # Unannounced PIT trap must be flagged and rejected
    trap_ctx = MarketContextSnapshot(
        as_of_date="2024-01-05",
        symbol="TCB",
        pit_value_quality_z=2.5,
        pit_is_announced=False,
        pit_announcement_lag_days=0,
        foreign_net_ratio_5d=-0.08,
        stock_ret3=0.04,
        language="vi",
    )
    resp = eng.analyze(trap_ctx)
    assert "PIT_UNANNOUNCED_FUNDAMENTAL_LEAKAGE" in resp.causal_violations_flagged
    assert "pit_announcement_lag_trap" in resp.detected_hidden_risks
    assert resp.action in ("REDUCE_OR_AVOID", "CASH_SHIELD")
    md = resp.render_markdown()
    assert "Chẩn đoán Chế độ" in md and "Pre-Mortem" in md


def test_32_scenario_veteran_benchmark_beats_all_baselines() -> None:
    scens = build_30_plus_veteran_scenarios()
    assert len(scens) >= 30
    rep = evaluate_veteran_benchmark(VeteranReasoningEngine(), scens)
    vet = rep.system_scores["MAVE-Q_v4_Veteran"]
    fomo = rep.system_scores["Naive_FOMO_Agent"]
    mom = rep.system_scores["Static_Momentum_Analyst"]
    gen = rep.system_scores["Generic_Rule_Analyst"]

    assert vet["decision_accuracy_pct"] >= 90.0
    assert vet["causal_consistency_pct"] == 100.0
    assert vet["hidden_risk_f1"] > gen["hidden_risk_f1"]
    assert vet["brier_score"] < fomo["brier_score"]
    assert vet["brier_score"] < mom["brier_score"]
    assert vet["ece"] < fomo["ece"]


def test_search_zone_sft_dpo_dataset_export_and_cpu_fallback(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2021-01-04", "2022-06-30", rules, seed=42)
    snap = SnapshotManager(tmp_path / "snaps").write_snapshot("s_vet", bundle)

    ds = build_veteran_sft_and_dpo_datasets(
        snapshot=snap,
        search_end_date="2021-12-31",
        out_dir=tmp_path / "vet_ds",
        max_examples=24,
    )
    assert ds.n_sft_examples >= 6
    assert ds.n_dpo_pairs == ds.n_sft_examples
    # Verify strictly search-zone dates ONLY
    for ln in Path(ds.sft_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        assert rec["as_of_date"] <= "2021-12-31"

    tr = train_veteran_two_stage_qlora_dpo(
        sft_path=ds.sft_path,
        dpo_path=ds.dpo_path,
        output_dir=tmp_path / "vet_ckpt",
        model_id="Qwen/Qwen3-1.7B",
    )
    assert tr["status"] in ("cpu_neuro_symbolic_fallback", "trained_sft_plus_dpo", "qlora_dpo_deps_unavailable")
