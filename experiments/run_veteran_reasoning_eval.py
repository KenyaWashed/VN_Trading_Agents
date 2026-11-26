"""Task 06 CLI: Build Search-Zone Veteran SFT/DPO Datasets & Run 32-Scenario Cognitive Reasoning Benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vn_rsi.data import SnapshotManager
from vn_rsi.utils import RunLogger, compute_config_hash, get_git_commit_hash, load_yaml
from vn_rsi.veteran import (
    VeteranReasoningEngine,
    build_veteran_sft_and_dpo_datasets,
    evaluate_veteran_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase8_veteran_reasoning_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10-Year Veteran Reasoning Benchmark & SFT/DPO Dataset Export")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--out-dir", default="artifacts/veteran_datasets")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--log-to-registry", action="store_true")
    args = parser.parse_args()

    data_cfg = load_yaml(REPO_ROOT / args.data_config)
    sandbox_cfg = load_yaml(REPO_ROOT / args.sandbox_config)
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    search_end = str(sandbox_cfg["zones"]["search"]["end"])

    ds_rep = build_veteran_sft_and_dpo_datasets(
        snapshot=snap,
        search_end_date=search_end,
        out_dir=REPO_ROOT / args.out_dir,
        max_examples=90,
        horizon_days=int(sandbox_cfg.get("prediction_horizon_days", 5)),
    )
    bench = evaluate_veteran_benchmark(VeteranReasoningEngine())

    summary = {
        "dataset_report": {
            "n_sft_examples": ds_rep.n_sft_examples,
            "n_dpo_pairs": ds_rep.n_dpo_pairs,
            "category_counts": ds_rep.category_counts,
            "search_end_verified": ds_rep.search_end_verified,
            "sft_path": ds_rep.sft_path,
            "dpo_path": ds_rep.dpo_path,
        },
        "benchmark": bench.to_dict(),
    }

    out_json = REPO_ROOT / args.logs_dir / "veteran_reasoning_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.log_to_registry:
        logger = RunLogger(
            run_id=RUN_ID,
            logs_dir=REPO_ROOT / args.logs_dir,
            git_commit=get_git_commit_hash(REPO_ROOT),
            config_hash=compute_config_hash({"search_end": search_end, "n_scenarios": bench.n_scenarios})[:16],
            metadata={"phase": 8, "n_scenarios": bench.n_scenarios},
        )
        logger.finalize_summary(summary)

    print(
        json.dumps(
            {
                "status": "OK",
                "n_sft_examples": ds_rep.n_sft_examples,
                "n_dpo_pairs": ds_rep.n_dpo_pairs,
                "n_scenarios": bench.n_scenarios,
                "system_scores": bench.system_scores,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
