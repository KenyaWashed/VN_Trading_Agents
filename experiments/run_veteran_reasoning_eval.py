"""Task 06 CLI: Build Search-Zone Anti-Sycophancy SFT/DPO Datasets & Run 45-Scenario Veteran Evaluation + Multi-SLM Audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vn_rsi.data import SnapshotManager
from vn_rsi.models.cloud_presets import MULTI_SLM_REGISTRY, verify_sub1b_blueprint_zero_ram
from vn_rsi.models.dca_backbone import DCAMicroLM
from vn_rsi.utils import RunLogger, compute_config_hash, get_git_commit_hash, load_yaml
from vn_rsi.veteran import (
    VeteranReasoningEngine,
    build_45_anti_sycophancy_scenarios,
    build_veteran_sft_and_dpo_datasets,
    evaluate_veteran_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase12_v5_veteran_antisycophancy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 45-Scenario Anti-Sycophancy Veteran Eval & SFT/DPO Export")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--out-dir", default="artifacts/veteran_datasets")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--run-id", default=RUN_ID)
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
        max_examples=96,
        horizon_days=int(sandbox_cfg.get("prediction_horizon_days", 5)),
    )
    scens = build_45_anti_sycophancy_scenarios()
    bench = evaluate_veteran_benchmark(VeteranReasoningEngine(), scens)

    # Verify Multi-SLM 6-family registry + from-scratch DCA-MicroLM local & cloud sub-1B specs
    microlm_local = DCAMicroLM(preset="local_safe_20m", seed=2026)
    probe_nll = microlm_local.compute_sequence_nll(
        "BÁC BỎ đề xuất tất tay full margin sau 3 phiên trần; đứng ngoài tránh rủi ro kẹt T+2.5."
    )
    sub1b_bp = verify_sub1b_blueprint_zero_ram("sub1b_flagship_900m")
    multi_slm_audit = {
        "n_slm_families": len(MULTI_SLM_REGISTRY),
        "families": {
            k: {
                "repo_id": v.repo_id,
                "family": v.family,
                "params_millions": v.params_millions,
                "architecture": v.architecture,
                "qlora_4bit_vram_gb": v.qlora_vram_gb_4bit,
            }
            for k, v in MULTI_SLM_REGISTRY.items()
        },
        "dca_microlm_local_exact_params": int(microlm_local.exact_params),
        "dca_microlm_cloud_sub1b_params": int(sub1b_bp["blocks_params"]["DCA-MicroLM"]),
        "dca_microlm_probe_nll": round(float(probe_nll), 4),
        "sub1b_all_blocks_verified_under_1b": bool(sub1b_bp["all_blocks_strictly_sub_1b"]),
    }

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
        "multi_slm_and_sub1b_audit": multi_slm_audit,
    }

    out_json = REPO_ROOT / args.logs_dir / "veteran_reasoning_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.log_to_registry:
        logger = RunLogger(
            run_id=args.run_id,
            logs_dir=REPO_ROOT / args.logs_dir,
            git_commit=get_git_commit_hash(REPO_ROOT),
            config_hash=compute_config_hash(
                {"search_end": search_end, "n_scenarios": bench.n_scenarios, "n_traps": bench.n_adversarial_traps}
            )[:16],
            metadata={"phase": 12, "n_scenarios": bench.n_scenarios, "n_adversarial_traps": bench.n_adversarial_traps},
        )
        logger.finalize_summary(summary)

    print(
        json.dumps(
            {
                "status": "OK",
                "n_sft_examples": ds_rep.n_sft_examples,
                "n_dpo_pairs": ds_rep.n_dpo_pairs,
                "n_scenarios": bench.n_scenarios,
                "n_adversarial_traps": bench.n_adversarial_traps,
                "system_scores": bench.system_scores,
                "multi_slm_families": list(MULTI_SLM_REGISTRY.keys()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
