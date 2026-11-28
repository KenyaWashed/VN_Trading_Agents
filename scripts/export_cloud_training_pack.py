"""Export Zero-Leakage Cloud Training Pack for MAVE-Q v5 (<1B/block on Kaggle 2x T4 / TPU v5e-8 / Modal A100).

Packages strictly search-zone Anti-Sycophancy SFT/DPO datasets, `configs/maveq_v5.yaml`, the 6-family
Multi-SLM Registry manifest, and the zero-RAM verified `sub1b_flagship_900m` architectural blueprint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vn_rsi.data import SnapshotManager  # noqa: E402
from vn_rsi.models.cloud_presets import (  # noqa: E402
    MULTI_SLM_REGISTRY,
    HardwareSafetyGuard,
    verify_sub1b_blueprint_zero_ram,
)
from vn_rsi.utils import load_yaml  # noqa: E402
from vn_rsi.veteran import build_veteran_sft_and_dpo_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MAVE-Q v5 Free-Cloud Training Pack (Kaggle / Colab / Modal)")
    parser.add_argument("--out-dir", default="artifacts/cloud_training_pack")
    parser.add_argument("--max-examples", type=int, default=96)
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    sandbox_cfg = load_yaml(REPO_ROOT / "configs/sandbox.yaml")
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    search_end = str(sandbox_cfg["zones"]["search"]["end"])

    ds_rep = build_veteran_sft_and_dpo_datasets(
        snapshot=snap,
        search_end_date=search_end,
        out_dir=out_dir / "veteran_datasets",
        max_examples=args.max_examples,
        horizon_days=int(sandbox_cfg.get("prediction_horizon_days", 5)),
    )

    shutil.copy2(REPO_ROOT / "configs/maveq_v5.yaml", out_dir / "maveq_v5.yaml")
    guard = HardwareSafetyGuard()
    local_res = guard.resolve_safe_preset("auto")
    sub1b_bp = verify_sub1b_blueprint_zero_ram("sub1b_flagship_900m")

    manifest = {
        "research_topic": "MAVE-Q v5 (DCA-Chronos-RSI + Popperian Falsification)",
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "search_end_verified": ds_rep.search_end_verified,
        "sealed_test_included": False,
        "local_hardware_safety_resolution": {
            "resolved_preset": local_res.resolved_preset,
            "is_low_ram_local": guard.is_low_ram_local(),
            "reason": local_res.reason,
        },
        "cloud_sub1b_blueprint": sub1b_bp,
        "multi_slm_registry": {
            k: {
                "repo_id": v.repo_id,
                "family": v.family,
                "params_millions": v.params_millions,
                "architecture": v.architecture,
                "qlora_4bit_vram_gb": v.qlora_vram_gb_4bit,
            }
            for k, v in MULTI_SLM_REGISTRY.items()
        },
        "dataset_report": {
            "n_sft_examples": ds_rep.n_sft_examples,
            "n_dpo_pairs": ds_rep.n_dpo_pairs,
            "category_counts": ds_rep.category_counts,
        },
        "launch_commands": {
            "local_safe_cpu": "python train_colab_kaggle.py --mode maveq_v5 --preset auto --slm-family dca_microlm --smoke-test",
            "kaggle_2x_t4_sub1b": "python train_colab_kaggle.py --mode maveq_v5 --preset auto --slm-family gemma3_1b --enable-veteran-dpo --resume",
            "kaggle_tpu_v5e8_sub1b": "python train_colab_kaggle.py --mode maveq_v5 --preset sub1b_flagship_900m --slm-family dca_microlm --resume",
            "modal_a100_sub1b": "modal run modal_train.py --preset sub1b_flagship_900m --mode maveq_v5 --slm-family llama32_1b --enable-slm-qlora",
        },
    }
    manifest_path = out_dir / "cloud_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "OK", "manifest_path": str(manifest_path), "n_dpo_pairs": ds_rep.n_dpo_pairs}, indent=2))


if __name__ == "__main__":
    main()
