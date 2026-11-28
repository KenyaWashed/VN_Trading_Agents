"""Serverless Modal Cloud Trainer for Large-Scale Holo-MAVE-Q (A100-80GB / H100-80GB / L40S-48GB).

Author: Tran Duy Khang — AI Software Engineer / Freelance Vibe Researcher

Why Modal?
- Modal offers **$30/month FREE recurring serverless compute credits** (pay-per-second, scale-to-zero!).
- Unlike Colab (16GB T4, 90-min idle disconnect) or Kaggle (T4 x2 FP16), Modal gives you native
  **NVIDIA A100-80GB (~$2.50/hr -> ~12 hrs/month FREE)** or **H100-80GB (~$3.95/hr -> ~7.5 hrs/month FREE)**
  with native BF16 & FlashAttention/SDPA.
- A full `cloud_a100_350m` (~357M Tree-MDLM + Holo-HRM) + `Qwen3-8B` 4-bit QLoRA run takes ~4–8 minutes
  (~$0.20–$0.40 out of your $30/month free credits) and automatically saves checkpoints to a Modal Volume!

Quickstart (from your local machine — zero local GPU required):
    pip install modal
    modal setup
    modal run modal_train.py --preset cloud_a100_350m --enable-slm-qlora
    modal volume get mave-q-artifacts /checkpoints ./artifacts/modal_checkpoints
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import modal  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - optional cloud CLI dependency
    raise SystemExit(
        "Modal CLI is not installed locally. Install it with `pip install modal && modal setup`, "
        "then run `modal run modal_train.py --preset cloud_a100_350m`."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent
APP_NAME = "mave-q-holo-trainer"
VOLUME_NAME = "mave-q-artifacts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0.1",
        "pydantic>=2.6.0",
        "httpx>=0.27.0",
        "transformers>=4.46.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "trl>=0.12.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/workspace",
        ignore=[
            ".venv",
            ".git",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "artifacts",
            "*.zip",
        ],
    )
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="A100-80GB",  # Switch to "H100" or "L40S" if desired
    timeout=3600,
    volumes={"/artifacts": vol},
)
def train_remote(
    preset: str = "sub1b_flagship_900m",
    mode: str = "maveq_v5",
    slm_family: str = "gemma3_1b",
    slm_model: str = "",
    enable_slm_qlora: bool = False,
    epochs: int = 12,
    rounds: int = 5,
    seed: int = 2026,
    run_combat_arena: bool = False,
    score_sealed: bool = False,
) -> dict[str, str]:
    """Run MAVE-Q v5 (<1B/module) or legacy training on Modal A100-80GB and persist checkpoints to Modal Volume."""
    cmd = [
        sys.executable,
        "/workspace/train_colab_kaggle.py",
        "--preset",
        preset,
        "--mode",
        mode,
        "--slm-family",
        slm_family,
        "--epochs",
        str(epochs),
        "--rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--checkpoint-dir",
        "/artifacts/checkpoints",
    ]
    if slm_model:
        cmd.extend(["--slm-model", slm_model])
    if enable_slm_qlora:
        cmd.append("--enable-slm-qlora")
    if score_sealed:
        cmd.append("--score-sealed")

    print("[Modal] Executing:", " ".join(cmd))
    subprocess.run(cmd, cwd="/workspace", check=True)

    if run_combat_arena:
        combat_cmd = [
            sys.executable,
            "/workspace/experiments/run_holo_maveq_combat.py",
            "--preset",
            preset,
            "--logs-dir",
            "/artifacts/logs",
        ]
        if not score_sealed:
            combat_cmd.append("--dev")
        print("[Modal] Executing Combat Arena:", " ".join(combat_cmd))
        subprocess.run(combat_cmd, cwd="/workspace", check=True)

    vol.commit()
    return {
        "status": "OK",
        "preset": preset,
        "volume": VOLUME_NAME,
        "checkpoints_path": "/artifacts/checkpoints",
    }


@app.local_entrypoint()
def main(
    preset: str = "sub1b_flagship_900m",
    mode: str = "maveq_v5",
    slm_family: str = "gemma3_1b",
    slm_model: str = "",
    enable_slm_qlora: bool = False,
    epochs: int = 12,
    rounds: int = 5,
    seed: int = 2026,
    run_combat_arena: bool = False,
    score_sealed: bool = False,
) -> None:
    res = train_remote.remote(
        preset=preset,
        mode=mode,
        slm_family=slm_family,
        slm_model=slm_model,
        enable_slm_qlora=enable_slm_qlora,
        epochs=epochs,
        rounds=rounds,
        seed=seed,
        run_combat_arena=run_combat_arena,
        score_sealed=score_sealed,
    )
    print("[Modal] Remote job completed:", res)
