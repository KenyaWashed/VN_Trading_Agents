"""Unified Hardware Detection (CUDA / Kaggle TPU v5e-8 / Modal / Colab / CPU) & Model Scale Presets.

Author: Tran Duy Khang — AI Software Engineer / Freelance Vibe Researcher

Supports zero-friction execution across free & low-cost cloud training platforms:
- `cpu_fast`           (~0.2M params): Instant CPU CI / laptop prototyping
- `kaggle_t4_optimal`  (~10M params):  Google Colab Free (1x T4 16GB) alongside Qwen3-4B QLoRA
- `kaggle_max_100m`    (~158M params): Kaggle Free (T4 x2 32GB, 30h/wk) alongside Qwen3-8B/4B QLoRA
- `cloud_a100_350m`    (~357M params): Modal Free ($30/mo credits -> A100-80GB / H100-80GB) or Lightning AI
- `tpu_v5e_350m`       (~357M params): Kaggle Free TPU v5e-8 (8x16GB = 128GB HBM, 20h/wk, BF16 via torch_xla)
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_PRESETS: tuple[str, ...] = (
    "cpu_fast",
    "kaggle_t4_optimal",
    "kaggle_max_100m",
    "cloud_a100_350m",
    "tpu_v5e_350m",
)


@dataclass(frozen=True)
class PresetArchitecture:
    """Calibrated dimensions for Tree-MDLM / AST-Diffusion and HRM / Holo-HRM."""

    name: str
    diff_d_model: int
    diff_n_layers: int
    diff_n_heads: int
    diff_d_ff: int
    diff_denoise_steps: int
    diff_lr: float
    hrm_d_hidden: int
    hrm_n_layers: int
    hrm_lr: float
    holo_hrm_np_dim: int
    use_amp: bool
    recommended_platform: str


PRESET_SPECS: dict[str, PresetArchitecture] = {
    "cpu_fast": PresetArchitecture(
        name="cpu_fast",
        diff_d_model=64,
        diff_n_layers=2,
        diff_n_heads=4,
        diff_d_ff=256,
        diff_denoise_steps=6,
        diff_lr=1e-3,
        hrm_d_hidden=64,
        hrm_n_layers=1,
        hrm_lr=1e-3,
        holo_hrm_np_dim=32,
        use_amp=False,
        recommended_platform="Local CPU (0 GPU)",
    ),
    "kaggle_t4_optimal": PresetArchitecture(
        name="kaggle_t4_optimal",
        diff_d_model=256,
        diff_n_layers=8,
        diff_n_heads=8,
        diff_d_ff=1024,
        diff_denoise_steps=8,
        diff_lr=3e-4,
        hrm_d_hidden=256,
        hrm_n_layers=4,
        hrm_lr=3e-4,
        holo_hrm_np_dim=96,
        use_amp=True,
        recommended_platform="Google Colab Free (1x T4 16GB)",
    ),
    "kaggle_max_100m": PresetArchitecture(
        name="kaggle_max_100m",
        diff_d_model=768,
        diff_n_layers=14,
        diff_n_heads=12,
        diff_d_ff=3072,
        diff_denoise_steps=10,
        diff_lr=1.5e-4,
        hrm_d_hidden=768,
        hrm_n_layers=10,
        hrm_lr=1.5e-4,
        holo_hrm_np_dim=128,
        use_amp=True,
        recommended_platform="Kaggle Free (T4 x2 32GB, 30h/wk)",
    ),
    "cloud_a100_350m": PresetArchitecture(
        name="cloud_a100_350m",
        diff_d_model=1024,
        diff_n_layers=20,
        diff_n_heads=16,
        diff_d_ff=4096,
        diff_denoise_steps=12,
        diff_lr=1.0e-4,
        hrm_d_hidden=1024,
        hrm_n_layers=10,
        hrm_lr=1.0e-4,
        holo_hrm_np_dim=160,
        use_amp=True,
        recommended_platform="Modal Free ($30/mo A100-80GB / H100) or Lightning AI",
    ),
    "tpu_v5e_350m": PresetArchitecture(
        name="tpu_v5e_350m",
        diff_d_model=1024,
        diff_n_layers=20,
        diff_n_heads=16,
        diff_d_ff=4096,
        diff_denoise_steps=12,
        diff_lr=1.0e-4,
        hrm_d_hidden=1024,
        hrm_n_layers=10,
        hrm_lr=1.0e-4,
        holo_hrm_np_dim=160,
        use_amp=True,
        recommended_platform="Kaggle Free TPU v5e-8 (128GB HBM, 20h/wk, BF16)",
    ),
}


def get_preset_spec(preset: str) -> PresetArchitecture:
    return PRESET_SPECS.get(preset, PRESET_SPECS["cpu_fast"])


def detect_torch_xla() -> Any | None:
    """Return `torch_xla.core.xla_model` if running on a Kaggle/GCP TPU, else None."""
    if not (os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR") or os.environ.get("PJRT_DEVICE") == "TPU"):
        return None
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

        return xm
    except Exception:
        return None


def resolve_torch_device(device: str | None = None) -> tuple[Any, bool]:
    """Resolve PyTorch device across CUDA, TPU (torch_xla), MPS, and CPU.

    Returns `(torch_device, is_xla_tpu)`.
    """
    import torch

    if device == "xla" or device == "tpu":
        xm = detect_torch_xla()
        if xm is not None:
            return xm.xla_device(), True
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

    if device:
        return torch.device(device), False

    xm = detect_torch_xla()
    if xm is not None:
        return xm.xla_device(), True
    if torch.cuda.is_available():
        return torch.device("cuda"), False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


def xla_mark_step_if_needed(is_xla: bool) -> None:
    """Flush XLA lazy graph execution on TPU without touching CUDA/CPU paths."""
    if not is_xla:
        return
    xm = detect_torch_xla()
    if xm is not None:
        xm.mark_step()


def amp_autocast_and_scaler(device: Any, enabled: bool, is_xla: bool = False) -> tuple[Any, Any | None]:
    """Return (autocast_context_manager, optional_grad_scaler) calibrated for T4 FP16 vs A100/H100/TPU BF16."""
    if not enabled:
        return nullcontext(), None
    import torch

    if is_xla:
        return torch.amp.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True), None

    dev_type = getattr(device, "type", str(device))
    if dev_type != "cuda":
        return nullcontext(), None

    use_bf16 = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
    if use_bf16:
        # BF16 does not require gradient scaling
        return ctx, None
    scaler = torch.amp.GradScaler("cuda", enabled=True) if hasattr(torch.amp, "GradScaler") else None
    return ctx, scaler


def detect_cloud_compute() -> dict[str, Any]:
    """Detect cloud platform (Kaggle GPU/TPU, Colab, Modal, Lightning AI, Local) & recommend optimal preset."""
    platform = "local"
    if os.environ.get("MODAL_IMAGE_ID") or os.environ.get("MODAL_TASK_ID"):
        platform = "modal"
    elif os.environ.get("LIGHTNING_CLOUD_SPACE_ID") or os.environ.get("LIGHTNING_CLOUDSPACE_HOST"):
        platform = "lightning_ai"
    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        platform = "kaggle"
    elif os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        platform = "colab"

    gpus: list[dict[str, Any]] = []
    has_torch = False
    has_tpu = detect_torch_xla() is not None
    bf16_supported = has_tpu

    try:
        import torch

        has_torch = True
        if torch.cuda.is_available():
            bf16_supported = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "vram_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
    except Exception:
        pass

    max_vram = max((g["vram_gb"] for g in gpus), default=0.0)
    if has_tpu:
        slm_device, micro_device = "xla", "xla"
        rec_preset = "tpu_v5e_350m"
    elif max_vram >= 38.0:
        # A100-40/80GB, H100-80GB, L40S-48GB (Modal / Colab Pro / Lightning AI)
        slm_device = "cuda:0"
        micro_device = "cuda:1" if len(gpus) >= 2 else "cuda:0"
        rec_preset = "cloud_a100_350m"
    elif len(gpus) >= 2:
        # Kaggle T4 x2 (32GB total across 2 GPUs)
        slm_device, micro_device = "cuda:0", "cuda:1"
        rec_preset = "kaggle_max_100m"
    elif len(gpus) == 1:
        # Colab Free T4 (16GB) or L4 (24GB)
        slm_device, micro_device = "cuda:0", "cuda:0"
        rec_preset = "kaggle_max_100m" if max_vram >= 20.0 else "kaggle_t4_optimal"
    else:
        slm_device, micro_device = "cpu", "cpu"
        rec_preset = "cpu_fast"

    return {
        "platform": platform,
        "has_torch": has_torch,
        "has_tpu": has_tpu,
        "bf16_supported": bf16_supported,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "max_gpu_vram_gb": max_vram,
        "slm_device": slm_device,
        "micro_device": micro_device,
        "recommended_preset": rec_preset,
    }
