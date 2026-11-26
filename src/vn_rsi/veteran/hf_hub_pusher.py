"""Hugging Face Hub Uploader for MAVE-Q v4 (VetQuant-RSI) Open-Source SLM Adapters, Merged Weights & Datasets.

Supports zero-hardcoded-credential authentication across:
1. Environment variables (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`)
2. Google Colab Secrets (`google.colab.userdata.get('HF_TOKEN')`)
3. Kaggle Secrets (`kaggle_secrets.UserSecretsClient().get_secret('HF_TOKEN')`)
4. Local Hugging Face CLI login cache (`huggingface_hub.get_token()`)

Automatically creates the target Hugging Face repository if missing, resolves `<username>/<repo_name>`
when only `<repo_name>` (or a full `https://huggingface.co/...` URL) is supplied, generates a complete
Hugging Face Model Card (`README.md`), and uploads:
- Fine-tuned PEFT LoRA+DPO adapter + tokenizer (or full merged FP16 standalone weights when `merge_16bit=True`)
- MAVE-Q v4 specialist checkpoints (`Tree-MDLM`, `Holo-HRM`, `VeteranCaseMemory`, `maveq_v4_bundle_*.json`)
- Optional 5-step SFT + DPO dataset repository (`sft_veteran_5step.jsonl`, `dpo_veteran_vs_fomo.jsonl`)
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    """Safely resolve a Hugging Face write token without ever hardcoding secrets."""
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()

    # Google Colab Secrets (`🔑` sidebar)
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        colab_tok = userdata.get("HF_TOKEN")
        if isinstance(colab_tok, str) and colab_tok.strip():
            return colab_tok.strip()
    except Exception:
        pass

    # Kaggle Notebook Secrets (`Add-ons -> Secrets`)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        kag_tok = UserSecretsClient().get_secret("HF_TOKEN")
        if isinstance(kag_tok, str) and kag_tok.strip():
            return kag_tok.strip()
    except Exception:
        pass

    # Cached `huggingface-cli login` / `notebook_login()` token
    try:
        from huggingface_hub import get_token  # type: ignore[import-not-found]

        cached = get_token()
        if isinstance(cached, str) and cached.strip():
            return cached.strip()
    except Exception:
        pass

    return None


def normalize_hf_repo_id(repo_id: str, api: Any) -> str:
    """Normalize `<repo>`, `<user>/<repo>`, or `https://huggingface.co/<user>/<repo>` into `<user>/<repo>`."""
    cleaned = repo_id.strip().rstrip("/")
    for prefix in (
        "https://huggingface.co/datasets/",
        "http://huggingface.co/datasets/",
        "https://huggingface.co/",
        "http://huggingface.co/",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip("/")
            break

    if not cleaned:
        raise ValueError("`hf_repo_id` cannot be empty (e.g. 'your-username/maveq-v4-qwen3-veteran').")

    if "/" in cleaned:
        parts = [p for p in cleaned.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"

    who = api.whoami()
    username = str(who.get("name", "")).strip()
    if not username:
        raise ValueError(
            f"Could not infer Hugging Face username for short repo '{cleaned}'. "
            "Please pass full 'username/repo-name'."
        )
    return f"{username}/{cleaned}"


def build_veteran_model_card(
    repo_id: str,
    base_model_id: str,
    is_merged_16bit: bool,
    has_adapter: bool,
    training_summary: dict[str, Any] | None = None,
) -> str:
    """Generate a Hugging Face Hub `README.md` Model Card with valid YAML metadata and usage snippets."""
    summary = training_summary or {}
    v4_info = summary.get("maveq_v4", {}) if isinstance(summary.get("maveq_v4"), dict) else {}
    val_m = v4_info.get("validation_metrics", {}) if isinstance(v4_info.get("validation_metrics"), dict) else {}
    dpo_info = summary.get("veteran_qlora_dpo", {}) if isinstance(summary.get("veteran_qlora_dpo"), dict) else {}

    lib_name = "transformers" if is_merged_16bit else ("peft" if has_adapter else "pytorch")
    sft_loss = dpo_info.get("sft_loss", "N/A")
    dpo_loss = dpo_info.get("dpo_loss", "N/A")
    peak_vram = dpo_info.get("peak_vram_gb", "N/A")
    val_sharpe = val_m.get("net_sharpe", "N/A")
    val_dsr = val_m.get("deflated_sharpe_prob", val_m.get("deflated_sharpe_ratio", "N/A"))
    preset = summary.get("preset", v4_info.get("preset", "kaggle_t4_optimal"))

    if is_merged_16bit:
        usage_code = f'''```python
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{repo_id}"
tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

system_prompt = (
    "Bạn là chuyên viên phân tích và quản lý danh mục cổ phiếu Việt Nam dày dặn thực chiến. "
    "Trả lời hoàn toàn bằng tiếng Việt tự nhiên, súc tích, không in phần suy nghĩ nội bộ (<think>) "
    "và không tự nhắc đến số năm kinh nghiệm."
)
prompt = (
    "Dữ liệu thị trường ngày 2024-06-28 cho mã FPT (sàn HOSE): VN-Index 20 phiên +3.4%, "
    "độ rộng 62%, biến động 0.94x; cổ phiếu 3 phiên +1.8%, 20 phiên +8.5%, "
    "thanh khoản 1.25x, khối ngoại 5 phiên +4.2%, BCTC đã công bố, cách sàn 6.1%, cách trần 4.8%. "
    "Hãy đánh giá cơ hội, rủi ro T+2.5 và đề xuất hành động cụ thể cho mã FPT."
)
messages = [{{"role": "system", "content": system_prompt}}, {{"role": "user", "content": prompt}}]
text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text_in, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.08)
ans = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(re.sub(r"<think>.*?</think>\\s*", "", ans, flags=re.DOTALL).strip())
```'''
    elif has_adapter:
        usage_code = f'''```python
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "{base_model_id}"
adapter_repo_id = "{repo_id}"

tok = AutoTokenizer.from_pretrained(adapter_repo_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_repo_id)
model.eval()

system_prompt = (
    "Bạn là chuyên viên phân tích và quản lý danh mục cổ phiếu Việt Nam dày dặn thực chiến. "
    "Trả lời hoàn toàn bằng tiếng Việt tự nhiên, súc tích, không in phần suy nghĩ nội bộ (<think>) "
    "và không tự nhắc đến số năm kinh nghiệm."
)
prompt = (
    "Dữ liệu thị trường ngày 2024-06-28 cho mã FPT (sàn HOSE): VN-Index 20 phiên +3.4%, "
    "độ rộng 62%, biến động 0.94x; cổ phiếu 3 phiên +1.8%, 20 phiên +8.5%, "
    "thanh khoản 1.25x, khối ngoại 5 phiên +4.2%, BCTC đã công bố, cách sàn 6.1%, cách trần 4.8%. "
    "Hãy đánh giá cơ hội, rủi ro T+2.5 và đề xuất hành động cụ thể cho mã FPT."
)
messages = [{{"role": "system", "content": system_prompt}}, {{"role": "user", "content": prompt}}]
text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text_in, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.08)
ans = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(re.sub(r"<think>.*?</think>\\s*", "", ans, flags=re.DOTALL).strip())
```'''
    else:
        usage_code = f'''```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="{repo_id}")
print("Downloaded MAVE-Q v4 specialist bundle to:", local_dir)
```'''

    return f"""---
license: mit
library_name: {lib_name}
base_model: {base_model_id}
language:
  - vi
  - en
tags:
  - finance
  - quantitative-trading
  - vietnam-equities
  - qlora
  - dpo
  - mave-q
  - agent
---

# {repo_id.split('/')[-1]} — MAVE-Q v4 (`VetQuant-RSI`) 10-Year Veteran CIO Model

Fine-tuned open-source **10-Year Veteran Quantitative CIO** model + **MAVE-Q v4 (`VetQuant-RSI`)** neuro-symbolic specialist bundle for rule-constrained emerging equity markets (T+2.5 settlement locks, ±7%/10%/15% daily price limits, and strict point-in-time fundamental lag discipline).

> **Disclaimer**: Strictly for academic quantitative research and simulation ONLY. Not financial advice; zero live brokerage execution.

## Training & Validation Telemetry

| Metric / Setting | Value |
| :--- | :--- |
| **Base Open-Source SLM** | `{base_model_id}` |
| **Artifact Format** | `{"Full Merged FP16 Standalone Model" if is_merged_16bit else ("PEFT 4-bit QLoRA SFT + DPO Adapter" if has_adapter else "MAVE-Q v4 Neuro-Symbolic Bundle")}` |
| **Structural Preset** | `{preset}` |
| **Stage-1 SFT Loss (5-Step Veteran CIO)** | `{sft_loss}` |
| **Stage-2 DPO Loss (Veteran vs. FOMO)** | `{dpo_loss}` |
| **Peak GPU VRAM** | `{peak_vram} GB` |
| **MAVE-Q v4 Validation Net Sharpe** | `{val_sharpe}` |
| **MAVE-Q v4 Validation DSR** | `{val_dsr}` |

## Quickstart Inference

{usage_code}

## Repository Contents

- Root directory: Open-source LLM/SLM {"merged FP16 weights & tokenizer" if is_merged_16bit else "PEFT LoRA/DPO adapter & tokenizer"} (when `--enable-veteran-dpo` was run on GPU).
- `maveq_v4_core/`: MAVE-Q v4 specialist checkpoints (`Tree-MDLM`, `Holo-HRM`, `VeteranCaseMemory`, champion factor bundle JSON, and telemetry report).
"""


def _merge_adapter_to_16bit(base_model_id: str, adapter_dir: Path, merged_out_dir: Path) -> None:
    """Merge PEFT LoRA adapter into base model in FP16 on CPU/RAM (avoids T4 16GB VRAM OOM)."""
    import torch
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    merged_out_dir.mkdir(parents=True, exist_ok=True)
    tok_src = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else base_model_id
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), device_map="cpu")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(merged_out_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_out_dir))
    del merged, peft_model, base
    gc.collect()


def push_trained_veteran_to_hub(
    checkpoint_dir: str | Path,
    hf_repo_id: str,
    token: str | None = None,
    private: bool = False,
    merge_16bit: bool = False,
    base_model_id: str | None = None,
    hf_dataset_repo: str | None = None,
    training_summary: dict[str, Any] | None = None,
    api_client: Any | None = None,
) -> dict[str, Any]:
    """Upload trained Open-Source Veteran SLM (LoRA/DPO adapter or merged FP16) + MAVE-Q v4 core to Hugging Face Hub."""
    resolved_token = resolve_hf_token(token)
    if not resolved_token and api_client is None:
        raise ValueError(
            "Không tìm thấy Hugging Face Write Token (`HF_TOKEN`)! "
            "Hãy thêm Secret `HF_TOKEN` trên Colab (biểu tượng 🔑 bên trái) / Kaggle (Add-ons -> Secrets), "
            "hoặc đặt `os.environ['HF_TOKEN'] = 'hf_...'` trước khi push."
        )

    if api_client is None:
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài `huggingface_hub`. Hãy chạy `pip install -U huggingface_hub`."
            ) from exc
        api = HfApi(token=resolved_token)
    else:
        api = api_client

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    # Load saved training report if not passed directly
    if training_summary is None and (ckpt_path / "training_report.json").exists():
        try:
            training_summary = json.loads((ckpt_path / "training_report.json").read_text(encoding="utf-8"))
        except Exception:
            training_summary = {}

    adapter_dir = ckpt_path / "veteran_slm_qlora_dpo" / "final_veteran_adapter"
    has_adapter = adapter_dir.exists() and any(adapter_dir.iterdir())

    inferred_base = base_model_id
    if not inferred_base and has_adapter:
        for cfg_name in ("adapter_config.json", "veteran_training_meta.json"):
            p = adapter_dir / cfg_name
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    inferred_base = d.get("base_model_name_or_path") or d.get("base_model_id")
                    if inferred_base:
                        break
                except Exception:
                    pass
    if not inferred_base and isinstance(training_summary, dict):
        dpo_d = training_summary.get("veteran_qlora_dpo", {})
        if isinstance(dpo_d, dict):
            inferred_base = dpo_d.get("model_id")
    eff_base_model = str(inferred_base or "Qwen/Qwen3-1.7B")

    full_model_repo = normalize_hf_repo_id(hf_repo_id, api)
    api.create_repo(repo_id=full_model_repo, repo_type="model", private=private, exist_ok=True)

    staging_dir = ckpt_path / "_hf_upload_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    did_merge = False
    if has_adapter and merge_16bit:
        _merge_adapter_to_16bit(eff_base_model, adapter_dir, staging_dir)
        did_merge = True
    elif has_adapter:
        for item in adapter_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, staging_dir / item.name)

    # Bundle MAVE-Q v4 specialist weights & metadata under `maveq_v4_core/`
    core_stage = staging_dir / "maveq_v4_core"
    core_stage.mkdir(parents=True, exist_ok=True)
    uploaded_core_files: list[str] = []
    core_patterns = (
        "maveq_v4_bundle_*.json",
        "v4_tree_mdlm_*",
        "v4_holo_hrm_*",
        "v4_case_memory_*",
        "holo_tree_mdlm_*",
        "holo_hrm_*",
        "training_report.json",
    )
    for pat in core_patterns:
        for fpath in sorted(ckpt_path.glob(pat)):
            if fpath.is_file():
                shutil.copy2(fpath, core_stage / fpath.name)
                uploaded_core_files.append(fpath.name)

    if training_summary and not (core_stage / "training_report.json").exists():
        (core_stage / "training_report.json").write_text(
            json.dumps(training_summary, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )
        uploaded_core_files.append("training_report.json")

    card_md = build_veteran_model_card(
        repo_id=full_model_repo,
        base_model_id=eff_base_model,
        is_merged_16bit=did_merge,
        has_adapter=has_adapter,
        training_summary=training_summary,
    )
    (staging_dir / "README.md").write_text(card_md, encoding="utf-8")

    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=full_model_repo,
        repo_type="model",
        commit_message=(
            f"Upload MAVE-Q v4 Veteran ({'merged FP16' if did_merge else ('QLoRA+DPO adapter' if has_adapter else 'specialist core')}) "
            f"base={eff_base_model}"
        ),
    )
    shutil.rmtree(staging_dir, ignore_errors=True)

    result: dict[str, Any] = {
        "status": "pushed_to_hf",
        "model_repo_id": full_model_repo,
        "model_url": f"https://huggingface.co/{full_model_repo}",
        "base_model_id": eff_base_model,
        "private": private,
        "uploaded_slm_adapter": has_adapter and not did_merge,
        "uploaded_merged_16bit": did_merge,
        "uploaded_v4_core_files": uploaded_core_files,
    }

    # Optional: Push verified search-zone 5-step SFT + DPO datasets to Hugging Face Datasets
    if hf_dataset_repo and hf_dataset_repo.strip():
        ds_dir = ckpt_path / "veteran_datasets"
        if ds_dir.exists() and any(ds_dir.glob("*.jsonl")):
            full_ds_repo = normalize_hf_repo_id(hf_dataset_repo, api)
            api.create_repo(repo_id=full_ds_repo, repo_type="dataset", private=private, exist_ok=True)
            ds_stage = ckpt_path / "_hf_ds_staging"
            if ds_stage.exists():
                shutil.rmtree(ds_stage)
            ds_stage.mkdir(parents=True, exist_ok=True)
            ds_files: list[str] = []
            for jf in sorted(ds_dir.glob("*.jsonl")):
                shutil.copy2(jf, ds_stage / jf.name)
                ds_files.append(jf.name)
            ds_readme = (
                "---\nlicense: mit\nlanguage:\n  - vi\n  - en\ntags:\n  - finance\n  - dpo\n  - sft\n  - vietnam-equities\n---\n\n"
                f"# {full_ds_repo.split('/')[-1]} — MAVE-Q v4 Search-Zone 5-Step SFT & DPO Dataset\n\n"
                "Strictly causal search-zone (`<= 2022-12-30`, zero validation/test leakage) 10-Year Veteran CIO "
                "5-step reasoning SFT trajectories (`sft_veteran_5step.jsonl`) and Veteran-vs-FOMO preference pairs "
                "(`dpo_veteran_vs_fomo.jsonl`).\n"
            )
            (ds_stage / "README.md").write_text(ds_readme, encoding="utf-8")
            api.upload_folder(
                folder_path=str(ds_stage),
                repo_id=full_ds_repo,
                repo_type="dataset",
                commit_message="Upload verified search-zone 5-step Veteran SFT & DPO preference datasets",
            )
            shutil.rmtree(ds_stage, ignore_errors=True)
            result["dataset_repo_id"] = full_ds_repo
            result["dataset_url"] = f"https://huggingface.co/datasets/{full_ds_repo}"
            result["uploaded_dataset_files"] = ds_files

    return result
