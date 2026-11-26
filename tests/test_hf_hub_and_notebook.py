"""Automated tests for Colab/Kaggle Notebook AST validity, QLoRA/DPO compatibility helpers, and Hugging Face Hub uploader."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from vn_rsi.veteran.hf_hub_pusher import (
    build_veteran_model_card,
    normalize_hf_repo_id,
    push_trained_veteran_to_hub,
    resolve_hf_token,
)
from vn_rsi.veteran.preference_builder import (
    _filter_supported_kwargs,
    _format_chat_sft_text,
    _resolve_lora_target_modules,
)
from vn_rsi.veteran.reasoning_engine import (
    MarketContextSnapshot,
    VeteranReasoningEngine,
    strip_think_tags,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeLinear:
    pass


class _FakeModel:
    def __init__(self, leaf_names: list[str]) -> None:
        self._mods = [(f"layers.0.{n}", _FakeLinear()) for n in leaf_names]

    def named_modules(self) -> list[tuple[str, Any]]:
        # Rename class dynamically so `"Linear" in cls_name` holds
        return self._mods


class _FakeHfApi:
    def __init__(self, username: str = "khang_quant") -> None:
        self.username = username
        self.created_repos: list[tuple[str, str, bool]] = []
        self.uploads: list[tuple[str, str, list[str]]] = []

    def whoami(self) -> dict[str, str]:
        return {"name": self.username}

    def create_repo(self, repo_id: str, repo_type: str = "model", private: bool = False, exist_ok: bool = True) -> None:
        _ = exist_ok
        self.created_repos.append((repo_id, repo_type, private))

    def upload_folder(
        self,
        folder_path: str,
        repo_id: str,
        repo_type: str = "model",
        commit_message: str = "",
    ) -> None:
        _ = commit_message
        p = Path(folder_path)
        rel_files = sorted(str(f.relative_to(p)).replace("\\", "/") for f in p.rglob("*") if f.is_file())
        self.uploads.append((repo_id, repo_type, rel_files))


def test_notebook_json_and_all_python_cells_compile() -> None:
    nb_path = REPO_ROOT / "notebooks" / "train_mave_q_kaggle_colab.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 8
    for idx, cell in enumerate(code_cells, start=1):
        src = "".join(cell.get("source", []))
        assert src.strip(), f"Code cell {idx} is empty"
        # Every cell uses pure Python (`subprocess` / `os.chdir`) without fragile shell magics
        ast.parse(src, filename=f"notebook_cell_{idx}.py")


def test_qlora_compatibility_helpers_and_adapter_base_inference(tmp_path: Path) -> None:
    def dummy_fn(a: int, max_length: int = 512) -> int:
        return a + max_length

    filtered = _filter_supported_kwargs(
        dummy_fn, {"a": 1, "max_length": 1024, "unsupported_legacy_arg": True}
    )
    assert filtered == {"a": 1, "max_length": 1024}

    qwen_like = _FakeModel(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"])
    assert _resolve_lora_target_modules(qwen_like) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    phi_like = _FakeModel(["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
    assert _resolve_lora_target_modules(phi_like) == ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

    chat_txt = _format_chat_sft_text(object(), "Phân tích FPT?", "1. Mua vừa phải.")
    assert "Phân tích FPT?" in chat_txt and "1. Mua vừa phải." in chat_txt

    adapter_dir = tmp_path / "final_veteran_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B"}), encoding="utf-8"
    )
    eng = VeteranReasoningEngine(adapter_dir=adapter_dir, base_model_id=None)
    assert eng.base_model_id == "Qwen/Qwen3-1.7B"

    # Verify `<think>` stripping and natural Vietnamese output (private persona, no bragging)
    assert strip_think_tags("<think>internal thoughts</think>\nNhận định FPT tích cực.") == "Nhận định FPT tích cực."
    resp = eng.analyze(MarketContextSnapshot(as_of_date="2024-06-28", symbol="FPT", language="vi"))
    nat_vi = resp.render_natural_vietnamese()
    assert "Nhận định cổ phiếu FPT" in nat_vi
    assert "10 năm kinh nghiệm" not in nat_vi
    assert "<think>" not in nat_vi


def test_hf_hub_pusher_repo_normalization_and_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_dummy_token")
    assert resolve_hf_token() == "hf_test_dummy_token"

    fake_api = _FakeHfApi(username="KenyaWashed")
    assert normalize_hf_repo_id("maveq-v4-veteran", fake_api) == "KenyaWashed/maveq-v4-veteran"
    assert normalize_hf_repo_id("custom_org/maveq-v4", fake_api) == "custom_org/maveq-v4"
    assert (
        normalize_hf_repo_id("https://huggingface.co/KenyaWashed/maveq-v4-qwen3/", fake_api)
        == "KenyaWashed/maveq-v4-qwen3"
    )

    ckpt_dir = tmp_path / "ckpts"
    adapter_dir = ckpt_dir / "veteran_slm_qlora_dpo" / "final_veteran_adapter"
    ds_dir = ckpt_dir / "veteran_datasets"
    adapter_dir.mkdir(parents=True)
    ds_dir.mkdir(parents=True)

    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B"}), encoding="utf-8"
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake_weights")
    (ckpt_dir / "maveq_v4_bundle_kaggle_t4_optimal.json").write_text(
        json.dumps({"preset": "kaggle_t4_optimal", "validation_metrics": {"net_sharpe": 1.358}}),
        encoding="utf-8",
    )
    (ds_dir / "sft_veteran_5step.jsonl").write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
    (ds_dir / "dpo_veteran_vs_fomo.jsonl").write_text('{"prompt":"p","chosen":"c","rejected":"r"}\n', encoding="utf-8")

    res = push_trained_veteran_to_hub(
        checkpoint_dir=ckpt_dir,
        hf_repo_id="maveq-v4-qwen3-veteran",
        private=True,
        hf_dataset_repo="maveq-v4-veteran-data",
        training_summary={
            "preset": "kaggle_t4_optimal",
            "maveq_v4": {"validation_metrics": {"net_sharpe": 1.3585, "deflated_sharpe_ratio": 0.9533}},
            "veteran_qlora_dpo": {"sft_loss": 0.612, "dpo_loss": 0.421, "peak_vram_gb": 3.38},
        },
        api_client=fake_api,
    )

    assert res["status"] == "pushed_to_hf"
    assert res["model_repo_id"] == "KenyaWashed/maveq-v4-qwen3-veteran"
    assert res["model_url"] == "https://huggingface.co/KenyaWashed/maveq-v4-qwen3-veteran"
    assert res["uploaded_slm_adapter"] is True
    assert res["dataset_repo_id"] == "KenyaWashed/maveq-v4-veteran-data"

    assert ("KenyaWashed/maveq-v4-qwen3-veteran", "model", True) in fake_api.created_repos
    assert ("KenyaWashed/maveq-v4-veteran-data", "dataset", True) in fake_api.created_repos

    model_upload_files = fake_api.uploads[0][2]
    assert "README.md" in model_upload_files
    assert "adapter_model.safetensors" in model_upload_files
    assert "maveq_v4_core/maveq_v4_bundle_kaggle_t4_optimal.json" in model_upload_files

    card = build_veteran_model_card(
        repo_id="KenyaWashed/maveq-v4-qwen3-veteran",
        base_model_id="Qwen/Qwen3-1.7B",
        is_merged_16bit=False,
        has_adapter=True,
    )
    assert "library_name: peft" in card and "base_model: Qwen/Qwen3-1.7B" in card
