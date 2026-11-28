# Free-Cloud Compute & Sub-1B Scaling Proposals — MAVE-Q v5 Research Topic
# Đề Xuất Mở Rộng Quy Mô Sub-1B (<1B/Khối) Trên Hạ Tầng Cloud Miễn Phí — Đề Tài Nghiên Cứu MAVE-Q v5

> **Nhãn tài liệu / Document Label**: **Research Topic Technical Proposal / Đề xuất kỹ thuật thuộc Đề tài nghiên cứu MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification)** — *không phải bài báo đã nộp hay công bố (not a submitted/published paper)*.

---

## 1. Giới Hạn Thực Nghiệm Local vs. Cầu Chì Phần Cứng Kép (`HardwareSafetyGuard`)

Trên máy tính nghiên cứu nội bộ (**AMD Ryzen 5 5500U 6C/12T, 7.35 GB RAM, 0 GPU CUDA**), việc cố nạp mô hình lớn gây rủi ro tràn bộ nhớ (OOM) hoặc treo hệ thống. Do đó, `HardwareSafetyGuard` (`src/vn_rsi/models/cloud_presets.py`) khóa cứng cơ chế hai tầng thông qua công tắc `--preset auto`:

| Môi trường / Tier | RAM / VRAM khả dụng | Preset tự động (`--preset auto`) | Quy mô thực thi | Ghi chú an toàn |
| :--- | :--- | :--- | :--- | :--- |
| **Local Laptop (CPU)** | `7.35 GB RAM`, `0 GPU` | `local_safe_20m` | `0.48M–20.47M` (`float32`, peak `< 1.2 GB RAM`, thực tế `~284 MB`) | Chặn cứng mọi cấp phát >250 MB/tensor; kiểm chứng kiến trúc `<1B` bằng `device="meta"` / công thức giải tích (`0 MB RAM`). |
| **Google Colab FREE** | `1x T4 16 GB VRAM` | `kaggle_t4_optimal` / `local_safe_20m` | `10M–22M` + 1 SLM 4-bit QLoRA (`< 4.5 GB VRAM`) | Dọn VRAM tự động giữa các stage (`gc.collect()` + `torch.cuda.empty_cache()`). |
| **Kaggle Notebooks FREE (GPU)** | `2x T4 32 GB VRAM` (30h/tuần) | `sub1b_flagship_900m` | 4 khối chuyên biệt **đều `< 1B` tham số/khối** (`FP16`/4-bit NF4, phân tải `cuda:0` & `cuda:1`) | Tự động bung đầy đủ 4 khối mô hình thật + hỗ trợ `--resume` qua giới hạn phiên 12h. |
| **Kaggle Notebooks FREE (TPU)** | `TPU v5e-8 128 GB HBM` (20h/tuần) | `sub1b_flagship_900m` | 4 khối `< 1B` native `BF16` qua `torch_xla` | Lý tưởng cho pretraining đa thị trường (`VN100 + US + CSI + ASEAN`). |
| **Modal Serverless FREE** | `$30/tháng` (`A100-80GB` / `H100`) | `sub1b_flagship_900m` | 4 khối `< 1B` native `BF16` + FlashAttention/SDPA (`~5-10 phút / ~$0.35`) | Lưu checkpoint bền vững vào Modal Volume `mave-q-artifacts`. |

---

## 2. Kiến Trúc `sub1b_flagship_900m` (4 Khối Chuyên Biệt Đều `< 1B` Tham Số/Khối)

Tất cả 4 khối dưới đây đều đã được kiểm chứng giải tích chính xác `0 MB RAM` bởi `verify_sub1b_blueprint_zero_ram("sub1b_flagship_900m")`:

1. **`DCA-QuantFormer` Backbone (`906M < 1B`)**: 24 tầng Grouped Depth-Wise Cross-Attention (`G-DCA`, Heddes et al., ICML 2025, `arXiv:2502.06785`) + `muP` (`d_model=1536, G=16, d_ff=6144`), bảo toàn vi cấu trúc tầng nông ở độ sâu lớn.
2. **`DCA-AST-Diffusion` (`849M < 1B`)**: Bộ sinh cây cú pháp công thức alpha (`Tree-MDLM`) dẫn hướng bằng `diffu-GRPO` với `TypedASTGuard`.
3. **`DCA-Holo-HRM` (`802M < 1B`)**: Bộ đánh giá tiềm năng nhân tố phân cấp (Hierarchical Reasoning Model) chống quá khớp.
4. **`DCA-MicroLM` (`828M < 1B` trên Cloud; `18.39M` trên Local)**: Mô hình ngôn ngữ nhân quả tự cài đặt 100% từ đầu tích hợp `G-DCA` + `muP` phục vụ phản biện định lượng Anti-Sycophancy.

---

## 3. Đấu Trường 6 Họ SLM (`MULTI_SLM_REGISTRY` — Không Bám Cứng Vào Qwen)

| Key CLI (`--slm-family`) | Model ID / Origin | Tham số | Kiến trúc | VRAM 4-bit QLoRA | Phù hợp cho |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `gemma3_1b` | `google/gemma-3-1b-it` | `1.0B` | Gemma-3 GQA | `~2.4 GB` | Kaggle `T4`, Colab `T4`, đa ngôn ngữ |
| `llama32_1b` | `meta-llama/Llama-3.2-1B-Instruct` | `1.23B` | Llama-3.2 GQA | `~2.8 GB` | Kaggle `T4`, Modal `A100` |
| `smollm2_360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | `0.36B` | SmolLM2 Ultra-Light | `~1.3 GB` | Colab `T4` cực nhanh / CPU thử nghiệm |
| `deepseek_r1_1p5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `1.5B` | Distilled Reasoning | `~3.2 GB` | Suy luận phản biện chuỗi bước |
| `qwen3_0p6b` | `Qwen/Qwen3-0.6B` | `0.60B` | Qwen3 Hybrid (`no-think`) | `~1.8 GB` | Tiếng Việt mượt, siêu nhẹ |
| `dca_microlm` | `custom://dca-microlm` (100% from-scratch) | `18.4M / 828M` | Custom `G-DCA` + `muP` | `~0.1 GB / ~2.6 GB` | Tự chủ hoàn toàn 100% không phụ thuộc checkpoint ngoài |

---

## 4. Bài Học Phản Biện Popperian Về Quy Mô Tham Số (Honest Scaling Law Finding)

Thực nghiệm trên tập nhỏ **VN30 (`30 cổ phiếu × 4 năm search`)** ở Task 04 cho thấy một kết quả phản biện trung thực quan trọng:
- Tăng tham số từ **`0.48M -> 5.12M -> 11.52M -> 20.47M`** giúp giảm độ lệch chuẩn giữa các seed (`SD: 0.107 -> 0.028`) nhưng **bị bác bỏ bởi tiêu chuẩn Occam's Razor BIC** (`BIC: 1.994 -> 1.807 -> 1.692 -> 1.635`) do tỷ lệ mẫu/tham số nhỏ trên VN30.
- **Đề xuất khoa học cho giai đoạn Cloud (`sub1b_flagship_900m`)**: Chỉ kích hoạt đầy đủ quy mô `600M–900M < 1B/khối` khi huấn luyện đa thị trường chéo (pretraining chung trên `VN100 + HNX + UPCoM + CSI300 + S&P500 + intraday orderbook/news`), sau đó dùng **LoRA/distillation tinh gọn** hoặc **zero-shot transfer** xuống từng thị trường hẹp như VN30.

---

## 5. Lệnh Khởi Chạy Nhanh (One-Click Reproducibility)

```bash
# 1. Xuất gói huấn luyện zero-leakage (strictly search-zone SFT/DPO + manifest)
python scripts/export_cloud_training_pack.py

# 2. Chạy an toàn trên Laptop Local (<1.2 GB RAM)
python train_colab_kaggle.py --mode maveq_v5 --preset auto --slm-family dca_microlm --smoke-test

# 3. Kaggle 2x T4 32GB (tự động nâng cấp lên sub1b_flagship_900m + Gemma-3-1B Anti-Sycophancy DPO)
python train_colab_kaggle.py --mode maveq_v5 --preset auto --slm-family gemma3_1b --enable-veteran-dpo --resume

# 4. Kaggle TPU v5e-8 128GB (from-scratch DCA-MicroLM 828M + DCA-QuantFormer 906M)
python train_colab_kaggle.py --mode maveq_v5 --preset sub1b_flagship_900m --slm-family dca_microlm --resume

# 5. Modal Serverless A100-80GB ($30/mo free credits)
modal run modal_train.py --preset sub1b_flagship_900m --mode maveq_v5 --slm-family llama32_1b --enable-slm-qlora
```
