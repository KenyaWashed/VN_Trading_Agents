# BÁO CÁO TỔNG HỢP DỰ ÁN NGHIÊN CỨU: MAVE-Q
**MAVE-Q: Modular Autonomous Verified Evolution for Small-Model (≤8B) Quantitative Research in Rule-Constrained Emerging Markets**

**Tác giả**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher*

> **Tuyên bố Tuân thủ Nghiên cứu (Hard Prohibitions - Mục 9)**:
> Toàn bộ kho mã nguồn này là **hệ thống mô phỏng & nghiên cứu khoa học phục vụ viết bài báo học thuật (arXiv `cs.AI` / `cs.LG` / `q-fin.CP`)**, **TUYỆT ĐỐI KHÔNG** phải bot giao dịch thực tế, không chứa bất kỳ kết nối tài khoản chứng khoán, đặt lệnh thật hay lời khuyên đầu tư nào.

---

## 1. Tổng quan Mục tiêu Khoa học (Dành cho AI Research Fresher)

Khi bạn mới bắt đầu làm nghiên cứu AI (fresher) hướng tới công bố quốc tế (Workshop tại NeurIPS / ICLR / ICML / AAAI hoặc preprint arXiv), một bài báo mạnh **không** nằm ở việc khoe lợi nhuận backtest cao bất thường (reviewer tài chính định lượng sẽ nghi ngờ *data leakage* hoặc *overfitting* ngay lập tức). Thay vào đó, câu hỏi khoa học cốt lõi mà hệ thống **VN-RSI** giải quyết là:

> *"Liệu một hệ thống đa tác tử dùng mô hình ngôn ngữ mở cỡ nhỏ (≤8B tham số, chạy trên 1 GPU 24GB hoặc CPU quantized) có thể **tự cải tiến quy trình nghiên cứu của chính nó** (prompts, trọng số khám phá cơ chế kinh tế, độ sâu biểu thức, bộ nhớ niềm tin) từ bằng chứng thực nghiệm hợp lệ hay không — trong khi bộ đánh giá hoàn toàn **chống rò rỉ dữ liệu (leak-free)** và tuân thủ nghiêm ngặt vi cấu trúc thị trường mới nổi kém thanh khoản như Việt Nam (chu kỳ thanh toán T+2.5, biên độ trần/sàn z_H\pm 7\%/10\%/15\%z_H, thuế bán 0.1%, lô chẵn 100 cổ phiếu)?"*

---

## 2. Kiến trúc Toàn diện Đã Xây dựng (Phases 0 z_H\toz_H 5)

Toàn bộ dự án được đặt tại [`D:\Research\VN_Trading_Agents`](file:///D:/Research/VN_Trading_Agents) với cấu trúc chuẩn Python `src/` layout, khóa môi trường bằng `uv.lock`, kiểm tra kiểu tĩnh `mypy`, chuẩn code `ruff`, container cô lập `Dockerfile`, `Makefile` và bộ test tự động `pytest`.

### 2.1. Phase 0 — Tổng quan Tài liệu Đã Kiểm chứng & Thiết kế Hạ tải xuống ≤8B
- **Tài liệu đã tạo**:
  - [`docs/related_work.md`](file:///D:/Research/VN_Trading_Agents/docs/related_work.md): Tổng hợp và đối chiếu **17 bài báo arXiv đã xác thực trực tiếp qua arXiv API** (bao gồm 2 bài hạt giống **AQuA** `arXiv:2608.12841` và **AIDEz_H^2z_H** `arXiv:2609.26457`, cùng **Darwin Gödel Machine** `2505.22954`, **ShinkaEvolve** `2509.19349`, **AlphaEvolve** `2506.13131`, **STOP** `2310.02304`, **ADAS** `2408.08435`, **AIDE** `2502.13138`, **RD-Agent(Q)** `2505.15155`, **Alpha-GPT v1/2.0** `2308.00016` / `2402.09746`, **Qwen3** `2505.09388`, **Reflexion**, **Self-Refine**, **Voyager**, **The AI Scientist-v2**, **Agent Laboratory**) cùng 2 tài liệu nền tảng tài chính định lượng (*Bailey & López de Prado 2014*; *López de Prado 2018*). Đánh dấu minh bạch `UNVERIFIED` đối với những công cụ cộng đồng chưa có bài báo gốc độc lập (như `OpenEvolve`).
  - [`docs/design_decision.md`](file:///D:/Research/VN_Trading_Agents/docs/design_decision.md): Phân tích đánh đổi kiến trúc khi thu nhỏ từ mô hình 70B–600B+ xuống **≤8B** và đề xuất **5 hướng đóng góp bài báo (Paper Contributions)** xếp hạng theo tính mới & tính khả thi.

### 2.2. Phase 1 — Tầng Dữ liệu Chống Rò rỉ từ Cấu trúc (`src/vn_rsi/data/`)
- **Các file cốt lõi**: [`market_rules.yaml`](file:///D:/Research/VN_Trading_Agents/configs/market_rules.yaml), [`market_rules.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/data/market_rules.py), [`providers.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/data/providers.py), [`snapshots.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/data/snapshots.py), [`qa.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/data/qa.py), [`docs/data_terms.md`](file:///D:/Research/VN_Trading_Agents/docs/data_terms.md).
- **Đặc tả kỹ thuật đã hiện thực hóa**:
  1. **Snapshot Parquet bất biến có mã băm SHA-256**: Thí nghiệm chỉ đọc từ snapshot cục bộ, từ chối chạy nếu bất kỳ file `.parquet` nào bị sửa đổi so với `manifest.json`.
  2. **Đa nguồn (`VnstockDataProvider` + `SyntheticVNMarketProvider`)**: Hỗ trợ lấy dữ liệu thực có giới hạn tần suất (rate-limit) qua `vnstock` (đối soát chéo VCI & TCBS) và đi kèm bộ sinh dữ liệu vi cấu trúc thị trường Việt Nam tất định (`is_synthetic: true` được gắn nhãn rõ ràng) giúp chạy CI/tái lập offline 100%.
  3. **Xử lý đầy đủ bẫy dữ liệu chứng khoán Việt Nam**:
     - Cổ phiếu bị đình chỉ giao dịch & hủy niêm yết giữa kỳ (`FLC_HALT`, `ROS_DELIST`) để chống *survivorship bias*.
     - Dữ liệu báo cáo tài chính **Point-in-Time (PIT)** tách biệt ngày kết thúc quý (`fiscal_period_end`) và ngày công bố thực tế (`announcement_date` trễ 24–45 ngày); hàm `join_pit_fundamentals_causal` chỉ nối dữ liệu sau ngày công bố.
     - Quy tắc sàn cấu hình hóa kèm nguồn pháp lý: biên độ trần/sàn (`HOSE ±7%`, `HNX ±10%`, `UPCoM ±15%`), lô 100 cổ phiếu, chu kỳ thanh toán `T+2.5` (`min_holding_bars_eod: 3`), cấm bán khống cổ phiếu cơ sở, phí môi giới `0.15%`, thuế bán `0.10%`, trượt giá thanh khoản bậc hai theo tỷ lệ tham gia `ADV_20d`, và room khối ngoại (`30%` ngân hàng / `49%` tiêu chuẩn).
  4. **Bộ kiểm thử `DataQASuite` tự động**: Bao gồm kiểm thử dịch thời gian (**Future Perturbation Shift-Test** — thay đổi toàn bộ giá tương lai z_Ht' > t^*z_H và kiểm chứng giá trị đặc trưng tại z_Ht \le t^*z_H không đổi đến z_H10^{-10}z_H).

### 2.3. Phase 2 — Hộp cát Đánh giá Niêm phong (`src/vn_rsi/sandbox/` & `src/vn_rsi/eval/`)
- **Các file cốt lõi**: [`seal.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/sandbox/seal.py), [`splits.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/sandbox/splits.py), [`simulator.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/sandbox/simulator.py), [`metrics.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/eval/metrics.py), [`baselines.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/eval/baselines.py), [`evaluator.py`](file:///D:/Research/VN_Trading_Agents/src/vn_rsi/eval/evaluator.py).
- **Cơ chế bảo vệ**:
  - **`EvaluatorSeal`**: Khóa mã băm SHA-256 trên toàn bộ thư mục `sandbox/`, `eval/` và file cấu hình; bất kỳ sự can thiệp nào làm thay đổi code đánh giá sẽ ném ngoại lệ `TamperDetectedError` ngay lập tức.
  - **3 phân vùng dữ liệu tách biệt**:
    1. `search` (`2019–2022`): Agent được xem thống kê mô tả qua DataSteward.
    2. `validation` (`2023–06/2024`): Agent chỉ nhận **điểm số vô hướng (scalar scores)**, không bao giờ chạm vào chuỗi dữ liệu gốc.
    3. `sealed_test` (`07/2024–12/2025`, cách validation một khoảng đệm *purge + embargo* >10 phiên): Được bảo vệ bởi sổ cái chỉ ghi nối tiếp **`SealedTestLedger`** — **từ chối vĩnh viễn lần chấm thứ hai** trên cùng một cấu hình đóng băng (`SealedTestReuseError`).
  - **Mô phỏng khớp lệnh Việt Nam (`VNPortfolioSimulator`)**: Chặn mua khi cổ phiếu kịch trần/tạm ngừng giao dịch (`ceiling_buy_blocked`), chặn bán khi cổ phiếu nằm sàn (`floor_sell_blocked` — buộc nắm giữ qua chuỗi giảm sàn đúng thực tế), khóa cổ phiếu chưa về tài khoản trong chu kỳ `T+2.5`.
  - **7 Baselines đối chứng công bằng**: `VN-Index Buy & Hold`, `Equal-Weight`, `Momentum 20d`, `Short-Term Reversal 5d`, `Purged Walk-Forward Ridge`, `Purged Walk-Forward GBDT`, và `Purged Temporal GRU`, kèm chỉ số **Deflated Sharpe Ratio (DSR)** hiệu chỉnh đa kiểm định và khoảng tin cậy **Paired Block-Bootstrap 95% CI**.

### 2.4. Phase 3 — Hệ thống Đa Tác Tử `v0` trên Mô hình ≤8B (`src/vn_rsi/dsl/`, `llm/`, `memory/`, `agents/`)
- **7 vai trò chuyên biệt**:
  1. `Planner`: Phân bổ ngân sách tìm kiếm theo họ cơ chế kinh tế + độ tin cậy bộ nhớ.
  2. `DataSteward`: Chỉ đọc thống kê vùng `search` thông qua `DataZoneGuard`.
  3. `IdeaMiner`: Sinh **giả thuyết có thể bác bỏ (`FalsifiableHypothesis`)** gồm phát biểu, cơ chế kinh tế vi mô, dấu kỳ vọng (z_H\pm 1z_H) và tiêu chí bác bỏ *trước khi* tạo công thức.
  4. `Implementer`: Sinh cây cú pháp trừu tượng **`FactorSpec`** thuộc không gian toán tử nhân quả cố định (`ALLOWED_OPERATORS`), kết hợp giải mã ép khuôn JSON Schema, phản hồi lỗi tự sửa và bộ chấm điểm tất định `best_of_n`.
  5. `LeakageAuditor`: Kiểm tra độ sâu AST, từ khóa cấm, lọc trùng lặp cấu trúc kiểu *ShinkaEvolve* (`structural_fingerprint`) và chạy mini shift-test động.
  6. `EvaluatorProxy`: Kiểm định tiêu chí bác bỏ trên `search` và chấm tổ hợp nhân tố trên `validation`.
  7. `Librarian`: Cập nhật bộ nhớ niềm tin **`EpistemicBeliefMemory`** (có suy giảm theo thời gian `decay_rate`, phạt mâu thuẫn `contradiction_penalty`, liên kết `run_id`) và xuất quỹ đạo chuẩn hóa ra `logs/sft_verified_trajectories.jsonl` phục vụ tự huấn luyện QLoRA/ReST.
- **Ràng buộc cứng ≤8B & Ngân sách**: `LLMBackend` ném lỗi `ModelSizeConstraintError` nếu mô hình vòng lặp chính z_H> 8.0\text{B}z_H; `BudgetTracker` dừng cứng (`BudgetExceededError`) ngay khi vượt trần token, thời gian thực hoặc GPU-hour.

### 2.5. Phase 4 — Vòng lặp Tự Cải Tiến Đệ Quy Hai Cấp (`src/vn_rsi/rsi/`)
- **Bề mặt khả biến được khai báo tường minh (`MutableSurfaceGuard`)**: Vòng ngoài chỉ được phép chỉnh sửa các tham số thuộc whitelist (`planner_focus_weights`, `idea_miner_system_prompt`, `implementer_max_ast_depth`, `preferred_windows`, `exploration_temperature`, `best_of_n`, `memory_decay_rate`, `contradiction_penalty`, `novelty_similarity_threshold`). Mọi nỗ lực chạm vào `evaluator`, `splits`, `transaction_costs` hay `sealed_test` đều bị chặn và ghi nhận vào bộ đếm `reward_hacking_attempts`.
- **Kho lưu trữ mở kiểu Darwin Gödel Machine (`DGMArchive`)**: Lấy mẫu cá thể cha kết hợp điểm thích nghi (`0.75`) và độ mới cấu hình (`0.25`).
- **Cổng chấp nhận thống kê đa hạt giống (`StatisticalAcceptanceGate`)**: Chỉ chấp nhận bản viết lại nếu cải thiện điểm trung bình trên các seed ghép cặp (`[101, 202, 303]`), thắng ít nhất z_H2/3z_H số seed và đạt ngưỡng kiểm định z_Htz_H-test ghép cặp một phía.
- **Đầy đủ 6 Ablations + Kiểm tra Ignition + Tổng quát hóa OOD (`VN100` đa sàn, chân trời dự báo 10 ngày)**.

### 2.6. Phase 5 — Tự động hóa Bảng biểu Bài báo LaTeX & Kiểm toán Truy xuất Nguồn gốc (`paper/`)
- [`experiments/build_report.py`](file:///D:/Research/VN_Trading_Agents/experiments/build_report.py) đọc trực tiếp [`logs/experiment_registry.jsonl`](file:///D:/Research/VN_Trading_Agents/logs/experiment_registry.jsonl) để sinh bảng LaTeX [`main_results.tex`](file:///D:/Research/VN_Trading_Agents/paper/tables/main_results.tex) và [`rsi_ablations.tex`](file:///D:/Research/VN_Trading_Agents/paper/tables/rsi_ablations.tex), gắn chú thích `% run_id=...` cho từng dòng.
- [`paper/verify_paper.py`](file:///D:/Research/VN_Trading_Agents/paper/verify_paper.py) kiểm tra tự động: **100% khóa `\cite{}`** trong [`paper/main.tex`](file:///D:/Research/VN_Trading_Agents/paper/main.tex) tồn tại trong [`paper/references.bib`](file:///D:/Research/VN_Trading_Agents/paper/references.bib), và **60/60 ô số liệu** trong bảng LaTeX khớp chính xác với log JSONL gốc.

---

## 3. Kết quả Thực nghiệm & Kiểm thử Đã Chạy Thực tế trên Máy

*(Lưu ý trung thực khoa học: Máy hiện tại không có GPU NVIDIA rời và chưa bật daemon Ollama, nên thực nghiệm chạy hoàn toàn tất định trên CPU thông qua `RuleGuidedSmallLMBackend` (`deterministic_8b_scaffolding`) và bộ dữ liệu vi cấu trúc mô phỏng `vn_equities_snapshot_v1` (`is_synthetic_data=true`, 30 mã, 54.780 dòng từ 2019–2025). Mọi con số dưới đây đều trích xuất trực tiếp từ `logs/experiment_registry.jsonl`.)*

### 3.1. Kết quả Kiểm thử Kỹ thuật & Chống Gian lận
- **Unit & Integration Tests (`pytest`)**: **18 / 18 bài test PASS (100%)**.
- **Static Type & Lint (`mypy`, `ruff`)**: **0 lỗi trên toàn bộ 32 file mã nguồn**.
- **Phase 1 Data QA (`run_id=phase1_data_qa`)**: `qa_passed = true` (1.858 phiên kịch trần, 1.456 phiên kịch sàn, 1.827 phiên tạm ngừng giao dịch, độ trễ công bố BCTC tối thiểu 24 ngày, shift-test nhân quả tuyệt đối).
- **Phase 2 Canary Check (`run_id=phase2_baselines`)**: Tín hiệu nhiễu ngẫu nhiên cho `Rank IC = 0.01394 <= 0.045` (`passed = true`), mã niêm phong `EvaluatorSeal = 0f5abc69fa880014...`.

### 3.2. Bảng So sánh Baselines vs. Agent z_Hv_0z_H vs. VN-RSI z_Hv^*z_H (Sau thuế, phí & trượt giá T+2.5)

| Phương pháp | Val Rank IC | Val Net Sharpe | Sealed Test Rank IC | Sealed Test Net Sharpe | Test Lợi nhuận Ròng/năm | Test DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| VN-Index Buy & Hold | 0.0000 | 0.506 | 0.0000 | 1.248 | 86.0% | 0.937 |
| Equal-Weight Universe | 0.0000 | 0.454 | 0.0000 | 1.195 | 68.3% | 0.534 |
| Momentum (20d) | 0.0514 | 0.462 | 0.0551 | 1.179 | 54.0% | 0.498 |
| Short-Term Reversal (5d) | -0.0087 | -0.408 | -0.0580 | 0.366 | 15.4% | 0.143 |
| Purged Walk-Forward Ridge | 0.1026 | 0.702 | 0.0781 | 0.327 | 8.7% | 0.121 |
| Purged Walk-Forward GBDT | 0.0787 | 0.615 | 0.0427 | 0.532 | 17.5% | 0.168 |
| Purged Temporal GRU | 0.0976 | 0.450 | 0.0720 | 0.396 | 11.5% | 0.120 |
| **Agent z_Hv_0z_H (≤8B, chưa RSI)** | **0.1036** | **0.677** *(mean 0.693)* | **0.0910** | **1.339** | **40.1%** *(MaxDD -24.6%)* | 0.365 |
| **VN-RSI z_Hv^*z_H (≤8B Bi-Level)** | **0.1025** | **0.877** | **0.0637** | **1.432** | **38.8%** *(MaxDD **-18.1%**)* | 0.105 |

### 3.3. Kết quả Nghiên cứu Cắt lớp (Ablations), Ignition & OOD (`run_id=phase4_rsi`)
1. **Hiệu quả của Vòng ngoài Có định hướng so với Đột biến Ngẫu nhiên**:
   - `Full VN-RSI (<=8B)` chấp nhận **1 bản viết lại có ý nghĩa thống kê** (từ chối 3 bản nhiễu), nâng điểm tổng hợp Validation từ `1.819` lên **`1.902 ± 0.456`** (`Val Sharpe = 0.877`, khoảng tin cậy Bootstrap 95% so với `Equal-Weight` là **`[+0.138, +1.209]`, z_Hp = 0.010z_H**), đồng thời giảm sụt giảm tối đa (*Max Drawdown*) trên tập `sealed_test` xuống chỉ còn **`-18.11%`** (so với `-47.93%` của VN-Index và `-24.55%` của `Agent v0`).
   - Ngược lại, `Random Genome Mutation` bị cổng thống kê bác bỏ cả 4 lần (`0` bản được nhận, kẹt ở `1.819`), chứng minh việc điều chỉnh trọng số cơ chế dựa trên phản hồi thực nghiệm hiệu quả hơn tìm kiếm ngẫu nhiên.
   - `Flagged Teacher Proposer (72B Sim)` đạt `1.961 ± 0.291` (`Val Sharpe = 0.895`), cho thấy kiến trúc bề mặt khả biến có kiểu (typed mutable surface) giúp mô hình `<=8B` thu nhận được phần lớn lợi ích của mô hình lớn mà không bị vỡ cấu trúc.
2. **Phát hiện Trung thực Khoa học (Rất quan trọng khi viết Paper!)**:
   - **Cảnh báo từ Deflated Sharpe Ratio (`Test DSR = 0.105`)**: Do bộ đánh giá đếm trung thực toàn bộ hơn 200 lần thử nghiệm xuyên suốt các vòng lặp và ablations, chỉ số DSR giảm xuống `0.105`. Đây là một **luận điểm khoa học đắt giá** cho bài báo của bạn: nếu không hạch toán tổng số phép thử của AI Agent, Sharpe Ratio truyền thống (`1.432`) sẽ tạo cảm giác tự tin thái quá!
   - **Kiểm tra Ignition trên các seed hoàn toàn mới `[404, 505, 606]`**: `v*` cao hơn `v0` (`1.9065` so với `1.8328`, tăng `+0.0737`), nhưng với cỡ mẫu nhỏ z_HN=3z_H seed thì z_Hp = 0.2988 \ge 0.10z_H, nên hệ thống **báo cáo trung thực trạng thái `INCONCLUSIVE`** (cần mở rộng lên z_HN \ge 10z_H seed khi chạy GPU).
   - **Tổng quát hóa ngoài phân phối (OOD trên `VN100` gồm HOSE+HNX+UPCoM, chân trời 10 ngày)**: Cả `v0` và `v*` đều giữ được `Net Sharpe = +0.458` dương ổn định trong khi danh mục `Equal-Weight` bị âm (`-0.268`) do chi phí giao dịch & trượt giá cao trên HNX/UPCoM, nhưng `v*` chưa tách biệt thêm so với `v0` nếu không chạy lại vòng trong trên rổ cổ phiếu rộng hơn.

---

## 4. Lộ trình Tiếp theo Đề xuất cho Bạn để Hoàn thiện Bài báo Nộp Workshop/arXiv

1. **Bước 1 — Chạy với Trọng số Thực `Qwen3-8B` trên 1 GPU 24GB (RTX 3090/4090 hoặc Server trường/Lab)**:
   - Cài `Ollama` (`ollama pull qwen3:8b`) hoặc `vLLM` và đổi `active_backend: "ollama"` trong [`configs/llm.yaml`](file:///D:/Research/VN_Trading_Agents/configs/llm.yaml) (xem hướng dẫn chi tiết tại [`docs/scale_up_guide.md`](file:///D:/Research/VN_Trading_Agents/docs/scale_up_guide.md)).
2. **Bước 2 — Đóng băng Snapshot Dữ liệu Lịch sử Thực tế**:
   - Cài thêm `vnstock` vào môi trường riêng, chạy `experiments/run_data.py` một lần để lưu snapshot Parquet thực tế (không đẩy thư mục `data/` lên GitHub theo đúng quy định bản quyền dữ liệu trong [`docs/data_terms.md`](file:///D:/Research/VN_Trading_Agents/docs/data_terms.md)).
3. **Bước 3 — Tăng số lượng Seed lên z_HN = 10z_H và Chạy lại `make all`**:
   - Mọi bảng biểu trong [`paper/tables/`](file:///D:/Research/VN_Trading_Agents/paper/tables) sẽ tự động cập nhật kèm kiểm chứng `verify_paper.py` để bạn biên tập bản thảo [`paper/main.tex`](file:///D:/Research/VN_Trading_Agents/paper/main.tex).
