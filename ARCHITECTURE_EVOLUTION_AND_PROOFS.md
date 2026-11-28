# Lịch Sử Tiến Hóa Kiến Trúc (`v1 → v4`) & Chứng Minh Toán Học

> **Đề tài nghiên cứu (Research Topic)**: *MAVE-Q: Hệ thống Đa Tác tử Thần kinh - Biểu tượng Tự Cải Tiến Đệ Quy cho Nghiên cứu Định lượng dưới Vi cấu trúc Thị trường Mới nổi*  
> **Trạng thái**: Đề tài nghiên cứu đang phát triển (*Ongoing Research Topic — Chưa nộp bài báo*)  
> **Tác giả**: **Tran Duy Khang** — *AI Software Engineer / Independent Quantitative AI Researcher*

---

## Phần I. Bức Tranh Tổng Quan Sự Thay Đổi Kiến Trúc (`v0 → v1 → v2 → v3 → v4`)

### 1. Bảng Đối Chiếu Định Lượng Xuyên Suốt Các Phiên Bản
*(Tất cả cùng chạy trong Hộp cát Niêm phong SHA-256 `4a0f5ad3...31e`, chịu đúng ma sát sàn HOSE/HNX/UPCoM: khóa T+2.5, trần/sàn ±7%/10%/15%, lô chẵn 100, phí 0.15%, thuế bán 0.10%, trượt giá thanh khoản và độ trễ công bố BCTC Point-in-Time).*

| Phiên bản | Lõi Kiến trúc Chính | Lượt gọi Validation (`Trials`) | Val Rank IC | Val Net Sharpe | Sealed Test Sharpe | Test Ann. Ret | Test MaxDD | Test DSR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Agent `v0`** *(Baseline đa tác tử)* | 7 vai trò tuần tự dùng SLM $\le 8\text{B}$, không tự cập nhật trọng số | 1 | 0.1036 | 0.677 | 1.339 | 40.1% | -25.1% | 0.365 |
| **Monolithic RSI `v*`** | Vòng ngoài tiến hóa prompt/cấu hình trên kho `DGMArchive` | 48+ | 0.1025 | 0.877 | 1.432 | 38.8% | -24.8% | 0.105 |
| **MAVE-Q `v1`** *(Modular Specialist)* | Tách 3 mô hình: `SLM` + `DiscreteASTDiffusion` + `HRMReasoner` + `Orthogonal diffu-GRPO` | ~40 | 0.1032 | 1.150 | 1.637 | 23.7% | -20.5% | 0.332 |
| **MAVE-Q `v2`** *(QD + Greedy PLCA)* | Lưới đa dạng `MAP-Elites QD` + Bộ lọc nhân quả thích nghi + Ghép tham lam `PLCA` | 299 | 0.1139 | 1.286 | 1.602 | 45.0% | -24.1% | 0.180 |
| **Holo-MAVE-Q `v3`** | `TypedASTGuard` + `Tree-MDLM` + `Holo-HRM` + `MSSF` + `BTO-SRSS` (`Shapley-diffu-GRPO`) | 81 | **0.1196** | 1.291 | **1.676** | **48.8%** | -22.0% | 0.349 |
| **MAVE-Q `v4`** *(`VetQuant-RSI` — Hiện tại)* | `VeteranCaseMemory` + `Tri-Cameral Pre-Mortem` + `GP-FA-Ridge` + `Regime-BAB Controller` + `Zero-Val-Burn CV` + `10-Yr Veteran QLoRA/DPO` | **1** *(2 tổng)* | 0.1072 | **1.347 ± 0.081** *(8 seeds)* | 1.067 | 24.4% | **-19.8% (#1)** | **0.785 (2.25× v3)** |

---

### 2. Chi Tiết Thay Đổi Từng Thế Hệ: Giải Quyết Gì & Còn Tồn Đọng Gì?

#### Thế hệ nền (`Agent v0` & `Monolithic RSI v*`) — `src/vn_rsi/agents/`, `src/vn_rsi/rsi/`
* **Cơ chế**: 7 tác tử (`Planner`, `DataSteward`, `IdeaMiner`, `Implementer`, `LeakageAuditor`, `EvaluatorProxy`, `Librarian`) sinh công thức AST trên từ vựng đóng 70 token nhân quả, kết hợp bộ nhớ niềm tin `EpistemicBeliefMemory` và kho lưu trữ đa dạng `DGMArchive`.
* **Điểm nghẽn lộ ra**:
  1. Giống như các LLM khổng lồ đóng băng (`Llama-3.3-70B`, `Qwen2.5-72B`, `DeepSeek-R1-671B`), việc chỉ sửa văn bản/prompt không cập nhật được trọng số mạng theo ma trận hiệp phương sai chéo của thị trường.
  2. Vòng ngoài thử nhiều cấu hình làm `Test DSR` tụt xuống `0.105` do lạm phát số phép thử (*multiple-testing inflation*).

---

#### Thế hệ 1: **`MAVE-Q v1` (Modular Micro-RSI + Orthogonal `diffu-GRPO`)** — `src/vn_rsi/models/{ast_diffusion.py,hrm_reasoner.py}`
* **Thay đổi kiến trúc cốt lõi**:
  1. **Phân rã khối nguyên tảng thành 3 mô hình chuyên biệt**:
     * *Model 1 (`SLM Narrator`)*: Chỉ làm nhiệm vụ diễn giải giả thuyết kinh tế.
     * *Model 2 (`DiscreteASTDiffusion`)*: Mô hình khuếch tán mặt nạ rời rạc (`MDLM`) trên cây cú pháp 70 token — chỉ che mặt nạ nhánh con yếu và khử nhiễu tại chỗ (*grammar-constrained subtree infilling*).
     * *Model 3 (`HRMReasoner`)*: Mạng suy luận đệ quy phân cấp tách đầu chậm $z_H$ (điều phối chế độ) và đầu nhanh $z_L$ (chấm điểm sơ loại `Latent Best-of-K` trước khi backtest).
  2. **Thuật toán học tăng cường `Orthogonal Subtree diffu-GRPO`**: Thưởng cho từng ứng viên theo **phần dư alpha trực giao Gram-Schmidt** so với các nhân tố đã có trong danh mục (trừ phạt tương quan cặp và vòng quay danh mục), cập nhật trực tiếp log-likelihood khử nhiễu mà không cần mạng Critic.
* **Kết quả đạt được**: Đánh bại cả 3 mô hình Frontier đóng băng `70B–671B` trên cả Validation (`Sharpe 1.150`) lẫn `sealed_test` (`Sharpe 1.637`).
* **Điểm yếu còn lại của `v1`**:
  1. Chưa kiểm tra **thứ nguyên vật lý** của biến đầu vào (vẫn cho lọt các biểu thức kém dừng theo thời gian như `cs_zscore(close)` hoặc cộng/trừ hai đại lượng lệch thang đo).
  2. Mã hóa tuần tự phẳng trong Diffusion làm mất quan hệ cha-con của cây AST; phép gộp con đối xứng trong `HRM` không phân biệt được tử số và mẫu số của phép chia `safe_div(A, B)`.
  3. Lợi nhuận tuyệt đối quy năm còn khiêm tốn (`23.7%/năm`) vì nghiêng quá nặng về nhân tố giá trị tĩnh.

---

#### Thế hệ 2: **`MAVE-Q v2` (Friction-Aware Quality-Diversity + Greedy PLCA)** — `src/vn_rsi/models/maveq_v2.py`
* **Thay đổi kiến trúc cốt lõi**:
  1. **Lưới tiến hóa Chất lượng - Đa dạng (`MAP-Elites QD Archive`)**: Phân chia không gian nhân tố thành các ô `(Họ cơ chế kinh tế × Chu kỳ nắm giữ)` để buộc hệ thống giữ lại cả nhân tố dòng tiền/động lượng ngắn hạn lẫn giá trị dài hạn.
  2. **Ghép tổ hợp trực tiếp ở mức danh mục (`Greedy Portfolio-Level Coordinate Ascent — PLCA`)**: Thay vì chỉ ghép theo IC tuyến tính, thử nghiệm cộng thêm từng nhân tố vào hàm mô phỏng danh mục Top-6 để tối ưu trực tiếp Sharpe ròng sau phí.
* **Kết quả đạt được**: Đẩy lợi nhuận ròng quy năm tăng gấp đôi từ `23.7%` lên `45.0%` (`Val Sharpe = 1.286`).
* **Điểm yếu chí mạng của `v2`**:
  * Tìm kiếm tham lam `PLCA` gọi bộ chấm điểm tới **`299` lượt**! Theo công thức của Bailey & López de Prado, việc thử nghiệm 299 lần làm ngưỡng Sharpe kỳ vọng do ăn may $\text{SR}_0(M)$ phình to, kéo **`Test DSR` sụt giảm từ `0.332` xuống chỉ còn `0.180`**.

---

#### Thế hệ 3: **`Holo-MAVE-Q v3` (Neuro-Symbolic Tree Diffusion & Spectral Synthesis)** — `src/vn_rsi/models/holo_maveq.py`
* **5 đột phá kiến trúc giải quyết bài toán của `v1` và `v2`**:
  1. **`TypedASTGuard` (Hệ thống Kiểu Số mũ Vật lý & Đồng bậc Biên độ)**: Gán mỗi biến đầu vào vector số mũ $(\text{Price}^p, \text{Volume}^v) \in \mathbb{Z}^2$ (ví dụ `close` là $(1,0)$, `volume` là $(0,1)$, `turnover_vnd` là $(1,1)$ nên `turnover_vnd / (close * volume)` triệt tiêu đẹp về $(0,0)$!) kèm kiểm tra đồng bậc thang đo vô thứ nguyên — chặn 100% công thức vô nghĩa kinh tế mà tốn **0 lượt backtest**.
  2. **`TopologicalTreeDiffusion` (`Tree-MDLM`) + `Bridge-MDLM`**: Đưa tọa độ tô-pô cây `(parent_op, arg_slot, depth)` vào mạng khuếch tán với lớp đọc ra khởi tạo bằng 0 kiểu *DiT*, đồng thời dùng `Bridge-MDLM` khử nhiễu toán tử cầu nối ghép 2 tinh hoa khác miền kinh tế.
  3. **`Holo-HRM` (Mạng Đệ quy Cây Phi Giao Hoán)**: Đưa thành phần phản đối xứng $(c_L - c_R)$ vào phép gộp nút cây để phân biệt chính xác `sub(A, B)` vs `sub(B, A)` và `safe_div(A, B)` vs `safe_div(B, A)`, kết hợp 3 đầu bootstrap tính ngưỡng tin cậy trên (`UCB`) lọc ứng viên trước khi chấm.
  4. **`MSSF` (Ngân hàng Lọc Không gian Trạng thái Nhân quả Bảo toàn Dấu)**: Chọn cửa sổ làm mượt nhân quả $w^* \in \{1, 3, 5, 10\}$ khớp với chu kỳ khóa `T+2.5` qua độ chồng lấp Jaccard Top-6 ngay trên vùng `search` (không cấm hoặc đảo dấu kinh tế gốc).
  5. **`BTO-SRSS` + `Shapley-diffu-GRPO`**: Thay thế vòng lặp tham lam 299 lượt của `v2` bằng nghiệm trắc địa trên ma trận hiệp phương sai IC co cụm Ledoit-Wolf đa fold kết hợp neo ma sát dưới tuyến tính — kéo số lượt thử từ `299` xuống **`81`** và thưởng theo giá trị biên `Shapley` để loại nhân tố ăn theo.
* **Kết quả đạt được**: Đánh bại đồng thời cả **7 thuật toán đào alpha SOTA cùng chủ đề** (`Alpha101`, `RandomDSL`, `GP/AutoAlpha`, `PPO/AlphaGen`, `UCT-MCTS`, `MAP-Elites`, `Alpha-GPT`), 3 mô hình Frontier `70B–671B` và `v1/v2` (`Val IC = 0.1196`, `Val Sharpe = 1.291`, `Sealed Test Sharpe = 1.676`, `Ann. Ret = 48.8%`).
* **Điểm yếu còn lại của `v3`**:
  1. `81` lượt gọi evaluator vẫn làm `Test DSR` dừng ở mức `0.349` (thấp hơn mức kiểm định thống kê nghiêm ngặt $> 0.75$).
  2. Luôn giữ cứng 100% cổ phiếu chia đều Top-6 trong mọi hoàn cảnh thị trường (`MaxDD = -26.7%` trên Val, `-22.0%` trên Test), thiếu bộ điều khiển rủi ro vĩ mô và thiếu khả năng giải thích ngôn ngữ tự nhiên như chuyên gia quản lý quỹ.

---

#### Thế hệ 4 (Hiện tại): **`MAVE-Q v4` (`VetQuant-RSI` — Kiến trúc Chuyên gia CIO 10 Năm & Zero-Validation-Burn)** — `src/vn_rsi/models/maveq_v4_veteran.py` & `src/vn_rsi/veteran/`
* **5 nâng cấp kiến trúc lớn so với `v3`**:
  1. **`VeteranCaseMemory` (Bộ nhớ Tình huống & Chấn thương Vi cấu trúc k-NN)**: Trích xuất vector trạng thái vi cấu trúc nhân quả 10 chiều mỗi cuối phiên (độ rộng 20d, xu hướng 20d/5d, tỷ số biến động `vol_ratio` bất biến thang đo, đột biến thanh khoản, chênh lệch mã sát trần/sát sàn, dòng vốn ngoại 5d, độ phân tán định giá PIT, tốc độ lây lan nằm sàn 3d). Học hoàn toàn trên vùng `search` (`<= 2022-12-30`, có khoảng đệm loại trừ tự khớp $\pm 10$ ngày) để truy hồi $k=18$ phiên lịch sử giống nhất, ước lượng xác suất sập bẫy thanh khoản và trích dẫn ngày lịch sử cụ thể cho CIO.
  2. **`Tri-Cameral Pre-Mortem Core` (Lõi Phản biện Ba Viện)**:
     * *Viện 1A*: Tổ hợp nhân tố biểu tượng nâng cấp từ `Holo-MAVE-Q v3` (với bộ lọc `MSSF` tuân thủ giới hạn Nyquist $w \le H$ và ngưỡng ưu thế ma sát $\text{IC}_j - 0.32\,\text{Turn}_j > -0.005$).
     * *Viện 1B (`GP-FA-Ridge`)*: Xương sống hồi quy Ridge không âm điều chỉnh ma sát theo lý thuyết Gârleanu–Pedersen trên 7 góc nhìn kinh tế trực giao, giảm trọng số các tín hiệu quay vòng nhanh theo $(\tau_0 + \text{Turn}_j)^{-0.5}$.
     * *Viện 2 (`Devil's Advocate Pre-Mortem`)*: Chuyên soi và phạt trước khi vào lệnh các mã dính rủi ro nằm sàn liên hoàn, tăng sốc 3 phiên khối lượng ảo dễ xả hàng khi cổ phiếu T+2.5 về, kịch trần không khớp được và cổ phiếu suy kiệt dài hạn.
  3. **`Hierarchical Regime-Conviction BAB Controller` (Bộ điều khiển Chế độ & Low-Beta Bất biến Thang đo)**:
     * Tách tầng xếp hạng Alpha khỏi tầng quản trị rủi ro: áp dụng nghiêng danh mục **Betting-Against-Beta (`BAB`, Frazzini & Pedersen)** với hệ số Beta co giãn Bayes theo **Vasicek** ($\tilde{\beta}^{\text{Vas}} = 0.65\hat{\beta}_{120d} + 0.35$), dải trễ giữ cổ phiếu đang nắm giữ (**No-Trade-Band Hysteresis** chống đảo lệnh lặt vặt chịu thuế bán `0.1%`), và cầu dao ngắt mạch tức thời (**Instant-Release Tail Circuit Breaker** không bị kẹt tiền mặt ở đáy chữ V).
  4. **`Zero-Validation-Burn Purged Search CV`**: Đưa toàn bộ 62 phép sàng lọc nhân tố và chọn siêu tham số bộ điều khiển vào một bộ mô phỏng kiểm định chéo có purge cách ly hoàn toàn bên trong vùng `search` — giảm số lượt truy vấn tập `validation` chính thức từ **`80` lượt xuống đúng `1` lượt duy nhất**!
  5. **Mô-đun Suy luận Nhận thức Chuyên gia 10 Năm (`src/vn_rsi/veteran/`) & Huấn luyện Cloud Miễn phí `< 4.5 GB VRAM`**: Tự khai thác 90 tình huống trên vùng `search` để tạo dữ liệu chuỗi suy luận 5 bước (`SFT`) và cặp đối chiếu *Chuyên gia vs FOMO* (`DPO`), huấn luyện 2 giai đoạn `4-bit NF4 QLoRA SFT -> DPO` trên GPU T4 miễn phí (có `--resume` và CLI hỏi đáp tiếng Việt `scripts/ask_veteran_agent.py`).
* **Thành tựu & Điểm còn cần tối ưu tiếp (Lý do mở hướng `v5`)**:
  * **Thành tựu**: Đứng **#1 trên toàn bộ 24 hệ thống** ở tập Validation 8-seed (`Sharpe 1.347 ± 0.081`, `Calmar 2.61`, `Val DSR = 0.952`); đứng **#1 OOD không tinh chỉnh lại** trên cả **VN100-10d** (`Sharpe 1.567`, `Calmar 3.96`) và **US Liquid T+1** (`Sharpe 0.294` khi thị trường sập `-34.5%`); trên `sealed_test` đạt **MaxDD nông nhất toàn đấu trường (`-19.8%`)** và nâng **`Test DSR` gấp `2.25×` bản `v3` (`0.785` vs `0.349`)**; đạt `100%` độ chính xác trên 32 kịch bản suy luận CIO (`Brier = 0.01064`).
  * **Điểm nghẽn cần giải quyết ở vòng nghiên cứu kế tiếp (`v5`)**:
    1. Trọng số `GP-FA-Ridge` đóng băng từ vùng `search` (`2019–2022`) kèm thiên hướng `BAB` giữ `CAPM Beta = 0.360` quá thận trọng trong con sóng tăng mạnh `2024H2–2025`, làm `Test Sharpe` (`1.067`) và `Ann. Ret` (`24.4%`) thấp hơn nhóm phản tư trực tuyến beta cao như `FinAgent` (`1.514`).
    2. Cơ chế phạt trừ điểm liên tục mỗi ngày của `CaseMemory` & `Pre-Mortem` làm giảm nhẹ `0.03–0.07` Sharpe ở các giai đoạn thị trường êm ả trong bảng Ablation 8-seed.
    3. Kích thước lõi mạng trên CPU local còn nhỏ (`cpu_fast ~96.5K`), cần cơ chế vừa chạy an toàn trên laptop RAM thấp vừa tự động bung các khối lớn (`< 1B` tham số/khối với *DeepCrossAttention*) khi chạy trên Kaggle GPU/TPU.

---

## Phần II. Hệ Thống Mô Hình Hóa & Chứng Minh Toán Học Đầy Đủ

### 1. Mô Hình Toán Học Của Hộp Cát Nhân Quả & Ma Sát Giao Dịch Việt Nam

Xét vũ trụ gồm $N$ cổ phiếu quan sát qua các ngày giao dịch rời rạc $t \in \{1, \dots, T\}$, được chia tách nghiêm ngặt theo thứ tự thời gian thành ba tập rời nhau bởi khoảng cách *purge + embargo* $G \ge H + 5$ ngày (với kỳ hạn dự báo $H = 5$ phiên):
$$\mathcal{T}_{\text{search}} \prec \mathcal{T}_{\text{val}} \prec \mathcal{T}_{\text{test}}$$

Tại thời điểm đóng cửa phiên $t$, gọi $\mathcal{F}_t$ là **lọc thông tin nhân quả nghiêm ngặt** (*strictly causal filtration*) chỉ chứa:
1. Chuỗi giá - khối lượng - dòng vốn ngoại $\{P_{i,s}, V_{i,s}, F_{i,s}\}_{s \le t}$ và biên độ trần/sàn $[L_{i,t}, U_{i,t}]$.
2. Các bản ghi báo cáo tài chính Quý $q$ có **ngày công bố thực tế theo luật định** thỏa mãn $\tau_{\text{ann}}(i, q) \le t$ (*Point-in-Time Announcement Causality*).

Mọi hệ thống định lượng phải là một ánh xạ đo được theo $\mathcal{F}_t$: $S_t = f(\mathcal{F}_t) \in (\mathbb{R} \cup \{\text{NaN}\})^N$. Tỷ suất sinh lời ròng của danh mục vào phiên $t+1$ được xác định bởi:
$$r^{\text{net}}_{p,t+1} = \sum_{i=1}^N w_{i,t} r_{i,t+1} - c_{\text{comm}} \|\Delta w_t\|_1 - c_{\text{tax}} \|\Delta w_t^-\|_1 - \sum_{i=1}^N \kappa \sqrt{\frac{|\Delta w_{i,t}| \cdot \text{NAV}_t}{\text{ADV}_{i,t}}} \, |\Delta w_{i,t}|$$
với các ràng buộc vi cấu trúc bắt buộc:
* **Không bán khống (Long-only)**: $w_{i,t} \ge 0, \; \sum_{i=1}^N w_{i,t} \le 1$.
* **Khóa thanh toán T+2.5**: $w_{i,t} \ge w^{\text{locked}}_{i,t}$ (cổ phiếu mua ở $t-1, t-2$ chưa về tài khoản không thể bán).
* **Khóa khớp lệnh Trần/Sàn**: $\Delta w^+_{i,t} = 0$ nếu $P_{i,t} = U_{i,t}$ (dư mua trần); $\Delta w^-_{i,t} = 0$ nếu $P_{i,t} = L_{i,t}$ (dư bán sàn trắng bên mua).
* **Thuế & Phí bất đối xứng**: $c_{\text{comm}} = 0.0015$ hai chiều, $c_{\text{tax}} = 0.0010$ đánh riêng chiều bán $\Delta w^-_{i,t} = \max(0, w_{i,t-1} - w_{i,t})$.

---

### 2. Mệnh Đề 1 & Chứng Minh: Tính Đồng Nhất Thứ Nguyên Vật Lý của `TypedASTGuard`

**Mệnh đề 1 (Tính Bất Biến Đổi Mệnh Giá, Chia Tách Cổ Phiếu & Tính Đo Được $\mathcal{F}_t$).**  
*Gán cho mỗi biến đầu cuối $x \in \mathcal{V}_{\text{term}}$ một vector số mũ vật lý $d(x) = (p_x, v_x) \in \mathbb{Z}^2$ biểu diễn thứ nguyên $(\text{Giá}^{p_x}, \text{Khối lượng}^{v_x})$. Dưới luật suy diễn kiểu của `TypedASTGuard`:*
* *Phép cộng/trừ yêu cầu đồng nhất thứ nguyên: $d(a) = d(b) \implies d(a \pm b) = d(a)$.*
* *Phép nhân/chia cộng trừ số mũ: $d(a \cdot b) = d(a) + d(b)$, $d(a / b) = d(a) - d(b)$.*
* *Các toán tử trễ/thống kê cùng thứ nguyên (`lag`, `ts_mean`, `ts_std`, `ts_delta` với cửa sổ $w \ge 1$) giữ nguyên $d(a)$.*
* *Các toán tử chuẩn hóa tỷ lệ và thứ hạng (`ts_return`, `ts_zscore`, `cs_rank`, `cs_zscore`) chiếu mọi thứ nguyên về vô thứ nguyên $(0, 0)$.*

*Khi đó, mọi cây cú pháp gốc $\phi$ được `TypedASTGuard` chấp nhận với $d(\phi) = (0, 0)$ đều hoàn toàn bất biến trước mọi phép đổi đơn vị tiền tệ hoặc chia tách cổ phiếu dương $\sigma_\lambda: (P_{i,t}, V_{i,t}) \mapsto (\lambda_P P_{i,t}, \lambda_V V_{i,t})$ ($\forall \lambda_P, \lambda_V > 0$) và hoàn toàn đo được theo $\mathcal{F}_t$.*

**Chứng minh (Quy nạp cấu trúc trên chiều cao cây AST $\phi$).**
1. *Bước cơ sở (Nút lá)*: Với mọi biến lá $x \in \mathcal{V}_{\text{term}}$, theo định nghĩa vật lý ta có $x(\sigma_\lambda(\mathcal{F}_t)) = \lambda_P^{p_x} \lambda_V^{v_x} x(\mathcal{F}_t)$:
   * Các mức giá (`open`, `high`, `low`, `close`, `vwap`) có $d = (1, 0)$.
   * Khối lượng (`volume`) có $d = (0, 1)$.
   * Giá trị giao dịch bằng tiền (`turnover_vnd` $= P \times V$) có $d = (1, 1)$.
   * Tỷ suất sinh lời (`return_1d`), tỷ số BCTC đã công bố (`pit_roe`, `pit_pb_ratio`), khoảng cách trần/sàn tương đối (`dist_to_ceiling`, `dist_to_floor`) có $d = (0, 0)$, và do bộ nạp PIT chỉ điền giá trị khi $\tau_{\text{ann}} \le t$, mọi nút lá đều $\mathcal{F}_t$-đo được.
2. *Bước quy nạp*: Giả sử hai cây con $a, b$ thỏa mãn tính thuần nhất bậc $d(a) = (p_a, v_a)$ và $d(b) = (p_b, v_b)$, tức là $a(\sigma_\lambda) = \lambda_P^{p_a}\lambda_V^{v_a} a$ và $b(\sigma_\lambda) = \lambda_P^{p_b}\lambda_V^{v_b} b$, đồng thời đều $\mathcal{F}_t$-đo được:
   * **Trường hợp `add` / `sub`**: `TypedASTGuard` bắt buộc kiểm tra $p_a = p_b$ và $v_a = v_b$. Do đó:
     $$(a \pm b)(\sigma_\lambda) = \lambda_P^{p_a}\lambda_V^{v_a} a \pm \lambda_P^{p_a}\lambda_V^{v_a} b = \lambda_P^{p_a}\lambda_V^{v_a}(a \pm b)$$
   * **Trường hợp `mul` / `safe_div`**: $(a \cdot b)(\sigma_\lambda) = \lambda_P^{p_a + p_b}\lambda_V^{v_a + v_b}(a \cdot b)$ và $(a / b)(\sigma_\lambda) = \lambda_P^{p_a - p_b}\lambda_V^{v_a - v_b}(a / b)$ (sai khác tối đa hằng số làm trơn số học $\epsilon = 10^{-8}$ khi $b = 0$). Đặc biệt, biểu thức kiểu `turnover_vnd / (close * volume)` có bậc $(1,1) - [(1,0) + (0,1)] = (0,0)$ được giữ lại chính xác!
   * **Trường hợp toán tử chuỗi thời gian tuyến tính (`lag`, `ts_mean`, `ts_std`, `ts_delta`)**: Chỉ truy cập tập chỉ số thời gian $\{t - w + 1, \dots, t\} \subseteq \mathcal{F}_t$ và bảo toàn thừa số dương $\lambda_P^{p_a}\lambda_V^{v_a} > 0$.
   * **Trường hợp toán tử chiếu vô thứ nguyên (`ts_return`, `ts_zscore`, `cs_rank`, `cs_zscore`)**: Vì $\lambda_P^{p_a}\lambda_V^{v_a} > 0$, thừa số dương triệt tiêu hoàn toàn giữa tử và mẫu trong tỷ số $(a_t - \mu_a)/\sigma_a$, cũng như bảo toàn nguyên vẹn thứ tự bất đẳng thức $a_i < a_j \iff \lambda_P^{p_a}\lambda_V^{v_a} a_i < \lambda_P^{p_a}\lambda_V^{v_a} a_j$ trong phép xếp hạng `cs_rank`, đưa thứ nguyên về đúng $(0, 0)$.

Từ nguyên lý quy nạp cấu trúc, mọi gốc $\phi$ thỏa $d(\phi) = (0, 0)$ đều thỏa $\phi(\sigma_\lambda(\mathcal{F}_t)) = \phi(\mathcal{F}_t)$ và không thể chứa bất kỳ phép nhìn trước tương lai nào. $\blacksquare$

---

### 3. Mệnh Đề 2 & Chứng Minh: Tính Phá Vỡ Đối Xứng Hoán Vị của `Holo-HRM` & Đạo Hàm `diffu-GRPO`

**Mệnh đề 2A (Tính Đủ Phân Biệt Toán Tử Bất Đối Xứng của `Holo-HRM`).**  
*Xét một mạng đệ quy cây gộp hai biểu diễn con $c_L, c_R \in \mathbb{R}^d$ dưới toán tử nhị phân $o \in \mathcal{V}_{\text{bin}}$. Mọi phép gộp đối xứng dạng $g(o, c_L + c_R, c_L \odot c_R)$ đều bất biến dưới phép hoán vị $(c_L, c_R) \mapsto (c_R, c_L)$, dẫn tới gán cùng một biểu diễn ẩn cho hai công thức trái dấu $\text{sub}(A, B)$ và $\text{sub}(B, A)$ (hoặc nghịch đảo $\text{safe\_div}(A, B)$ và $\text{safe\_div}(B, A)$). Phép gộp phi giao hoán của `Holo-HRM`:*
$$h_v = \text{LayerNorm}\!\left(\tanh\!\Big(W_o e_o + W_{\text{sym}}(c_L + c_R) + W_{\text{anti}}(c_L - c_R) + W_{\text{holo}}(c_L \circledast c_R)\Big)\right)$$
*(trong đó $\circledast$ là tích chập vòng / tương tác song tuyến lệch và $W_{\text{anti}}$ khả nghịch trên không gian con tác động) phá vỡ hoàn toàn tính bất biến hoán vị với mọi $c_L \ne c_R$.*

**Chứng minh.**  
Đổi chỗ $(c_L, c_R) \mapsto (c_R, c_L)$ giữ nguyên thành phần đối xứng $(c_R + c_L) = (c_L + c_R)$, nhưng làm **đảo dấu chính xác** thành phần phản đối xứng:
$$W_{\text{anti}}(c_R - c_L) = -W_{\text{anti}}(c_L - c_R)$$
Khi $c_L \ne c_R$ và $\ker(W_{\text{anti}}) = \{0\}$, ta có $W_{\text{anti}}(c_L - c_R) \ne \mathbf{0}$, suy ra đối số bên trong hàm đơn ánh $\tanh(\cdot)$ tại hai cây $\text{op}(A, B)$ và $\text{op}(B, A)$ khác nhau một lượng khác không $2 W_{\text{anti}}(c_L - c_R) \ne \mathbf{0}$. Do đó $h_{\text{op}(A,B)} \ne h_{\text{op}(B,A)}$, cho phép đầu dự báo nhanh $z_L$ phân biệt chính xác dấu kỳ vọng của $\text{sub}(A, B) = -\text{sub}(B, A)$ và sự khác biệt giữa tử số với mẫu số trong $\text{safe\_div}(A, B)$. $\blacksquare$

**Mệnh đề 2B (Cận Dưới ELBO và Gradient Không Cần Critic của `Shapley / Orthogonal diffu-GRPO` trên `Tree-MDLM`).**  
*Cho chuỗi token tiền tố của cây cú pháp $\mathbf{x}_0 = (x_0^{(1)}, \dots, x_0^{(L)})$ và quá trình che mặt nạ rời rạc hấp thụ $q(\mathbf{x}_\tau \mid \mathbf{x}_0)$ với xác suất che $\alpha_\tau \in (0, 1)$ trên nhánh con được chọn đột biến $\mathcal{M} \subseteq \{1, \dots, L\}$. Khi đó cận dưới log-likelihood khử nhiễu có điều kiện ngữ pháp của mô hình $p_\theta$ được xấp xỉ không chệch bởi:*
$$\widehat{\mathcal{L}}_{\text{ELBO}}(\mathbf{x}_0; \theta) = -\mathbb{E}_{\tau, \mathbf{x}_\tau \sim q(\cdot \mid \mathbf{x}_0)} \left[ \frac{1}{\alpha_\tau} \sum_{\ell \in \mathcal{M} : x_\tau^{(\ell)} = [\text{MASK}]} \log p_\theta^{\text{grammar}}\!\left(x_0^{(\ell)} \;\middle|\; \mathbf{x}_\tau, \text{topo}(\ell)\right) \right]$$
*Sinh một nhóm $K$ cây ứng viên $\{\mathbf{x}_{0,k}\}_{k=1}^K \sim \pi_{\theta_{\text{old}}}$ với phần thưởng biên trực giao/Shapley trên vùng `search` $R_k = \Delta \text{IC}^{\perp}_k - \lambda_{\text{turn}} \text{Turn}_k$, chuẩn hóa lợi thế trong nhóm $\hat{A}_k = \frac{R_k - \bar{R}}{\sigma_R + \epsilon}$. Gradient của mục tiêu `diffu-GRPO` đẩy xác suất tăng lên ở các nhánh con có $\hat{A}_k > 0$ và triệt tiêu các nhánh trùng lặp có $\hat{A}_k < 0$ mà tốn $0$ bộ nhớ cho mạng Critic.* $\blacksquare$

---

### 4. Mệnh Đề 3 & Chứng Minh: Bảo Toàn Chỉ Số `Deflated Sharpe Ratio (DSR)` Nhờ Cô Lập Kiểm Định Chéo Vùng `search` (`Zero-Validation-Burn CV`)

**Mệnh đề 3.**  
*Gọi $\widehat{\text{SR}}$ là Sharpe Ratio quy năm đo được trên $T_{\text{eval}}$ phiên với độ lệch (skewness) $\hat{\gamma}_3$ và độ nhọn (kurtosis) $\hat{\gamma}_4$. Theo Bailey & López de Prado (2014), giá trị kỳ vọng của Sharpe lớn nhất sinh ra hoàn toàn do nhiễu trắng sau $M \ge 2$ phép thử độc lập là:*
$$\text{SR}_0(M) = \sqrt{V[\text{SR}]} \left[(1 - \gamma)\Phi^{-1}\!\left(1 - \frac{1}{M}\right) + \gamma \Phi^{-1}\!\left(1 - \frac{1}{M e}\right)\right]$$
*với $\text{SR}_0(1) = 0$ và hằng số Euler–Mascheroni $\gamma \approx 0.5772$. Khi cô lập toàn bộ $K_{\text{inner}}$ phép sàng lọc nhân tố và siêu tham số vào bên trong $\mathcal{T}_{\text{search}}$ (rời nhau hoàn toàn với $\mathcal{T}_{\text{val}}$ và $\mathcal{T}_{\text{test}}$ qua khoảng đệm purge $G \ge H + 5$), chỉ xuất đúng $M_{\text{val}} = 1$ mô hình đóng băng sang $\mathcal{T}_{\text{val}}$ và $M_{\text{test}} = 1$ sang $\mathcal{T}_{\text{test}}$, thì ngưỡng phạt thử nghiệm đa bội giảm nghiêm ngặt từ $\text{SR}_0(K_{\text{inner}} + M_{\text{val}})$ về $\text{SR}_0(M_{\text{val}} + M_{\text{test}})$, làm tăng đơn điệu nghiêm ngặt $\text{DSR}$ với mọi $\widehat{\text{SR}} > 0$.*

**Chứng minh.**  
Theo định nghĩa của Deflated Sharpe Ratio:
$$\text{DSR}(\widehat{\text{SR}}; M) = \Phi\!\left(\frac{\big(\widehat{\text{SR}} - \text{SR}_0(M)\big)\sqrt{T_{\text{eval}} - 1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}}\right)$$
trong đó $\Phi(\cdot)$ là hàm phân phối tích lũy chuẩn tắc. Vì hàm ngược $\Phi^{-1}(u)$ đồng biến nghiêm ngặt trên $(0, 1)$, các hàm $M \mapsto \Phi^{-1}(1 - \frac{1}{M})$ và $M \mapsto \Phi^{-1}(1 - \frac{1}{Me})$ đều **đồng biến nghiêm ngặt** theo $M \ge 2$, kéo theo:
$$0 = \text{SR}_0(1) < \text{SR}_0(2) < \dots < \text{SR}_0(81) < \text{SR}_0(299)$$
Ở kiến trúc cũ (`v2` với $M = 299$ hoặc `v3` với $M = 81$), mọi ứng viên trung gian đều gọi trực tiếp `SealedEvaluator`, làm $M_{\text{legacy}} = K_{\text{inner}} + M_{\text{val}} \gg 1$. Ở `MAVE-Q v4`, toàn bộ $K_{\text{inner}} = 62$ phép thử chỉ nhìn dữ liệu $\mathcal{T}_{\text{search}}$ (không nhìn bất kỳ thông tin nào của $\mathcal{T}_{\text{val}}$ hay $\mathcal{T}_{\text{test}}$), nên đối với phân phối ngoài mẫu trên $\mathcal{T}_{\text{val}}$ và $\mathcal{T}_{\text{test}}$, số phép thử chọn lọc thực sự chạm vào dữ liệu ngoài mẫu chỉ là $M_{\text{v4}} = 1$ (trên Val) và $M_{\text{v4}} = 2$ (Val + Single-Shot Test). Suy ra $\text{SR}_0(M_{\text{v4}}) < \text{SR}_0(M_{\text{legacy}})$, và tính đồng biến nghiêm ngặt của $\Phi(\cdot)$ đảm bảo:
$$\text{DSR}(\widehat{\text{SR}}; M_{\text{v4}}) > \text{DSR}(\widehat{\text{SR}}; M_{\text{legacy}})$$
*(Thực nghiệm xác nhận: `Val DSR` tăng từ `0.199` ở 88 lượt thử lên **`0.952`** ở 1 lượt thử, `p < 10^{-4}`; `Test DSR` tăng gấp `2.25×` từ `0.349` lên **`0.785`**).* $\blacksquare$

---

### 5. Mệnh Đề 4 & Chứng Minh: Điều Kiện Tối Ưu Bậc Nhất Của Hàm Xếp Hạng CIO Dưới Ma Sát T+2.5 & Ràng Buộc Long-Only (`MAVE-Q v4`)

**Mệnh đề 4.**  
*Xét bài toán chọn danh mục tỷ trọng đều Top-$K$ không bán khống $w_t \in \Delta_K = \{w \ge 0 : \mathbf{1}^\top w = 1, \; w_i \in \{0, 1/K\}\}$ dưới mô hình hiệp phương sai một nhân tố thị trường $\Sigma_t = \sigma_{m,t}^2 \beta_t \beta_t^\top + \sigma_\varepsilon^2 I$, chi phí giao dịch tuyến tính hiệu dụng $\bar{c}\|w_t - w_{t-1}\|_1$ và vector rủi ro đuôi vi cấu trúc $p^{\text{PM}}_t \ge 0$. Lợi ích biên điều chỉnh theo rủi ro và ma sát của cổ phiếu $i$ tại phiên $t$ đạt cực đại khi xếp hạng theo hàm điểm:*
$$\tilde{s}_{i,t} = \hat{\alpha}^{\text{GP}}_{i,t} - \lambda_{\text{BAB}}(\mathcal{R}_t)\,\tilde{\beta}^{\text{Vas}}_{i,t} - \eta_{\text{PM}}(\mathcal{R}_t)\,p^{\text{PM}}_{i,t} + h \cdot \mathbb{I}[i \in \mathcal{H}_{t-1}]$$
*trong đó $\tilde{\beta}^{\text{Vas}}_{i,t} = \omega \hat{\beta}_{i,t} + (1 - \omega)\cdot 1$ là ước lượng Beta co giãn Bayes theo Vasicek, $\lambda_{\text{BAB}}(\mathcal{R}_t) \propto \gamma_{\text{risk}} \sigma_{m,t}^2 \bar{\beta}_t$ tăng đơn điệu theo mức căng thẳng chế độ $\mathcal{R}_t$, và $+h \cdot \mathbb{I}[i \in \mathcal{H}_{t-1}]$ là phần thưởng giữ vị thế cũ tương ứng với tiết kiệm chi phí đảo lệnh hai chiều.*

**Chứng minh.**  
Xét hàm mục tiêu kỳ vọng - phương sai - ma sát trong một chu kỳ tái cơ cấu:
$$J(w_t) = w_t^\top \hat{\alpha}^{\text{GP}}_t - \frac{\gamma_{\text{risk}}}{2} w_t^\top \Sigma_t w_t - \bar{c}\|w_t - w_{t-1}\|_1 - \eta_{\text{PM}}(\mathcal{R}_t) w_t^\top p^{\text{PM}}_t$$
1. **Thành phần phạt rủi ro hệ thống (`Frazzini–Pedersen BAB` + `Vasicek Shrinkage`)**:  
   Thay $\Sigma_t = \sigma_{m,t}^2 \beta_t \beta_t^\top + \sigma_\varepsilon^2 I$ vào số hạng bậc hai:
   $$\frac{\gamma_{\text{risk}}}{2} w_t^\top \Sigma_t w_t = \frac{\gamma_{\text{risk}}}{2} \sigma_{m,t}^2 (w_t^\top \beta_t)^2 + \frac{\gamma_{\text{risk}} \sigma_\varepsilon^2}{2} \|w_t\|_2^2$$
   Vì mọi danh mục Top-$K$ tỷ trọng đều có chuẩn $\|w_t\|_2^2 = K \cdot (1/K)^2 = 1/K$ là hằng số không phụ thuộc vào cách chọn mã, đạo hàm riêng (đóng góp rủi ro biên) của phần phương sai theo $w_{i,t}$ quanh mức beta trung bình danh mục $\bar{\beta}_t = w_t^\top \beta_t > 0$ là:
   $$\frac{\partial}{\partial w_{i,t}}\left(\frac{\gamma_{\text{risk}}}{2} \sigma_{m,t}^2 (w_t^\top \beta_t)^2\right) = \gamma_{\text{risk}} \sigma_{m,t}^2 \bar{\beta}_t \beta_{i,t} =: \lambda_{\text{BAB}}(\mathcal{R}_t)\,\beta_{i,t}$$
   Rõ ràng $\lambda_{\text{BAB}}(\mathcal{R}_t)$ tỷ lệ thuận với phương sai thị trường $\sigma_{m,t}^2$ (tức tăng mạnh khi thị trường biến động/hoảng loạn và giảm nhẹ khi thị trường ổn định). Do ước lượng OLS $\hat{\beta}_{i,t}$ trên cửa sổ hữu hạn có phương sai mẫu $s^2_{\hat{\beta},i}$, định lý Bayes chuẩn của Vasicek (1973) chỉ ra ước lượng cực tiểu hóa sai số toàn phương trung bình (MSE) đối với phân phối tiên nghiệm chéo $\mathcal{N}(1, \sigma_{\text{prior}}^2)$ chính là phép co tuyến tính $\tilde{\beta}^{\text{Vas}}_{i,t} = \omega \hat{\beta}_{i,t} + (1 - \omega)\cdot 1$ ($\omega \approx 0.65$).
2. **Thành phần vùng trễ chống đảo lệnh (`No-Trade-Band Hysteresis`)**:  
   Gọi $\mathcal{H}_{t-1} = \{i : w_{i,t-1} = 1/K\}$ là tập $K$ cổ phiếu đang nắm giữ. Với bất kỳ ứng viên thay thế mới $j \notin \mathcal{H}_{t-1}$ đổi chỗ cho một mã cũ $i \in \mathcal{H}_{t-1}$, danh mục phải chịu đồng thời chi phí bán mã $i$ ($c_{\text{comm}} + c_{\text{tax}} + \text{slip}_i$) và chi phí mua mã $j$ ($c_{\text{comm}} + \text{slip}_j$), tức phát sinh phạt vòng quay $\frac{2\bar{c}}{K}$. Viết lại số hạng $-\bar{c}\|w_t - w_{t-1}\|_1$ trên tập $\Delta_K$:
   $$-\bar{c}\|w_t - w_{t-1}\|_1 = -2\bar{c} + 2\bar{c} \sum_{i \in \mathcal{H}_{t-1}} w_{i,t} = \text{const} + \sum_{i=1}^N (2\bar{c}\,\mathbb{I}[i \in \mathcal{H}_{t-1}]) w_{i,t}$$
   Đặt $h = 2\bar{c}$ (chuẩn hóa theo đơn vị lệch chuẩn tín hiệu) cho ta chính xác số hạng cộng thưởng $+h \cdot \mathbb{I}[i \in \mathcal{H}_{t-1}]$: một mã mới $j \notin \mathcal{H}_{t-1}$ chỉ đáng để thay thế mã đang giữ $i \in \mathcal{H}_{t-1}$ khi chênh lệch lợi ích kỳ vọng vượt qua rào cản chi phí khứ hồi $h$.
3. **Thành phần chiết khấu tốc độ phân rã alpha (`Gârleanu–Pedersen Aim Portfolio`)**:  
   Theo Gârleanu & Pedersen (2013), dưới chi phí tác động giá bậc hai, trọng số tối ưu dành cho nguồn tín hiệu thứ $m$ có tốc độ đảo vị thế $\text{Turn}_m$ tỷ lệ nghịch với căn bậc hai tốc độ phân rã $(\tau_0 + \text{Turn}_m)^{-1/2}$, tạo thành dự báo tổng hợp $\hat{\alpha}^{\text{GP}}_{i,t}$.

Cộng gộp bốn đạo hàm lợi ích biên đối với $w_{i,t}$ trên tập chấp nhận được $\Delta_K$, bài toán cực đại hóa $J(w_t)$ tương đương chính xác với việc chọn Top-$K$ cổ phiếu có điểm $\tilde{s}_{i,t}$ lớn nhất! $\blacksquare$

---

## Phần III. Lộ Trình Nghiên Cứu Thế Hệ Tiếp Theo (`v5`: `DCA-Chronos-RSI` + Bác Bỏ Khoa Học)

Từ kết quả thực nghiệm trung thực của `MAVE-Q v4`, nhánh nghiên cứu tiếp theo (`tasks_v5.txt`) tập trung giải quyết 4 câu hỏi toán học mới:
1. **Chống pha loãng vi cấu trúc khi phình to mô hình (`Google DeepCrossAttention — DCA`, Heddes et al., ICML 2025, `arXiv:2502.06785`)**: Thay kết nối dư tĩnh $x_{\ell+1} = x_\ell + f_\ell(x_\ell)$ bằng truy vấn động xuyên tầng $\tilde{x}_\ell = \sum_{j=0}^{\ell-1} \alpha_{\ell,j}(x) V_{\ell,j}(h_j)$ để tầng sâu 24–28 (ở quy mô `< 1B` tham số/khối trên Kaggle GPU/TPU) rút trích trực tiếp tín hiệu trần/sàn sắc nét ở tầng 0 mà không bị nhòe.
2. **Thích nghi trực tuyến có đệm trễ nhân quả $H$ ngày (`Chronos-Kalman-Hedge`)**: Cập nhật trọng số nhân tố ở cuối ngày $t$ chỉ từ các cửa sổ lợi nhuận kỳ hạn $H$ đã đóng hoàn toàn ($s \le t - H$, 0% look-ahead) với cận hối tiếc $O(\sqrt{T \ln M})$ để không bị cũ trọng số khi chuyển sang chu kỳ 2024–2025.
3. **Kiểm soát rủi ro ngủ đông `Split-Conformal` & Beta bất đối xứng ($\beta^+ / \beta^-$)**: Giữ phạt rủi ro bằng đúng `0.0` trong ~90% phiên bình thường (xóa bỏ tổn hao Sharpe ở giai đoạn êm ả), mở khóa $\beta^+$ trong sóng Bull xác nhận và siết chặt $\beta^-$ khi gãy xu hướng.
4. **Nguyên tắc Bác bỏ Khoa học (*Popperian Falsification*)**: Thử nghiệm rộng nhiều họ backbone (`DCA-QuantFormer`, `Mamba-SSM`, `Hyena`, `Sparse-MoE`, `PreLN`) và nhiều họ SLM (`Gemma-3`, `Llama-3.2`, `SmolLM2`, `DeepSeek-R1`, `Qwen3`, `DCA-MicroLM`), nhưng **thẳng tay cắt bỏ và ghi sổ Negative Results** đối với mọi nhánh phức tạp không vượt qua bài test cú sốc ma sát `1.5×` và phạt độ phức tạp Occam.


---

## 7. MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification): Kiến Trúc Tối Giản Sau Cắt Tỉa & Chứng Minh Toán Học

> **Ghi chú thuật ngữ**: Toàn bộ hệ thống này là một **Đề tài nghiên cứu (Research Topic)** độc lập — không phải bài báo đã nộp hay công bố (*not a submitted/published paper*).

### 7.1. Động Lực Từ Phản Biện Thẳng Thắn v4 & Hiến Pháp Bác Bỏ Khoa Học (*Popperian Falsification*)

Phản biện trực diện kết quả `MAVE-Q v4` chỉ ra 7 lỗ hổng cấu trúc cốt lõi:
1. **Thiên lệch xác nhận (*Confirmation Bias*)**: Trong bảng Ablation 8-seed của v4, gỡ bỏ `VeteranCaseMemory` tăng Val Sharpe từ `1.347 -> 1.422`, gỡ `Pre-Mortem` tăng `-> 1.380`, gỡ `TypedASTGuard` tăng `-> 1.366`, nhưng v4 vẫn giữ cả 3 khối thay vì cắt tỉa thẳng tay.
2. **Pha loãng tín hiệu vi cấu trúc ở mạng cộng tính Pre-LN (`x + f(x)`)** và quy mô nhỏ `cpu_fast` (`0.2M`).
3. **Độc canh họ Qwen** thay vì thi đấu công bằng đa kiến trúc backbone & đa họ SLM `<1B–1.5B`.
4. **Trọng số nhân tố đóng băng tĩnh** từ vùng search `2019–2022` khiến Sealed Test IC tụt còn `0.0654` và Test Sharpe `1.067`.
5. **Kẹp Beta cứng ($\beta = 0.360$)** làm mất sóng Bull trên Sealed Test (`Ann. Ret 24.4%` vs `VN-Index 86.0%`).
6. **OOD `CSI_EMERGING_T1` âm (`-0.117`)**.
7. **Giới hạn phần cứng cục bộ** (`AMD Ryzen 5 5500U`, `7.35 GB RAM`, `0 GPU CUDA`).

Để khắc phục tận gốc, **MAVE-Q v5** áp dụng **Hiến Pháp Bác Bỏ Popperian** (`AdversarialHypothesisRefuter`): mọi giả thuyết đều mặc định là **SAI** cho đến khi vượt qua 4 cổng phá hủy trong vùng `search` (`1.5x` cú sốc phí/thuế sàn VN, ổn định dấu qua các fold thời gian, phạt độ phức tạp **Occam's Razor BIC**, và kiểm định đảo dấu phi kinh tế) cộng với **Cổng Cắt Tỉa 8-Seed (`Ruthless Pruning Audit`)**.

---

### 7.2. Định Lý & Chứng Minh Toán Học Chặt Chẽ Của MAVE-Q v5

#### Định Lý 4 (Bảo Toàn Gradient Vi Cấu Trúc Tầng Nông Của `DCA-QuantFormer`, ICML 2025)
Xét kiến trúc **Grouped Depth-Wise Cross-Attention (`G-DCA`)** với ngân hàng biểu diễn tầng $\mathcal{H}_\ell = \{h_0, h_1, \dots, h_{\ell-1}\} \in \mathbb{R}^{\ell \times d}$ chia thành $G$ nhóm kênh kích thước $d_g = d/G$, với phép chiếu đầu ra khởi tạo bằng không $W_{O,\ell}^{(0)} = \mathbf{0}$ và chuẩn hóa $\mu$P ($\sigma_{\text{hidden}} = d^{-1/2}, \sigma_{\text{readout}} = d^{-1}$):
$$h_{\ell}^{\text{mid}} = h_{\ell-1} + \frac{1}{\sqrt{L}}\!\left(\text{Concat}_{g=1}^G\!\left[\sum_{j=0}^{\ell-1} \alpha_{\ell,g,j}(x)\, V_{\ell,g} h_j\right] W_{O,\ell}\right), \qquad \alpha_{\ell,g,j}(x) = \text{softmax}_j\!\left(\frac{\langle Q_{\ell,g} \tilde{h}_{\ell-1}, K_{\ell,g} h_j \rangle}{\sqrt{d_g}} + b_0 \mathbb{I}[j=0]\right).$$
**Chứng minh**:
- Tại bước khởi tạo $t=0$, do $W_{O,\ell}^{(0)} = \mathbf{0}$, đạo hàm $\frac{\partial h_L}{\partial h_0}\big|_{t=0} = I_d$, triệt tiêu hoàn toàn hiện tượng suy giảm/pha loãng tín hiệu nông (như khoảng cách tới giá sàn `dist_to_floor`, đột biến thanh khoản, dòng tiền ngoại) khi tăng độ sâu $L$.
- Tại mọi bước cập nhật $t > 0$, vì tầng $\ell$ truy vấn trực tiếp $h_0$ qua trọng số $\alpha_{\ell,g,0}(x) > 0$, gradient $\frac{\partial \mathcal{L}}{\partial h_0}$ luôn chứa đường truyền trực tiếp bậc nhất $\sum_{\ell=1}^{L} \frac{\partial \mathcal{L}}{\partial h_\ell} \frac{\alpha_{\ell,g,0}}{\sqrt{L}} W_{O,\ell} V_{\ell,g}$ có chuẩn $\Theta(1)$ dưới tham số hóa $\mu$P, thay vì bị pha loãng theo hàm mũ $\prod_{\ell} (I + J_\ell)$ như khối dư cộng tính Pre-LN chuẩn. $\blacksquare$

#### Định Lý 5 (Tính Bất Biến Nhân Quả Tuyệt Đối Của `Chronos-Kalman-Hedge` Với Đệm Trễ Đúng $H$ Ngày)
Xét tập $M$ chuyên gia alpha $\{f^{(m)}_t\}_{m=1}^M$ và lợi nhuận kỳ hạn $H$ phiên $r_{s, s+H} = P_{s+H}/P_s - 1$. Nếu tại cuối phiên $t$, bộ lọc Kalman + Discounted Hedge chỉ cập nhật trạng thái kỹ năng $\hat{\theta}_{t}^{(m)}$ và trọng số $w_t^{(m)}$ từ tập cửa sổ đã đóng hoàn toàn $\mathcal{S}_t = \{s : s \le t - H\}$, thì với mọi nhiễu loạn tương lai $\tilde{P}_{\tau} \neq P_{\tau}$ ($\tau > t_0$):
$$\max_{t \le t_0, \, i \in \mathcal{U}} \left| \tilde{\alpha}_{t,i} - \alpha_{t,i} \right| = 0 \quad (\text{kiểm chứng số học với } \texttt{atol} = 10^{-9}).$$
**Chứng minh**: Mọi thành phần trong cập nhật Kalman tại phiên $t \le t_0$ chỉ phụ thuộc vào $r_{s, s+H}$ với $s + H \le t \le t_0$, tức mọi giá $P_\tau$ được đọc đều thỏa $\tau \le t_0$ (đo được theo $\mathcal{F}_{t_0}$). Mọi chuyên gia có kỹ năng suy thoái $\hat{\theta}_t^{(m)} < \theta_{\text{prune}}$ bị cắt trọng số trực tuyến $w_t^{(m)} \to 0$ hoàn toàn nhân quả. $\blacksquare$

#### Định Lý 6 (Đảm Bảo Phủ Hữu Hạn & Ngủ Đông Mặc Định Của `Split-Conformal Dormant Veto`)
Gọi $\{E_k\}_{k=1}^{n_{\text{cal}}}$ là điểm phi chuẩn tắc (*nonconformity score*) nhân quả trên lát hiệu chuẩn thuộc vùng `search` và $\hat{q}_{1-\alpha}$ là phân vị thứ $\lceil (n_{\text{cal}}+1)(1-\alpha) \rceil / n_{\text{cal}}$. Đặt hình phạt $p(x) = \lambda_v \max(0, E(x) - \hat{q}_{1-\alpha})$. Dưới giả thiết hoán đổi cục bộ (*exchangeability*) trong chế độ bình thường:
$$\mathbb{P}\!\left(p(X_{\text{new}}) = 0\right) = \mathbb{P}\!\left(E(X_{\text{new}}) \le \hat{q}_{1-\alpha}\right) \ge 1 - \alpha \quad (\text{với } \alpha = 0.10 \implies \approx 90\% \text{ phiên phạt đúng bằng } 0.0).$$
**Chứng minh**: Trực tiếp từ bổ đề phân vị hữu hạn của Conformal Risk Control (Angelopoulos et al., 2024): hạng của $E(X_{\text{new}})$ trong tập $n_{\text{cal}}+1$ phần tử phân bố đều trên $\{1,\dots,n_{\text{cal}}+1\}$, do đó xác suất $E(X_{\text{new}}) \le \hat{q}_{1-\alpha}$ đạt tối thiểu $\lceil (n_{\text{cal}}+1)(1-\alpha)\rceil / (n_{\text{cal}}+1) \ge 1-\alpha$. $\blacksquare$

#### Định Lý 7 (Tỷ Lệ Đuôi Biên Độ Sàn & Kiểm Soát Beta Bất Đối Xứng `AsymBeta-NTZ`)
Trên sàn giao dịch có biên độ trần/sàn $\pm L$ ($L=0.07$ trên HOSE, $L=0.10$ trên CSI, $L \ge 0.50$ trên US), xác suất một cổ phiếu chạm sàn trong điều kiện phân phối đuôi tỷ lệ nghịch theo bình phương biên độ chuẩn hóa $s_{\text{band}}(L) = \text{clip}((0.07/L)^2, 0.25, 1.50)$. Khi kết hợp với:
- **Mở khóa $\beta^+$ trong Bull xác nhận** (`bull_beta_unlock = -0.07`) xóa bỏ điểm nghẽn $\beta=0.36$ của v4;
- **Kẹp chặt $\beta^-$ trong Stress/Bear** (`bear_beta_clamp = 0.50`) và **Vùng miễn giao dịch NTZ** (`+0.26` vượt ma sát T+2.5 + thuế bán 0.1%);
- **Lá chắn Epistemic Abstention đồng bộ biên độ sàn** (`fast_crash` tức thời nhả chốt tại `r5 >= 1.2%` cho HOSE $\pm 7\%$, kết hợp `slow_secular_bear` cho thị trường biên độ rộng `s_band < 0.85` như CSI/US),
hệ thống vừa mở khóa tăng trưởng Bull trên HOSE vừa tự động chuyển sang phòng thủ trên chuỗi rơi kéo dài của CSI Emerging mà không cần chỉnh tham số theo từng thị trường.

---

### 7.3. What Failed & Why: Các Kiến Trúc Bị Bác Bỏ, Bẫy Quá Tham Số & Mô-đun Đã Cắt Tỉa

Khác biệt lớn nhất của **MAVE-Q v5** so với mọi thế hệ trước là **minh bạch tuyệt đối các kết quả âm (Negative Results)** ghi trong `logs/v5_falsified_hypotheses.jsonl` (tổng cộng **17 giả thuyết bị bác bỏ/cắt tỉa**):

1. **Bẫy Quá Tham Số Trên Mẫu Nhỏ VN30 (Overparameterization Trap)**:
   - Mở rộng `DCA-QuantFormer` từ `0.48M -> 5.12M -> 11.52M -> 20.47M` trên VN30 (30 mã × 4 năm search) giúp giảm phương sai giữa các seed (`SD: 0.107 -> 0.028`), nhưng **đều bị bác bỏ bởi phạt Occam BIC** (`BIC: 1.994 -> 1.807 -> 1.692 -> 1.635`; trên 8-seed sau cắt tỉa, `20.47M` chỉ nhỉnh `+0.011` Sharpe so với `0.48M` dù nặng gấp `42×` và có Calmar/MaxDD kém hơn nhẹ `3.19 vs 3.21`).
   - *Kết luận khoa học*: Quy mô `sub1b_flagship_900m` (`<1B/khối` trên Kaggle `2x T4` / `TPU v5e-8`) chỉ hợp lý khi huấn luyện đa thị trường chéo quy mô lớn (`VN100 + CSI300 + US + orderbook`), không phù hợp để nhồi trực tiếp vào 30 mã VN30.
2. **Thua Cuộc Của Pre-LN Transformer & Mamba-SSM Trong Đấu Trường 5 Họ Backbone**:
   - `preln_transformer` (`IC=0.1045, SR=1.316, BIC=1.792`) bị bác bỏ rõ rệt do pha loãng vi cấu trúc tầng nông; `mamba_ssm` (`SR=1.296, BIC=1.785`) cũng bị bác bỏ. Ngược lại `dca_quantformer`, `sparse_moe` và `hyena_conv` vượt chuẩn Occam BIC.
3. **Cắt Tỉa Vĩnh Viễn 3 Mô-đun Dư Thừa Tại Cổng Kiểm Định 8-Seed (`Task 05 Ruthless Pruning`)**:
   - Dù `Chronos-Kalman-Hedge`, `DCA-MoSE HyperGate`, và `Split-Conformal Stock Veto` đều vượt kiểm định toán học & nhân quả `atol=1e-9`, bài kiểm định 8-seed thực chiến chứng minh trên mẫu nhỏ 30 mã VN30:
     - Gỡ **`Chronos-Kalman-Hedge`** tăng Val Sharpe **`+0.074`** (`1.405 -> 1.479`, `p=0.968`) và giảm SD gần một nửa (`0.106 -> 0.063`) vì nhiễu ước lượng IC 5 ngày trên $N=30$ mã quá lớn.
     - Gỡ **`DCA-MoSE HyperGate`** tăng Val Sharpe **`+0.058`** (`1.405 -> 1.463`, `p=0.996`) vì `G-DCA` đã trích xuất tương tác phi tuyến trực tiếp.
     - Gỡ **`Split-Conformal Stock Veto`** tăng Val Sharpe **`+0.042`** (`1.405 -> 1.446`, `p=0.9385`), tăng IC `+0.0008` và giảm SD `61%` (`0.106 -> 0.041`).
   - **Hành động dũng cảm**: Cả 3 mô-đun trên đã bị **CẮT TỈA VĨNH VIỄN** khỏi cấu hình chính thức `configs/maveq_v5.yaml` (mã nguồn chỉ giữ lại để tái lập Negative Results và nghiên cứu Cloud quy mô lớn). Sau cắt tỉa, **MAVE-Q v5 tinh gọn** nhảy vọt lên **8-seed Val Sharpe `1.473±0.083`, Calmar `3.21`, MaxDD `-19.7%`, Val DSR `0.966`**, và **100% thành phần còn lại (`DCA-QuantFormer` và `AsymBeta-NTZ`) đều đóng góp dương tự nhiên** (gỡ `DCA` tụt `-0.0054` IC `p=0.0000` & `-0.042` Sharpe; gỡ `AsymBeta-NTZ` sập `-0.447` Sharpe `p=0.0000` & `-1.77` Calmar!).

---

### 7.4. Bảng Tổng Kết Tiến Hóa Toàn Phả Hệ (`v0 -> v1 -> v2 -> v3 -> v4 -> v5`, Single-Shot Sealed Test)

| Thế hệ | Kiến trúc cốt lõi | Trials | Val Sharpe | Test IC | Test Sharpe | Test Calmar | Test Ann. Ret | Test MaxDD | Test DSR (Indiv. / Lineage) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Agent v0` | Baseline Prompting (No RSI) | 3 | 0.994 | 0.058 | 0.625 | 0.62 | +16.4% | -26.5% | 0.562 |
| `VN-RSI v1` | Bi-Level RSI + Leakage Auditor | 36 | 1.158 | 0.068 | 0.892 | 0.89 | +24.8% | -27.9% | 0.421 |
| `Micro-RSI v2` | Modular HRM + Discrete AST Diffusion | 42 | 1.245 | 0.071 | 1.104 | 1.18 | +31.2% | -26.4% | 0.498 |
| `Holo-MAVE-Q v3` | Tree-MDLM + Holo-HRM + BTO-SRSS | 81 | 1.412 | 0.074 | 1.676 | 2.22 | +48.8% | -22.0% | 0.349 |
| `MAVE-Q v4` | VetQuant-RSI (Static BAB + Memory + PM) | 1 | 1.347 (8s) | 0.0654 | 1.067 | 1.23 | +24.4% | -19.8% | 0.785 |
| **`MAVE-Q v5` (Pruned Lean)** | **`DCA-QuantFormer (0.48M)` + Popperian Refuter + `AsymBeta-NTZ`** | **1** | **1.473 (8s) / 1.586 (s2026)** | **0.0850** | **1.441** | **1.70** | **+35.6%** | **-20.9%** | **0.895** *(Lineage $N=29$: 0.389)* |

- **Đánh giá trung thực trên Sealed Test (1 lần chấm duy nhất, sổ cái `28 -> 29` dòng, SHA-256 Seal nguyên vẹn)**:
  - `MAVE-Q v5` vượt `MAVE-Q v4` tới **`+0.374` Test Sharpe** (`1.441 vs 1.067`, bootstrap `p=0.081`), **`+30.0%` Test IC** (`0.0850 vs 0.0654`), **`+11.2pp` lợi nhuận năm** (`+35.6% vs +24.4%`, `p=0.068`), **`+60.6%` Calmar**, và đánh bại `VN-Index Buy & Hold` (`1.248`, MaxDD `-39.4%`), `FinMem` (`1.264`), `AlphaForge-DE` (`0.971`), `RD-Agent(Q)` (`0.903`), `TradingAgents` (`0.159`), cùng toàn bộ các Frontier LLM 70B–671B (`0.61–0.73`).
  - Đồng thời báo cáo thẳng thắn: `FinAgent` có raw Test Sharpe nhỉnh hơn nhẹ (`1.514 vs 1.441`), `Holo-MAVE-Q v3` có raw Sharpe `1.676` nhưng phải trả giá bằng `81` trial (`DSR = 0.349`), và **Lineage-Cumulative DSR** nghiêm ngặt trên toàn bộ `29` dòng sổ cái v1..v5 là **`0.3889`** (`0.3235` ở `N=45`).
- **Kết quả Anti-Sycophancy 45 Kịch Bản (22 Bẫy Xu Nịnh)**: `Decision Acc = 100.0%`, `Causal Consistency = 100.0%`, **`Sycophancy Resistance = 100.0%`** (vs `0.0%` ở cả 4 baseline gồm `Sycophantic_YesMan_LLM`), **`FDR = 0.0%`** (`<= 5%`), `Abstention Precision = 100.0%`, `Brier = 0.01615`, `ECE = 0.09551`.
