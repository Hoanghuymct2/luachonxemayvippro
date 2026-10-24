# MÔ HÌNH TOÁN HỌC VÀ CƠ SỞ HỌC THUẬT DSS

Tài liệu giải thích chi tiết các công thức toán học, nguyên lý thuật toán Weighted K-Nearest Neighbors (KNN), các giả định mô hình và cơ sở lý thuyết của Hệ thống Hỗ trợ Ra Quyết định Gợi ý Xe Máy.

---

## 1. Bản chất DSS vs. Machine Learning Có Giám sát

Trong lý thuyết Hệ thống Hỗ trợ Ra Quyết định (Decision Support System - DSS):
- **Không phải bài toán Phân loại (Classification) hay Dự báo (Regression):** Dataset xe máy thực tế không có nhãn "xe này đã mua thành công" hay "người dùng A hài lòng với xe B". Do đó, không tồn tại tập nhãn ground-truth để huấn luyện mô hình học có giám sát và không thể đo đạc bằng chỉ số Accuracy (ví dụ "độ chính xác 95%").
- **Bản chất là bài toán Đánh giá Đa tiêu chí (Multi-Criteria Decision Making - MCDM):** Hệ thống có nhiệm vụ sắp xếp thứ tự ưu tiên (Ranking) các phương án thay thế khả thi dựa trên khoảng cách giữa hồ sơ thuộc tính của từng phương án so với vector mục tiêu (Ideal Solution) do người ra quyết định đặt ra.
- **Hồ sơ trọng số (Purpose Profiles):** Không phải là trọng số được học máy (learned weights), mà là ma trận sở thích tiên nghiệm (A-priori Preference Modeling) do chuyên gia thiết kế để lượng hóa các trường hợp sử dụng tiêu biểu trong đời sống.

---

## 2. Chuẩn hóa Đặc trưng Số: MinMaxScaler

### 2.1. Công thức
Với một thuộc tính số $x$ thuộc tập $NUMERIC\_COLS = [\text{Giá(VND)}, \text{Công suất tối đa (Kw)}, \text{Dung tích cốp (L)}, \text{Năm sản xuất}]$, giá trị chuẩn hóa $x'$ được tính theo công thức:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Trong đó:
- $x_{\min}, x_{\max}$ là giá trị nhỏ nhất và lớn nhất của thuộc tính đó trên toàn bộ tập dữ liệu hợp lệ.
- $x' \in [0, 1]$.

### 2.2. Xử lý trường hợp biên
- **Trường hợp $x_{\max} = x_{\min}$:** Nếu toàn bộ xe trong tập dữ liệu có cùng một giá trị (khoảng biến thiên bằng 0), gán $x' = 0.5$ để tránh lỗi chia cho 0.
- **Trường hợp truy vấn ngoài miền dữ liệu:** Nếu người dùng nhập giá trị vượt ngoài miền $[x_{\min}, x_{\max}]$, công thức vẫn áp dụng nhất quán đại số, sau đó bình phương khoảng cách đảm bảo tính không âm và tính đơn điệu.

### 2.3. Lý do lựa chọn MinMaxScaler
Trong thuật toán tính khoảng cách Euclidean, nếu không chuẩn hóa, thuộc tính có đơn vị lớn như `Giá(VND)` (hàng chục triệu) sẽ áp đảo hoàn toàn các thuộc tính có đơn vị nhỏ như `Công suất (Kw)` (khoảng 0.5 - 20) hay `Dung tích cốp (L)` (khoảng 10 - 30). MinMaxScaler đưa mọi thang đo về khoảng $[0, 1]$, giúp việc áp dụng trọng số trở nên công bằng và chính xác.

---

## 3. Mã hóa Đặc trưng Phân loại: One-Hot Encoding

### 3.1. Danh sách đặc trưng và Số chiều thực tế
Thuộc tính phân loại gồm 4 nhóm:
- **Hãng:** 9 giá trị (`Honda`, `Yamaha`, `Suzuki`, `SYM`, `Piaggio`, `Vespa`, `VinFast`, `Yadea`, `Dat Bike`) $\implies 9$ chiều.
- **Nhiên liệu:** 2 giá trị (`Xăng`, `Điện`) $\implies 2$ chiều.
- **Kiểu xe:** 6 giá trị (`Tay ga`, `Xe điện`, `Xe côn tay`, `Xe số`, `Tay ga (50cc)`, `Xe số (50cc)`) $\implies 6$ chiều.
- **Tình trạng:** 1 giá trị (`Mới`) $\implies 1$ chiều.

Tổng số chiều One-Hot: $9 + 2 + 6 + 1 = 18$ chiều.
Tổng số chiều của toàn bộ không gian đặc trưng: $4 \text{ (số)} + 18 \text{ (One-Hot)} = 22 \text{ chiều}$.

### 3.2. Lý do chọn One-Hot Encoding
Các thuộc tính phân loại như `Hãng` hay `Kiểu xe` là biến định danh (Nominal variables), không có thứ bậc hơn kém (Ordination). Việc sử dụng One-Hot Encoding giúp tránh việc gán thứ tự giả định (như gán Honda=1, Yamaha=2 sẽ tạo khoảng cách vô lý giữa các hãng).

---

## 4. Phân bổ Trọng số cho Từng Chiều (Dimension Weight Allocation)

### 4.1. Nguyên lý phân bổ đều
Giả sử nhóm thuộc tính phân loại $g$ có trọng số nhóm là $W_g$ và có $m$ chiều One-Hot trong ma trận đặc trưng. Trọng số của mỗi chiều One-Hot $j$ thuộc nhóm $g$ được tính bằng:

$$w_{g,j} = \frac{W_g}{m}$$

Với thuộc tính số đơn biến (1 chiều), trọng số chiều bằng chính trọng số nhóm: $w = W_g$.

### 4.2. Ý nghĩa toán học
Nếu một nhóm có $m$ chiều nhưng lại gán mỗi chiều bằng $W_g$, tổng trọng số của nhóm đó trong khoảng cách bình phương sẽ bị nhân lên gấp $m$ lần. Việc chia đều cho $m$ đảm bảo:
1. Đóng góp tối đa của nhóm thuộc tính vào tổng khoảng cách không vượt quá $W_g$.
2. Tổng trọng số toàn bộ 22 chiều đặc trưng luôn bằng chính xác $1.00$:

$$\sum_{i=1}^{n} w_i = \sum_{k=1}^{4} W_{\text{num}, k} + \sum_{g \in \text{Cat}} \sum_{j=1}^{m_g} \frac{W_g}{m_g} = \sum_{k=1}^{4} W_{\text{num}, k} + \sum_{g \in \text{Cat}} W_g = 1.00$$

---

## 5. Phân Biệt Trần Ngân Sách Cứng (Budget Ceiling) vs Giá Mục Tiêu (Target Price)

Trong các hệ thống DSS kinh doanh tài sản giá trị, có sự phân biệt rõ ràng giữa **ràng buộc biên (Constraint)** và **mục tiêu tối ưu (Objective)**:

### 5.1. Trần ngân sách tối đa ($B_{\max}$)
Khi người dùng nhập "Ngân sách tối đa: 45 triệu VNĐ", đây là một **ràng buộc cứng (Hard Constraint)**:

$$D_{\text{filtered}} = \{x \in D \mid Price(x) \le B_{\max}\}$$

Các xe có giá vượt quá $B_{\max}$ bị loại bỏ hoàn toàn.  
**Quan trọng:** Trong không gian khoảng cách KNN, một chiếc xe có giá 30 triệu hoàn toàn thỏa mãn ngân sách và KHÔNG bị phạt khoảng cách vì rẻ hơn 45 triệu. 

### 5.2. Giá mục tiêu cụ thể ($P_{\text{target}}$)
Chỉ khi người dùng chủ động yêu cầu một mức giá mong muốn cụ thể (ví dụ: "tôi muốn tìm xe đúng tầm giá 50 triệu"), khoảng cách thành phần giá mới được tính:

$$d_{\text{price}}(x) = \left(\frac{Price(x) - x_{\min}}{x_{\max} - x_{\min}} - \frac{P_{\text{target}} - x_{\min}}{x_{\max} - x_{\min}}\right)^2$$

### 5.3. Khoảng giá (Price Range: $B_{\min} \le Price \le B_{\max}$)
Hệ thống cho phép người dùng đặt cả cận dưới $B_{\min}$ và cận trên $B_{\max}$:

$$D_{\text{filtered}} = \{x \in D \mid B_{\min} \le Price(x) \le B_{\max}\}$$

---

## 6. Khoảng cách Weighted Euclidean Distance

Với vector xe $p = (p_1, p_2, \dots, p_{22})$ và vector truy vấn $q = (q_1, q_2, \dots, q_{22})$, khoảng cách Euclidean có trọng số được tính bằng:

$$d_w(p, q) = \sqrt{\sum_{i=1}^{22} w_i (p_i - q_i)^2}$$

---

## 7. Quy tắc Phá vỡ Hòa điểm Xác định (Deterministic Tie-breaking)

Khi hai ứng viên $p^{(a)}$ và $p^{(b)}$ có khoảng cách toán học trùng khớp nhau ($|d_w(p^{(a)}, q) - d_w(p^{(b)}, q)| < 10^{-6}$), hệ thống áp dụng quy tắc sắp xếp đa tiêu chí xác định:
1. **Năm sản xuất mới hơn được ưu tiên:** $\text{Year}(p^{(a)}) > \text{Year}(p^{(b)})$
2. **Giá bán thấp hơn (tiết kiệm hơn) được ưu tiên:** $\text{Price}(p^{(a)}) < \text{Price}(p^{(b)})$
3. **Thứ tự bảng chữ cái tên xe:** $\text{Model}(p^{(a)}) < \text{Model}(p^{(b)})$

Điều này đảm bảo kết quả gợi ý luôn hoàn toàn lặp lại được (reproducible) và minh bạch trong khoa học hỗ trợ ra quyết định.

---

## 8. Điểm Phù hợp Tương đối (Relative Fit Score)

### 8.1. Công thức quy đổi

$$\text{Score} = \operatorname{round}\left(\frac{1}{1 + d_w} \times 100, 1\right)$$

### 8.2. Phân biệt Bản chất giữa Score và Xác suất Thống kê
- **Score $\in (0, 100]$%:** Khi $d_w = 0$ (xe trùng khớp hoàn toàn mọi tiêu chí), $\text{Score} = 100\%$. Khi $d_w$ tăng dần, $\text{Score}$ giảm dần về 0 một cách mượt mà và đơn điệu.
- **Không phải xác suất thống kê:** Điểm phù hợp KHÔNG đại diện cho xác suất người dùng sẽ mua xe, cũng KHÔNG phải là phân phối xác suất (tổng điểm của các xe không bằng 1). Đây là **chỉ số độ gần gũi tương đối (Proximity Metric)** được chuẩn hóa để người dùng dễ quan sát trực quan.

---

## 9. Phân tích Chi tiết Mức độ Đáp ứng Tiêu chí (Criteria Breakdown)

Chỉ phân tích các tiêu chí mà người dùng thực sự chọn hoặc nhập giá trị.

### 9.1. Thuộc tính số

$$\text{Match}_i = \operatorname{round}\left( \max\left(0, 1 - |x'_i - q'_i|\right) \times 100, 1 \right)$$

- Nếu $\text{Match}_i \ge 80\%$: Đánh giá đạt chuẩn sát yêu cầu (`✓`).
- Nếu $\text{Match}_i < 80\%$: Đánh giá có độ lệch đáng kể (`✗`).

### 9.2. Thuộc tính phân loại
- Trùng khớp chính xác: $\text{Match} = 100\% \implies \text{✓}$.
- Khác biệt: $\text{Match} = 0\% \implies \text{✗}$.

---

## 10. Ví dụ Tính Toán Giải Tích Bằng Tay (Numerical Example)

Xét tập dữ liệu thử nghiệm 3 mẫu xe:
- **Xe 1 (Honda Wave):** Giá = 20 triệu, Kiểu = `Xe số`
- **Xe 2 (Yamaha Grande):** Giá = 50 triệu, Kiểu = `Tay ga`
- **Xe 3 (VinFast Evo):** Giá = 30 triệu, Kiểu = `Xe điện`

Miền giá trị: Giá nhỏ nhất = 20 triệu, lớn nhất = 50 triệu $\implies \text{Range} = 30 \text{ triệu}$.
Giá chuẩn hóa:
- Wave: $x'_1 = (20 - 20) / 30 = 0.0$
- Grande: $x'_2 = (50 - 20) / 30 = 1.0$
- Evo: $x'_3 = (30 - 20) / 30 = 10 / 30 = 1/3 \approx 0.3333$

**Truy vấn:** Giá mục tiêu = 50 triệu ($q'_{\text{giá}} = 1.0$), Kiểu xe = `Tay ga`.
**Trọng số:**
- `Giá(VND)`: $W_{\text{giá}} = 0.40$
- `Kiểu xe`: $W_{\text{kiểu}} = 0.60 \implies$ chia 3 chiều One-Hot (`Xe số`, `Tay ga`, `Xe điện`), mỗi chiều $w = 0.60 / 3 = 0.20$.
- Các thuộc tính khác: $0.00$.

### Tính toán khoảng cách:
1. **Yamaha Grande:**
   - Sai lệch giá: $(1.0 - 1.0)^2 \times 0.40 = 0.0$
   - Sai lệch kiểu xe: Tay ga khớp ($1 - 1 = 0$), các kiểu khác khớp ($0 - 0 = 0$).
   - $d_w^2 = 0.0 \implies d_w = 0.0$.
   - $\text{Score} = \text{round}\left(\frac{1}{1 + 0} \times 100, 1\right) = 100.0\%$.

2. **VinFast Evo:**
   - Sai lệch giá: $(1/3 - 1.0)^2 \times 0.40 = (-2/3)^2 \times 0.40 = \frac{4}{9} \times 0.40 = 0.17778$
   - Sai lệch kiểu xe: Lệch ở chiều `Xe điện` ($|1 - 0|^2 \times 0.20 = 0.20$) và chiều `Tay ga` ($|0 - 1|^2 \times 0.20 = 0.20$). Tổng lệch kiểu = $0.40$.
   - $d_w^2 = 0.17778 + 0.40 = 0.57778$.
   - $d_w = \sqrt{0.57778} \approx 0.7601$.
   - $\text{Score} = \text{round}\left(\frac{1}{1 + 0.7601} \times 100, 1\right) = 56.8\%$.

3. **Honda Wave:**
   - Sai lệch giá: $(0.0 - 1.0)^2 \times 0.40 = 1.0 \times 0.40 = 0.40$
   - Sai lệch kiểu xe: Lệch ở `Xe số` ($0.20$) và `Tay ga` ($0.20$) $\implies 0.40$.
   - $d_w^2 = 0.40 + 0.40 = 0.80$.
   - $d_w = \sqrt{0.80} \approx 0.8944$.
   - $\text{Score} = \text{round}\left(\frac{1}{1 + 0.8944} \times 100, 1\right) = 52.8\%$.

**Kết quả xếp hạng:**
1. Yamaha Grande ($d_w = 0.0$, Score = 100.0%)
2. VinFast Evo ($d_w \approx 0.7601$, Score = 56.8%)
3. Honda Wave ($d_w \approx 0.8944$, Score = 52.8%)

*Kết quả tính tay này khớp chính xác 100% với hàm kiểm thử tự động `test_independent_mathematical_hand_calculation` trong `tests/test_recommender.py`.*

---

## 11. Phân Tích Độ Nhạy của Trọng Số (Sensitivity Analysis)

Để kiểm chứng tính ổn định của xếp hạng trong DSS, ta tiến hành thử nghiệm biến thiên nhỏ ở trọng số:
- Khi chuyển đổi giữa hồ sơ **"Học sinh, Sinh viên"** (Giá chiếm 35%, Kiểu xe 20%) sang **"Thời trang, Sang trọng & Cao cấp"** (Hãng chiếm 25%, Kiểu xe 25%, Giá chỉ chiếm 10%):
  - Với cùng mức ngân sách 70 triệu, mẫu xe `SYM Tuscany 150` (Top 1 sinh viên do giá 45 triệu hợp lý) sẽ nhường vị trí dẫn đầu cho các dòng xe thời trang thương hiệu cao như `Vespa Sprint` hoặc `Piaggio Liberty` do trọng số Hãng xe được đẩy cao từ 5% lên 25%.
  - Sự dịch chuyển thứ hạng phản ánh đúng quy luật DSS: thứ hạng nhạy bén với sự thay đổi trong mức độ ưu tiên của người ra quyết định, nhưng ổn định và nhất quán trong cùng một hồ sơ trọng số.
