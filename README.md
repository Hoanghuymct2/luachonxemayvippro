# Hệ Thống Hỗ Trợ Ra Quyết Định (DSS) Gợi Ý Xe Máy

> **Đề tài:** Xây dựng hệ thống hỗ trợ ra quyết định gợi ý xe máy theo nhu cầu người dùng bằng Weighted K-Nearest Neighbors và Google Gemini.  
> **Môn học:** Hệ trợ giúp quyết định (Decision Support Systems - DSS)

---

## 📌 1. Giới Thiệu Dự Án

Hệ thống được thiết kế để giải quyết bài toán hỗ trợ người tiêu dùng lựa chọn mẫu xe máy tối ưu nhất giữa hàng trăm lựa chọn trên thị trường Việt Nam. Hệ thống kết hợp chặt chẽ giữa hai thành phần:

1. **Thuật toán Định lượng Weighted K-Nearest Neighbors (KNN):**
   - Đảm nhận toàn bộ khâu tính toán, lọc điều kiện, chuẩn hóa dữ liệu, tính khoảng cách toán học và xếp hạng xe minh bạch, khách quan.
   - Áp dụng **Lọc trần ngân sách cứng (Hard Budget Ceiling)**: Loại bỏ các xe vượt ngân sách; không phạt khoảng cách đối với các xe rẻ hơn ngân sách tối đa.
   - Chuẩn hóa toàn bộ không gian 22 chiều đặc trưng (4 thuộc tính số bằng **MinMaxScaler** + 18 chiều thuộc tính phân loại bằng **One-Hot Encoding**) về cùng miền $[0, 1]$.
   - Áp dụng **Ma trận trọng số theo 5 hồ sơ mục đích sử dụng** với cơ chế chia đều trọng số cho các chiều phân loại, bảo đảm $\sum w_i = 1.00$.
   - **Quy tắc phá vỡ hòa điểm xác định (Deterministic Tie-breaking)**: Đảm bảo kết quả hoàn toàn lặp lại được (reproducible).

2. **Trí Tuệ Nhân Tạo Google Gemini AI:**
   - **Hiểu ngôn ngữ tự nhiên & Dự đoán thông số theo ngữ cảnh:** Tiếp nhận mô tả đời thường của người dùng, phân tích ngữ cảnh để dự đoán các thông số kỹ thuật thực tế (ngân sách, kiểu xe, hãng, cốp, công suất) thay vì trả về giá trị rỗng/None.
   - **Tự động đồng bộ giao diện:** Điền tức thì các thông số dự đoán vào **Hồ Sơ Mục Đích (Sidebar)** và **Form Tinh Chỉnh Tiêu Chí**.
   - **Tư vấn & Giải thích chuyên sâu:** Đóng vai trò chuyên gia phân tích bảng số liệu định lượng do Weighted KNN tính toán để giải thích cặn kẽ vì sao mẫu xe Top 1 được lựa chọn và đưa ra lời khuyên trải nghiệm thực tế.
   - **Cơ chế Fallback thông minh:** Tự động chuyển đổi giữa các phiên bản model Gemini (`gemini-2.5-flash`, `gemini-3.1-flash-lite`) và có bộ quy tắc Regex dự phòng khi không có Internet hoặc không có API key.

> [!IMPORTANT]
> **Nguyên tắc cốt lõi:** Google Gemini AI **tuyệt đối không tự bịa danh sách xe** hay thay thế thuật toán. Toàn bộ thứ tự xếp hạng, khoảng cách, điểm số và thông số đều xuất phát từ phép tính định lượng trên dữ liệu thực `dataset.csv`.

---

## 📂 2. Cấu Trúc Dự Án Tinh Gọn

Dự án được tổ chức gọn gàng ngay tại thư mục gốc, loại bỏ toàn bộ các thư mục con phân mảnh dư thừa:

```text
D:\TT/
├── dataset.csv            # Bộ dữ liệu thực tế 225 mẫu xe máy thuộc 9 hãng tại Việt Nam
├── keyapi.txt             # Lưu khóa Google Gemini API Key an toàn tại máy cục bộ
├── app.py                 # Giao diện ứng dụng Streamlit DSS trực quan, đầy đủ tính năng
├── recommender.py         # Lõi thuật toán Weighted KNN, tiền xử lý, chuẩn hóa, lọc ngân sách
├── gemini_helper.py       # Tích hợp Google Gemini AI, suy luận ngữ cảnh & NLP fallback
├── history.py             # Quản lý lưu trữ và hiển thị lịch sử tư vấn
├── history.json           # File lưu dữ liệu lịch sử tìm kiếm gần đây
├── api.py                 # Backend REST API chuẩn FastAPI (phục vụ mở rộng tích hợp)
├── requirements.txt       # Danh mục các thư viện phụ thuộc
├── README.md              # Tài liệu hướng dẫn sử dụng và báo cáo học thuật
├── run.bat                # Phím tắt khởi động ứng dụng nhanh 1-Click
├── .gitignore             # Danh sách loại trừ Git (bảo vệ key và file tạm)
└── .streamlit/
    └── config.toml        # Cấu hình tự động mở trình duyệt cho Streamlit
```

---

## 🚀 3. Hướng Dẫn Vận Hành

### ⚡ Cách 1: Khởi động 1-Click (Khuyên dùng trên Windows)
- **Kích đúp chuột vào file `run.bat`**: Hệ thống sẽ tự động khởi chạy và mở ngay giao diện Web trên trình duyệt tại địa chỉ:
  ```text
  http://localhost:8501
  ```

---

### 💻 Cách 2: Chạy Bằng Dòng Lệnh (Terminal / PowerShell)

Mở PowerShell tại thư mục `D:\TT`:

1. **Cài đặt thư viện:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Khởi chạy ứng dụng Streamlit:**
   ```powershell
   python -m streamlit run app.py
   ```
   *Giao diện Web sẽ tự động mở tại `http://localhost:8501`.*

3. **(Tùy chọn) Khởi chạy Backend REST API:**
   Nếu bạn muốn kiểm thử hoặc tích hợp API với các ứng dụng khác:
   ```powershell
   python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Tài liệu Swagger UI tương tác trực tiếp tại: `http://127.0.0.1:8000/docs`.*

---

## 📐 4. Cơ Sở Lý Thuyết & Mô Hình Toán Học

### 4.1. Bản chất DSS vs. Machine Learning Có Giám sát
- **Không phải bài toán Phân loại (Classification) hay Dự báo (Regression):** Dataset xe máy thực tế không có nhãn kết quả (ví dụ "người dùng A đã mua thành công xe B"). Do đó, không tồn tại tập nhãn ground-truth để huấn luyện mô hình có giám sát và không sử dụng các chỉ số như Accuracy, Precision hay F1.
- **Bản chất là bài toán Đánh giá Đa tiêu chí (Multi-Criteria Decision Making - MCDM):** Hệ thống có nhiệm vụ xếp hạng (Ranking) các phương án xe khả thi dựa trên khoảng cách giữa hồ sơ thuộc tính của từng phương án so với vector mục tiêu (Ideal Solution) do người dùng đặt ra.

---

### 4.2. Không Gian Đặc Trưng 22 Chiều

1. **4 Thuộc tính số (Continuous features):**
   - Giá xe (`Giá(VND)`): 12.900.000 VNĐ - 1.231.000.000 VNĐ.
   - Công suất động cơ (`Công suất tối đa (Kw)`): 0.5 kW - 140.0 kW.
   - Dung tích cốp (`Dung tích cốp (L)`): 0.0 L - 50.0 L.
   - Năm sản xuất (`Năm sản xuất`): 2022 - 2024.

2. **18 Chiều One-Hot Encoding từ 4 thuộc tính phân loại (Categorical features):**
   - Hãng sản xuất (9 hãng): `Honda`, `Yamaha`, `Suzuki`, `SYM`, `Piaggio`, `Vespa`, `VinFast`, `Yadea`, `Dat Bike`.
   - Loại nhiên liệu (2 loại): `Xăng`, `Điện`.
   - Kiểu dáng xe (6 kiểu): `Tay ga`, `Xe điện`, `Xe côn tay`, `Xe số`, `Tay ga (50cc)`, `Xe số (50cc)`.
   - Tình trạng xe (1 loại): `Mới`.

$\implies$ Tổng số chiều không gian đặc trưng: $4 + (9 + 2 + 6 + 1) = 22\text{ chiều}$.

---

### 4.3. Công Thức Chuẩn Hóa Đặc Trưng Số (MinMaxScaler)

Với thuộc tính số $x$, giá trị chuẩn hóa $x'$ được tính theo công thức:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \quad (x' \in [0, 1])$$

*Xử lý trường hợp biên:* Nếu $x_{\max} = x_{\min}$ (khoảng biến thiên bằng 0), hệ thống tự động gán $x' = 0.5$ để tránh lỗi chia cho 0.

---

### 4.4. Cơ Chế Phân Bổ Trọng Số Đều Cho Chiều One-Hot

Để đảm bảo các nhóm phân loại có nhiều giá trị không bị áp đảo nhóm có ít giá trị, trọng số của mỗi chiều One-Hot $j$ thuộc nhóm $g$ được chia đều:

$$w_{g, j} = \frac{W_g}{m_g}$$

Trong đó $W_g$ là trọng số nhóm theo hồ sơ người dùng, $m_g$ là số chiều One-Hot của nhóm đó.  
Công thức này bảo đảm tổng trọng số của toàn bộ 22 chiều đặc trưng luôn bằng chính xác $1.00$:

$$\sum_{i=1}^{22} w_i = \sum_{k=1}^{4} W_{\text{num}, k} + \sum_{g \in \text{Cat}} \sum_{j=1}^{m_g} \frac{W_g}{m_g} = 1.00$$

---

### 4.5. Ma Trận 5 Hồ Sơ Trọng Số DSS

| Thuộc tính | Đi làm & Gia đình | Học sinh, Sinh viên | Dịch vụ (Grab/Be) | Phượt & Thể thao | Thời trang & Cao cấp |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Giá(VND)** | 20% | **35%** | **30%** | 10% | 10% |
| **Dung tích cốp (L)** | **25%** | 15% | 10% | 5% | 15% |
| **Kiểu xe** | 20% | 20% | 15% | **25%** | **25%** |
| **Nhiên liệu** | 15% | 15% | 20% | 10% | 5% |
| **Hãng** | 10% | 5% | 5% | 10% | **25%** |
| **Công suất (Kw)** | 5% | 5% | 15% | **35%** | 10% |
| **Năm sản xuất** | 5% | 5% | 5% | 5% | 10% |
| **Tình trạng** | 0% | 0% | 0% | 0% | 0% |
| **Tổng cộng** | **100%** | **100%** | **100%** | **100%** | **100%** |

---

### 4.6. Lọc Trần Ngân Sách Cứng (Hard Budget Ceiling)

Khác với các thuộc tính sở thích, ngân sách là một **ràng buộc biên khả thi (Feasibility Constraint)**:

$$D_{\text{filtered}} = \{x \in D \mid \text{Price}(x) \le B_{\max}\}$$

- Mọi xe có giá lớn hơn $B_{\max}$ sẽ bị loại bỏ khỏi danh sách đánh giá.
- Đối với các xe thỏa mãn ($\text{Price}(x) \le B_{\max}$), khoảng cách thành phần giá không bị phạt nếu người dùng chỉ đặt mức ngân sách trần.

---

### 4.7. Khoảng Cách Weighted Euclidean Distance ($d_w$)

Khoảng cách Euclid có trọng số giữa phương án xe $p$ và vector nhu cầu $q$ trên 22 chiều đặc trưng:

$$d_w(p, q) = \sqrt{\sum_{i=1}^{22} w_i \cdot (p_i - q_i)^2}$$

---

### 4.8. Quy Tắc Phá Vỡ Hòa Điểm Xác Định (Deterministic Tie-breaking)

Khi hai xe có khoảng cách $d_w$ trùng nhau trong giới hạn sai số $|d_w(a) - d_w(b)| < 10^{-6}$, thứ tự ưu tiên được giải quyết xác định:
1. **Năm sản xuất mới hơn:** $\text{Year}(a) > \text{Year}(b)$
2. **Giá bán thấp hơn (tiết kiệm hơn):** $\text{Price}(a) < \text{Price}(b)$
3. **Thứ tự bảng chữ cái theo tên xe:** $\text{Name}(a) < \text{Name}(b)$

---

### 4.9. Điểm Phù Hợp Tương Đối (Relative Fit Score)

Điểm phù hợp tương đối biểu thị mức độ gần gũi trực quan cho người ra quyết định:

$$\text{Score} = \operatorname{round}\left(\frac{1}{1 + d_w} \times 100, 1\right) \quad (\text{Score} \in (0, 100]\%)$$

---

## 🤖 5. Cơ Chế Dự Đoán Ngữ Cảnh của Gemini AI

Khi người dùng mô tả nhu cầu bằng lời văn tự nhiên chung chung (ví dụ: *"tôi là sinh viên"*, *"mua xe cho nữ đi chợ"*, *"chạy xe ôm công nghệ"*), AI sẽ không để giá trị rỗng/None mà dựa vào tri thức thị trường thực tế để dự đoán trọn bộ thông số:

| Ngữ cảnh người dùng | Hồ sơ nhu cầu | Ngân sách dự đoán | Kiểu xe | Hãng xe | Dung tích cốp | Công suất |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| *"tôi là sinh viên"* | Học sinh, Sinh viên | **25 triệu** | Xe số | Honda | 10 L | 6.5 kW |
| *"nữ đi chợ, chở con"* | Đi làm & Gia đình | **42 triệu** | Tay ga | Honda | 28 L | 7.5 kW |
| *"chạy xe ôm, grab"* | Chạy dịch vụ | **25 triệu** | Xe số | Honda | 5 L | 6.5 kW |
| *"đi phượt, thể thao"* | Phượt & Tốc độ | **55 triệu** | Xe côn tay | Yamaha/Honda | 0 L | 11.5 kW |
| *"thời trang, sang trọng"* | Thời trang & Cao cấp | **85 triệu** | Tay ga | Vespa/Honda | 20 L | 9.5 kW |
| *"xe điện bảo vệ môi trường"*| Tùy chọn nhiên liệu | **35 triệu** | Xe điện | VinFast | 18 L | 3.0 kW |

Sau khi phân tích, hệ thống tự động điền toàn bộ các thông số này vào thanh điều hướng bên trái và form tìm kiếm, người dùng có thể bấm ngay nút gợi ý hoặc tùy chỉnh lại nếu muốn.

---

## 🔒 6. Chính Sách Bảo Mật Khóa API

- Khóa API Google Gemini được lưu trong file `keyapi.txt` tại thư mục dự án và được nạp an toàn vào bộ nhớ khi ứng dụng khởi chạy.
- Hệ thống tuyệt đối không in API key ra giao diện, không xuất ra terminal/console, không ghi vào file lịch sử `history.json` và file `keyapi.txt` đã được đưa vào `.gitignore`.
