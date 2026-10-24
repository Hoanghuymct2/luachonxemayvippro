# BÁO CÁO KIỂM THỬ VÀ XÁC MINH HỆ THỐNG (TEST REPORT)

**Tên đề tài:** Xây dựng hệ thống hỗ trợ ra quyết định gợi ý xe máy theo nhu cầu người dùng bằng Weighted K-Nearest Neighbors và Google Gemini.  
**Ngày thực hiện kiểm thử:** 24/10/2026  
**Môi trường thử nghiệm:** Windows 11, PowerShell, Python 3.12.10

---

## 1. Kết Quả Thực Thi Kiểm Thử Tự Động (Pytest Summary)

Lệnh thực thi:
```bash
python -m pytest -q
```

Kết quả thực tế ghi nhận:
```text
...............................                                          [100%]
31 passed in 2.64s
```

- **Tổng số test cases:** 31
- **Số test đạt (PASSED):** 31 / 31 (100%)
- **Số test thất bại (FAILED):** 0
- **Thời gian thực thi:** 2.64 giây

---

## 2. Chi Tiết Kết Quả Kiểm Thử Theo Module

### 2.1. Kiểm thử Xử lý và Làm sạch Dữ liệu (`tests/test_data.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_parse_price_formats` | Chuẩn hóa chuỗi giá đa dạng (dấu chấm, phẩy, số nguyên, số thực) | **PASS** |
| `test_parse_trunk_formats` | Chuẩn hóa dung tích cốp (chứa 'L', 'l', phẩy thập phân) | **PASS** |
| `test_parse_power_formats` | Chuẩn hóa công suất (chứa 'kW', phẩy thập phân) | **PASS** |
| `test_load_and_clean_real_dataset` | Đọc 225 dòng `dataset.csv`, forward-fill hãng, tiền tố model, không có NaN/Inf | **PASS** |
| `test_clean_data_handles_missing_and_corrupt` | Xử lý an toàn các dòng bị khuyết hoặc lỗi giá trị âm | **PASS** |

### 2.2. Kiểm thử Hồ sơ Trọng số DSS (`tests/test_weights.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_purpose_weights_sum_to_one` | Kiểm tra cả 5 hồ sơ trọng số đều có tổng bằng 1.0 (sai số $< 10^{-9}$) | **PASS** |
| `test_purpose_weights_structure` | Đủ 8 nhóm thuộc tính trong từng hồ sơ, không có trọng số âm | **PASS** |
| `test_purpose_descriptions_coverage` | Có đủ mô tả tiếng Việt chi tiết cho cả 5 hồ sơ | **PASS** |
| `test_per_dimension_weights_calculation` | Phân bổ đều One-Hot ($w_{g,j} = W_g/m$), tổng chiều = 1.0 | **PASS** |
| `test_per_dimension_weights_validation_errors` | Bắt lỗi ngoại lệ khi thiếu nhóm, tổng sai hoặc có số âm | **PASS** |

### 2.3. Kiểm thử Thuật toán Weighted KNN (`tests/test_recommender.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_independent_mathematical_hand_calculation` | Đối chiếu kết quả thuật toán với phép tính giải tích bằng tay trên dữ liệu giả lập | **PASS** |
| `test_hard_budget_filter` | Lọc ngân sách cứng: xe vượt ngân sách tuyệt đối không xuất hiện | **PASS** |
| `test_empty_results_when_budget_too_low` | Ngân sách thấp hơn toàn bộ xe trả về DataFrame rỗng an toàn | **PASS** |
| `test_k_larger_than_candidate_count` | Xử lý khi K lớn hơn số lượng xe thỏa mãn | **PASS** |
| `test_invalid_categorical_query_raises_error` | Bắt lỗi khi nhập giá trị phân loại ngoài danh mục hỗ trợ | **PASS** |
| `test_invalid_k_raises_error` | Bắt lỗi khi K <= 0 | **PASS** |
| `test_criteria_breakdown_messages` | Kiểm tra định dạng thông báo đáp ứng tiêu chí đúng quy chuẩn | **PASS** |

### 2.4. Kiểm thử Tích hợp Gemini & Fallback (`tests/test_gemini_helper.py`)
*Ghi chú: Toàn bộ kiểm thử Gemini sử dụng mock cô lập, không tiêu tốn quota API.*
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_fallback_purpose_keywords` | Nhận diện mục đích sử dụng bằng từ khóa tiếng Việt | **PASS** |
| `test_fallback_budget_parsing` | Bóc tách ngân sách với 'triệu', 'tr', 'củ', đổi sang VND đúng 1 lần | **PASS** |
| `test_fallback_trunk_and_inferred` | Bóc tách dung tích cốp và cụm từ suy luận 'cốp rộng' | **PASS** |
| `test_fallback_body_and_fuel` | Bóc tách kiểu xe và nhiên liệu | **PASS** |
| `test_parse_need_mock_gemini_success` | Gemini trả JSON hợp lệ -> chuyển đổi sang query chuẩn xác | **PASS** |
| `test_parse_need_mock_gemini_corrupt_json` | Gemini trả JSON lỗi -> tự động chuyển sang fallback an toàn | **PASS** |
| `test_parse_need_mock_gemini_exception` | Gemini gặp sự cố mạng -> tự động chuyển sang fallback | **PASS** |
| `test_parse_need_no_api_key` | Không có API key -> chuyển sang fallback trực tiếp | **PASS** |
| `test_explain_fallback_normal_and_empty` | Sinh đoạn văn giải thích kết quả khi không có API | **PASS** |

### 2.5. Kiểm thử Quản lý Lịch sử Tìm kiếm (`tests/test_history.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_save_and_load_history` | Lưu và nạp lịch sử tìm kiếm từ file JSON | **PASS** |
| `test_no_sensitive_keys_saved` | Tuyệt đối không lưu API key hoặc token vào lịch sử | **PASS** |
| `test_prevent_duplicate_rerun_saves` | Chống lưu trùng lặp khi Streamlit rerun với cùng thông số | **PASS** |
| `test_clear_history` | Xóa sạch toàn bộ lịch sử khi người dùng yêu cầu | **PASS** |
| `test_corrupt_history_file_handling` | Xử lý an toàn khi file JSON bị lỗi/hỏng, không làm sập ứng dụng | **PASS** |

---

## 3. Thống Kê Dữ Liệu Thực Tế Từ Dataset (`dataset.csv`)

- **Số dòng dữ liệu gốc:** 225 bản ghi.
- **Số dòng hợp lệ sau làm sạch:** 225 bản ghi (100% dòng đạt chuẩn).
- **Số dòng bị loại:** 0 dòng.
- **Số lượng thương hiệu:** 9 hãng:
  - Honda: 65 mẫu xe
  - Vespa: 31 mẫu xe
  - SYM: 21 mẫu xe
  - Yamaha: 20 mẫu xe
  - Suzuki: 20 mẫu xe
  - VinFast: 20 mẫu xe
  - Yadea: 20 mẫu xe
  - Dat Bike: 20 mẫu xe
  - Piaggio: 8 mẫu xe
- **Phân bố nhiên liệu:**
  - Xăng: 163 mẫu xe
  - Điện: 62 mẫu xe
- **Phân bố kiểu xe:**
  - Tay ga: 90 mẫu xe
  - Xe điện: 62 mẫu xe
  - Xe côn tay: 36 mẫu xe
  - Xe số: 20 mẫu xe
  - Tay ga (50cc): 9 mẫu xe
  - Xe số (50cc): 8 mẫu xe
- **Phân bố năm sản xuất:**
  - 2024: 152 mẫu xe
  - 2023: 52 mẫu xe
  - 2022: 21 mẫu xe
- **Chỉ số định lượng:**
  - Giá thấp nhất: 12.900.000 VNĐ
  - Giá cao nhất: 1.231.000.000 VNĐ (Honda Gold Wing)
  - Giá trung vị (Median): 49.091.000 VNĐ
  - Công suất thấp nhất: 0.5 kW
  - Công suất cao nhất: 140.0 kW
  - Công suất trung vị: 7.0 kW
  - Dung tích cốp nhỏ nhất: 0.0 L
  - Dung tích cốp lớn nhất: 50.0 L
  - Dung tích cốp trung vị: 17.0 L
- **Số chiều không gian đặc trưng sau One-Hot:** 22 chiều (4 số + 18 One-Hot).

---

## 4. Kết Quả Kiểm Chứng Thực Tế 3 Kịch Bản Truy Vấn (Benchmark Queries)

### Kịch bản 1: Sinh viên tìm xe ga ngân sách giới hạn $\le$ 45 triệu
- **Hồ sơ:** Học sinh, Sinh viên (Giá 35%, Kiểu xe 20%, Cốp 15%, Nhiên liệu 15%).
- **Lọc cứng:** Giá $\le$ 45.000.000 VNĐ.
- **Kết quả xếp hạng thực tế:**
  1. **SYM Tuscany 150** — Giá: 45.000.000 đ | Cốp: 27.0L | Độ phù hợp: **76.9%** ($d_w = 0.3010$)
  2. **SYM New Attila 125** — Giá: 33.700.000 đ | Cốp: 27.0L | Độ phù hợp: **76.8%** ($d_w = 0.3015$)
  3. **Honda Air Blade 125 (Bản đặc biệt)** — Giá: 43.582.909 đ | Cốp: 23.0L | Độ phù hợp: **75.7%** ($d_w = 0.3210$)
  4. **Honda Air Blade 125 (Bản tiêu chuẩn)** — Giá: 42.404.727 đ | Cốp: 23.0L | Độ phù hợp: **75.7%** ($d_w = 0.3210$)
  5. **Yamaha FreeGo S (Bản đặc biệt)** — Giá: 34.069.000 đ | Cốp: 25.0L | Độ phù hợp: **75.7%** ($d_w = 0.3215$)
- **Xác nhận lọc ngân sách:** 100% kết quả đều có giá $\le$ 45.000.000 VNĐ.

### Kịch bản 2: Người đi làm cần cốp rộng (28L), ngân sách $\le$ 70 triệu
- **Hồ sơ:** Đi làm, đi phố & Gia đình (Cốp 25%, Giá 20%, Kiểu xe 20%, Nhiên liệu 15%).
- **Lọc cứng:** Giá $\le$ 70.000.000 VNĐ.
- **Kết quả xếp hạng thực tế:**
  1. **SYM Tuscany 150** — Giá: 45.000.000 đ | Cốp: 27.0L | Độ phù hợp: **76.3%** ($d_w = 0.3100$)
  2. **SYM New Attila 125** — Giá: 33.700.000 đ | Cốp: 27.0L | Độ phù hợp: **76.3%** ($d_w = 0.3105$)
  3. **Suzuki Burgman Street 125** — Giá: 48.600.000 đ | Cốp: 22.5L | Độ phù hợp: **76.0%** ($d_w = 0.3157$)
  4. **SYM Naga 150** — Giá: 49.500.000 đ | Cốp: 28.0L | Độ phù hợp: **75.3%** ($d_w = 0.3283$)
  5. **Piaggio Liberty One 125** — Giá: 48.600.000 đ | Cốp: 17.0L | Độ phù hợp: **75.2%** ($d_w = 0.3292$)

### Kịch bản 3: Người quan tâm xe điện hoặc công suất cao (10 kW)
- **Hồ sơ:** Đi phượt, Thể thao & Đam mê tốc độ (Công suất 35%, Kiểu xe 25%, Giá 10%, Nhiên liệu 10%).
- **Kết quả xếp hạng thực tế:**
  1. **Vespa Elettrica** — Nhiên liệu: Điện | Công suất: 4.0 kW | Giá: 279.000.000 đ | Độ phù hợp: **80.0%** ($d_w = 0.2497$)
  2. **Yadea Fierce** — Nhiên liệu: Điện | Công suất: 10.0 kW | Giá: 55.000.000 đ | Độ phù hợp: **78.6%** ($d_w = 0.2728$)
  3. **Yamaha NEO'S (Xe máy điện)** — Nhiên liệu: Điện | Công suất: 2.3 kW | Giá: 49.091.000 đ | Độ phù hợp: **78.4%** ($d_w = 0.2758$)
  4. **Yadea BuyE** — Nhiên liệu: Điện | Công suất: 1.2 kW | Giá: 21.990.000 đ | Độ phù hợp: **78.1%** ($d_w = 0.2809$)
  5. **Yadea U-Like** — Nhiên liệu: Điện | Công suất: 1.2 kW | Giá: 17.990.000 đ | Độ phù hợp: **78.1%** ($d_w = 0.2810$)

---

## 5. Hiện Trạng Tích Hợp Google Gemini API

- **Model mặc định dự kiến:** `gemini-2.5-flash`.
- **Kết quả khảo sát API thực tế:** Khi gửi yêu cầu tới `gemini-2.5-flash`, Google API trả về mã lỗi:
  `404 NOT_FOUND: This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash or gemini-3.1-flash-lite.`
- **Cơ chế xử lý thông minh:** Hệ thống đã được thiết kế cơ chế tự động bắt mã 404 và chuyển tiếp mượt mà sang `gemini-3.1-flash-lite`. Kiểm tra thực tế xác nhận kết nối trực tiếp thành công với API key trong `keyapi.txt` và trả về kết quả trong vòng 2 - 3 giây.
- **An toàn tuyệt đối:** Khi không có mạng hoặc không có API key, bộ fallback regex đảm bảo 100% tính năng phân tích nhu cầu và giải thích kết quả vẫn hoạt động bình thường, không bao giờ gây sập ứng dụng.

---

## 6. Phiên Bản Thư Viện Sử Dụng

- Python: `3.12.10`
- pandas: `3.1.0`
- numpy: `2.5.4`
- scikit-learn: `1.9.1`
- streamlit: `1.66.1`
- google-genai: `2.33.0`
- pytest: `9.1.1`

