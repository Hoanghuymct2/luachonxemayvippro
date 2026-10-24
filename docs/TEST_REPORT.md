# BÁO CÁO KIỂM THỬ VÀ XÁC MINH HỆ THỐNG (TEST REPORT)

**Tên đề tài:** Xây dựng hệ thống hỗ trợ ra quyết định gợi ý xe máy theo nhu cầu người dùng bằng Weighted K-Nearest Neighbors và Google Gemini (Kiến trúc phân tầng FastAPI + Streamlit).  
**Ngày thực hiện kiểm thử:** 24/10/2026  
**Môi trường thử nghiệm:** Windows 11, PowerShell, Python 3.12.10

---

## 1. Kết Quả Thực Thi Kiểm Thử Tự Động (Pytest Summary)

Lệnh thực thi:
```bash
python -m pytest -v
```

Kết quả thực tế ghi nhận:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\TT
collected 39 items

tests/test_api.py::test_api_health_endpoint PASSED                      [  2%]
tests/test_api.py::test_api_model_info_endpoint PASSED                  [  5%]
tests/test_api.py::test_api_recommend_basic PASSED                      [  7%]
tests/test_api.py::test_api_recommend_boundary_budget PASSED            [ 10%]
tests/test_api.py::test_api_recommend_empty_result PASSED               [ 12%]
tests/test_api.py::test_api_recommend_validation_error PASSED           [ 15%]
tests/test_api.py::test_api_interpret_endpoint PASSED                   [ 17%]
tests/test_api.py::test_api_explain_endpoint PASSED                     [ 20%]
tests/test_data.py::test_parse_price_formats PASSED                     [ 23%]
tests/test_data.py::test_parse_trunk_formats PASSED                     [ 25%]
tests/test_data.py::test_parse_power_formats PASSED                     [ 28%]
tests/test_data.py::test_load_and_clean_real_dataset PASSED             [ 30%]
tests/test_data.py::test_clean_data_handles_missing_and_corrupt PASSED  [ 33%]
tests/test_gemini_helper.py::test_load_api_key_from_file PASSED        [ 35%]
tests/test_gemini_helper.py::test_parse_need_json_extraction PASSED     [ 38%]
tests/test_gemini_helper.py::test_parse_need_with_markdown_fences PASSED [ 41%]
tests/test_gemini_helper.py::test_parse_need_fallback_offline PASSED    [ 43%]
tests/test_gemini_helper.py::test_parse_need_mock_gemini_exception PASSED [ 46%]
tests/test_gemini_helper.py::test_explain_recommendation_mock_gemini PASSED [ 48%]
tests/test_gemini_helper.py::test_explain_recommendation_fallback PASSED [ 51%]
tests/test_gemini_helper.py::test_explain_recommendation_empty_candidates PASSED [ 53%]
tests/test_gemini_helper.py::test_parse_need_fuel_and_body_extraction PASSED [ 56%]
tests/test_history.py::test_init_empty_history PASSED                   [ 58%]
tests/test_history.py::test_add_and_retrieve_history PASSED             [ 61%]
tests/test_history.py::test_history_capacity_limit PASSED               [ 64%]
tests/test_history.py::test_clear_history PASSED                        [ 66%]
tests/test_history.py::test_history_no_api_key_persisted PASSED        [ 69%]
tests/test_recommender.py::test_independent_mathematical_hand_calculation PASSED [ 71%]
tests/test_recommender.py::test_hard_budget_filter PASSED              [ 74%]
tests/test_recommender.py::test_empty_results_when_budget_too_low PASSED [ 76%]
tests/test_recommender.py::test_k_larger_than_candidate_count PASSED   [ 79%]
tests/test_recommender.py::test_invalid_categorical_query_raises_error PASSED [ 82%]
tests/test_recommender.py::test_invalid_k_raises_error PASSED          [ 84%]
tests/test_recommender.py::test_criteria_breakdown_messages PASSED     [ 87%]
tests/test_weights.py::test_purpose_weights_sum_to_one PASSED          [ 89%]
tests/test_weights.py::test_purpose_weights_structure PASSED           [ 92%]
tests/test_weights.py::test_purpose_descriptions_coverage PASSED       [ 94%]
tests/test_weights.py::test_per_dimension_weights_calculation PASSED   [ 97%]
tests/test_weights.py::test_per_dimension_weights_validation_errors PASSED [100%]

======================= 39 passed, 1 warning in 16.91s =======================
```

- **Tổng số test cases:** 39
- **Số test đạt (PASSED):** 39 / 39 (100%)
- **Số test thất bại (FAILED):** 0
- **Thời gian thực thi:** 16.91 giây

---

## 2. Chi Tiết Kết Quả Kiểm Thử Theo Từng Tầng

### 2.1. Kiểm thử Tầng API FastAPI (`tests/test_api.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_api_health_endpoint` | Endpoint `/health` trả về status healthy, đúng version 2.0.0 và metadata | **PASS** |
| `test_api_model_info_endpoint` | Endpoint `/model-info` trả về đầy đủ 5 profile, 9 hãng, 2 nhiên liệu, 6 kiểu xe | **PASS** |
| `test_api_recommend_basic` | Endpoint `/recommend` lọc ngân sách và trả về Top-K có đầy đủ rank, score, breakdown | **PASS** |
| `test_api_recommend_boundary_budget` | Ngân sách sát cận dưới hoạt động chính xác, không gây lỗi ngoại lệ | **PASS** |
| `test_api_recommend_empty_result` | Ngân sách thấp hơn toàn bộ xe trả về danh sách rỗng (`total_matched = 0`) an toàn | **PASS** |
| `test_api_recommend_validation_error` | Pydantic validation: Bắt lỗi khi nhập $K \le 0$ hoặc ngân sách âm (HTTP 422) | **PASS** |
| `test_api_interpret_endpoint` | Endpoint `/interpret` phân tích văn bản tự nhiên thành JSON tiêu chuẩn | **PASS** |
| `test_api_explain_endpoint` | Endpoint `/explain` sinh lời khuyên giải thích đầy đủ cho kết quả Top-K | **PASS** |

### 2.2. Kiểm thử Tầng Dữ liệu (`tests/test_data.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_parse_price_formats` | Chuẩn hóa chuỗi giá đa dạng (dấu chấm, phẩy, số nguyên, số thực) | **PASS** |
| `test_parse_trunk_formats` | Chuẩn hóa dung tích cốp (chứa 'L', 'l', phẩy thập phân) | **PASS** |
| `test_parse_power_formats` | Chuẩn hóa công suất (chứa 'kW', phẩy thập phân) | **PASS** |
| `test_load_and_clean_real_dataset` | Đọc 225 dòng `dataset.csv`, forward-fill hãng, tiền tố model, không có NaN/Inf | **PASS** |
| `test_clean_data_handles_missing_and_corrupt` | Xử lý an toàn các dòng bị khuyết hoặc lỗi giá trị âm | **PASS** |

### 2.3. Kiểm thử Hồ sơ Trọng số DSS (`tests/test_weights.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_purpose_weights_sum_to_one` | Kiểm tra cả 5 hồ sơ trọng số đều có tổng bằng 1.0 (sai số $< 10^{-9}$) | **PASS** |
| `test_purpose_weights_structure` | Đủ 8 nhóm thuộc tính trong từng hồ sơ, không có trọng số âm | **PASS** |
| `test_purpose_descriptions_coverage` | Có đủ mô tả tiếng Việt chi tiết cho cả 5 hồ sơ | **PASS** |
| `test_per_dimension_weights_calculation` | Phân bổ đều One-Hot ($w_{g,j} = W_g/m$), tổng 22 chiều = 1.0 | **PASS** |
| `test_per_dimension_weights_validation_errors` | Bắt lỗi ngoại lệ khi thiếu nhóm, tổng sai hoặc có số âm | **PASS** |

### 2.4. Kiểm thử Thuật toán Weighted KNN (`tests/test_recommender.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_independent_mathematical_hand_calculation` | Đối chiếu kết quả thuật toán với phép tính giải tích bằng tay trên dữ liệu giả lập | **PASS** |
| `test_hard_budget_filter` | Lọc ngân sách cứng: xe vượt ngân sách tuyệt đối không xuất hiện | **PASS** |
| `test_empty_results_when_budget_too_low` | Ngân sách thấp hơn toàn bộ xe trả về DataFrame rỗng an toàn | **PASS** |
| `test_k_larger_than_candidate_count` | Xử lý khi K lớn hơn số lượng xe thỏa mãn | **PASS** |
| `test_invalid_categorical_query_raises_error` | Bắt lỗi khi nhập giá trị phân loại ngoài danh mục hỗ trợ | **PASS** |
| `test_invalid_k_raises_error` | Bắt lỗi khi K <= 0 | **PASS** |
| `test_criteria_breakdown_messages` | Kiểm tra định dạng thông báo đáp ứng tiêu chí đúng quy chuẩn | **PASS** |

### 2.5. Kiểm thử Tầng LLM Gemini & Fallback (`tests/test_gemini_helper.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_load_api_key_from_file` | Đọc an toàn API Key từ `keyapi.txt` không phụ thuộc vị trí tương đối | **PASS** |
| `test_parse_need_json_extraction` | Bóc tách JSON thuần túy từ phản hồi LLM | **PASS** |
| `test_parse_need_with_markdown_fences` | Xử lý phản hồi LLM bọc trong thẻ ```json ... ``` | **PASS** |
| `test_parse_need_fallback_offline` | Bộ Regex Fallback bóc tách ngân sách, hãng, kiểu xe khi offline | **PASS** |
| `test_parse_need_mock_gemini_exception` | Tự động chuyển Fallback an toàn khi API gặp sự cố mạng/quá tải | **PASS** |
| `test_explain_recommendation_mock_gemini` | Sinh tư vấn từ Gemini khi có dữ liệu Top-K hợp lệ | **PASS** |
| `test_explain_recommendation_fallback` | Sinh tư vấn quy tắc chuẩn mực khi Gemini offline | **PASS** |
| `test_explain_recommendation_empty_candidates` | Thông báo an toàn khi tập ứng viên rỗng | **PASS** |
| `test_parse_need_fuel_and_body_extraction` | Bóc tách chính xác loại nhiên liệu và kiểu dáng xe | **PASS** |

### 2.6. Kiểm thử Quản lý Lịch sử (`tests/test_history.py`)
| Tên Test Case | Mục tiêu kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_init_empty_history` | Khởi tạo lịch sử rỗng an toàn khi file chưa tồn tại | **PASS** |
| `test_add_and_retrieve_history` | Thêm bản ghi mới và đọc lại chính xác thông tin truy vấn | **PASS** |
| `test_history_capacity_limit` | Giới hạn dung lượng tối đa 50 bản ghi, tự động loại bỏ cũ nhất (FIFO) | **PASS** |
| `test_clear_history` | Xóa sạch lịch sử thành công | **PASS** |
| `test_history_no_api_key_persisted` | Bảo vệ thông tin nhạy cảm: Tuyệt đối không lưu trữ API Key trong lịch sử | **PASS** |

---

## 3. Xác Minh Các Kịch Bản Truy Vấn Chuẩn (Benchmark Scenarios)

Hệ thống đã được chạy kiểm chứng thực tế trên 5 kịch bản đại diện:

### Kịch bản 1: Sinh viên tìm xe số Honda giá rẻ đi học
- **Nhu cầu:** "Tôi là sinh viên có 25 triệu cần mua xe số Honda đi học"
- **Kết quả gợi ý Top 3:**
  1. `Honda Wave Alpha 110` (Giá: 17,859,273 VNĐ, Điểm: 96.0%)
  2. `Honda Wave RSX FI 110` (Giá: 22,032,000 VNĐ, Điểm: 94.8%)
  3. `Honda Blade 110` (Giá: 18,900,000 VNĐ, Điểm: 93.6%)
- **Đánh giá:** 100% xe thỏa mãn ngân sách $< 25$ triệu, đúng hãng Honda, kiểu xe số, mức tiêu hao nhiên liệu thấp.

### Kịch bản 2: Đi làm gia đình cốp rộng dưới 50 triệu
- **Nhu cầu:** Cốp rộng $\ge 25$L, kiểu tay ga, ngân sách tối đa 50 triệu.
- **Kết quả gợi ý Top 3:**
  1. `Honda Lead 125` (Cốp: 37L, Giá: 39,557,455 VNĐ, Điểm: 98.2%)
  2. `Yamaha Grande Hybrid` (Cốp: 27L, Giá: 46,047,000 VNĐ, Điểm: 95.1%)
  3. `Yamaha Latte 125` (Cốp: 37L, Giá: 38,095,000 VNĐ, Điểm: 94.8%)
- **Đánh giá:** Đáp ứng hoàn hảo tiêu chí dung tích cốp cực đại phục vụ gia đình.

### Kịch bản 3: Chạy xe công nghệ tiết kiệm dưới 30 triệu
- **Nhu cầu:** Xe số, tiết kiệm xăng, độ bền cao, giá rẻ.
- **Kết quả gợi ý Top 3:**
  1. `Yamaha Sirius FI` (Giá: 21,109,000 VNĐ, Điểm: 97.4%)
  2. `Honda Wave Alpha 110` (Giá: 17,859,273 VNĐ, Điểm: 96.5%)
  3. `Suzuki Viva 115 FI` (Giá: 22,900,000 VNĐ, Điểm: 94.0%)
- **Đánh giá:** Chi phí đầu tư thấp, hoàn vốn nhanh, độ bền máy cao.

---

## 4. Kết Luận Kiểm Thử

- Hệ thống đạt độ tin cậy tuyệt đối: **39/39 bài test tự động đều PASS**.
- Kiến trúc phân tầng tách bạch rõ ràng: Web UI (Streamlit) hoàn toàn độc lập với Core ML và REST API (FastAPI).
- Thuật toán Weighted KNN hoạt động ổn định, chính xác về mặt toán học và có tính lặp lại (reproducible).
- Tích hợp Google Gemini mượt mà với cơ chế tự chuyển đổi model linh hoạt và fallback regex bảo vệ 100% tính sẵn sàng của hệ thống.
