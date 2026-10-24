# Hệ Thống Hỗ Trợ Ra Quyết Định (DSS) Gợi Ý Xe Máy (Multi-tier Architecture)

> **Đề tài:** Xây dựng hệ thống hỗ trợ ra quyết định gợi ý xe máy theo nhu cầu người dùng bằng Weighted K-Nearest Neighbors và Google Gemini.  
> **Kiến trúc:** Multi-tier (FastAPI RESTful Backend + Streamlit Web Presentation + Gemini LLM Layer + Quantitative KNN Engine).  
> **Môn học:** Hệ trợ giúp quyết định (Decision Support Systems - DSS)

---

## 📌 Giới thiệu dự án

Hệ thống hỗ trợ người mua xe máy tại Việt Nam tìm kiếm mẫu xe phù hợp nhất với nhu cầu sử dụng và khả năng tài chính, giải quyết bài toán đa tiêu chí (Multi-Criteria Decision Making - MCDM) thông qua sự kết hợp giữa:
1. **Thuật toán định lượng Weighted KNN (K-Nearest Neighbors):** 
   - Lọc ngân sách cứng (Hard Budget Ceiling): không phạt xe giá rẻ hơn ngân sách tối đa.
   - Chuẩn hóa dữ liệu bằng `MinMaxScaler`, mã hóa `One-Hot Encoding` 22 chiều đặc trưng.
   - Áp dụng ma trận trọng số theo 5 hồ sơ mục đích sử dụng chuẩn.
   - Quy tắc phá vỡ hòa điểm xác định (Deterministic Tie-breaking) đảm bảo tính minh bạch.
2. **Trí tuệ nhân tạo Google Gemini AI:**
   - Tiếp nhận nhu cầu bằng ngôn ngữ tự nhiên đời thường, bóc tách thành bộ tiêu chí kỹ thuật JSON chuẩn.
   - Phân tích và sinh lời khuyên giải thích chuyên sâu (150 - 220 từ) dựa trên kết quả tính toán định lượng của thuật toán.
3. **Kiến trúc phân tầng Microservice (FastAPI + Streamlit):**
   - Tách biệt hoàn toàn tầng tính toán và dữ liệu (Backend REST API) với tầng hiển thị (Streamlit Web UI).
   - Bảo mật tuyệt đối: Khóa bí mật API Key được quản lý tập trung ở Server-side, Client không nắm giữ secret.
   - Cung cấp Swagger UI và OpenAPI Specification chuẩn tại `http://127.0.0.1:8000/docs`.

---

## 📂 Cấu trúc Thư mục Dự án

```text
D:\TT/
├── data/
│   └── dataset.csv              # Dữ liệu 225 mẫu xe máy đã làm sạch (Honda, Yamaha, VinFast, ...)
├── model/
│   ├── train_export.py          # Pipeline tiền xử lý và trích xuất artifact mô hình
│   ├── knn_model.joblib         # Scaler, ma trận đặc trưng & dữ liệu ứng viên
│   └── model_metadata.json      # Metadata mô hình (version 2.0.0, 22 chiều, khoảng giá trị)
├── api/
│   ├── __init__.py
│   ├── schemas.py               # Pydantic schemas (Request, Response DTOs)
│   ├── services.py              # Business services & Singleton MotoRecommender
│   └── main.py                  # FastAPI application, CORS, endpoints (/health, /recommend, ...)
├── core/
│   ├── __init__.py
│   └── recommender.py           # Lõi thuật toán Weighted KNN, Hard Filter, Tie-breaking
├── llm/
│   ├── __init__.py
│   └── gemini_helper.py         # Kết nối Gemini SDK, chuyển đổi model linh hoạt, Fallback offline
├── web/
│   └── app.py                   # Giao diện người dùng Streamlit thuần REST Client
├── tests/                       # Toàn bộ 39 bài kiểm thử tự động (100% Passed)
│   ├── test_api.py              # Kiểm thử toàn diện 8 test cases trên các REST endpoint FastAPI
│   ├── test_data.py             # Kiểm thử làm sạch và cấu trúc dữ liệu
│   ├── test_weights.py          # Kiểm thử hồ sơ trọng số & phân bổ One-Hot
│   ├── test_recommender.py      # Kiểm thử KNN & đối chiếu phép tính giải tích bằng tay
│   ├── test_gemini_helper.py    # Kiểm thử NLP parser & mock ngoại lệ Gemini
│   └── test_history.py          # Kiểm thử lưu trữ phiên tư vấn an toàn
├── docs/                        # Tài liệu học thuật phục vụ báo cáo
│   ├── ARCHITECTURE.md          # Chi tiết kiến trúc phân tầng, đặc tả API, sơ đồ tương tác
│   ├── MATHEMATICAL_MODEL.md    # Mô hình toán học, phân tích độ nhạy, tính tay giải tích
│   └── TEST_REPORT.md           # Báo cáo chi tiết 39 test cases và kịch bản thực nghiệm
├── app.py                       # Proxy script chuyển tiếp hỗ trợ tương thích ngược
├── recommender.py               # Proxy module xuất khẩu từ core.recommender
├── gemini_helper.py             # Proxy module xuất khẩu từ llm.gemini_helper
├── history.py                   # Quản lý lưu trữ lịch sử truy vấn
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── keyapi.txt                   # Chứa Google Gemini API Key (bảo mật server-side)
├── run.bat                      # 1-Click: Tự động khởi động cả FastAPI và Streamlit
├── run_api.bat                  # Khởi động riêng Backend FastAPI (Port 8000)
├── run_web.bat                  # Khởi động riêng Frontend Streamlit (Port 8501)
└── run_test.bat                 # Chạy toàn bộ 39 unit & API tests với Pytest
```

---

## 🚀 Hướng Dẫn Vận Hành Hệ Thống

### ⚡ Cách 1: Khởi động 1-Click (Khuyên dùng trên Windows)
Chỉ cần mở thư mục dự án và click đúp chuột vào:
- **`run.bat`**: Tự động mở Backend FastAPI tại `http://127.0.0.1:8000` và khởi chạy giao diện Web Streamlit tại `http://localhost:8501`.
- **`run_test.bat`**: Tự động chạy toàn bộ **39 bài kiểm thử** và hiển thị kết quả.

---

### 💻 Cách 2: Khởi chạy bằng Dòng Lệnh (PowerShell / Terminal)

#### Bước 1: Cài đặt thư viện phụ thuộc
Mở PowerShell tại thư mục `D:\TT`:
```powershell
pip install -r requirements.txt
```

#### Bước 2: Huấn luyện và xuất khẩu Model Artifact (nếu có cập nhật dữ liệu)
```powershell
python model/train_export.py
```

#### Bước 3: Khởi động Backend REST API (FastAPI)
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Server sẽ hoạt động tại: `http://127.0.0.1:8000`
- Giao diện tài liệu tương tác **Swagger UI**: `http://127.0.0.1:8000/docs`
- Giao diện tài liệu **ReDoc**: `http://127.0.0.1:8000/redoc`
- Kiểm tra sức khỏe hệ thống: `http://127.0.0.1:8000/health`

#### Bước 4: Khởi động Giao diện Web (Streamlit)
Mở một cửa sổ PowerShell mới:
```powershell
python -m streamlit run web/app.py
```
*(Hoặc chạy lệnh tương thích ngược: `streamlit run app.py`)*
Trình duyệt sẽ tự động mở giao diện tại `http://localhost:8501`.

---

## 🧪 Kiểm Thử Tự Động (Automated Testing)

Chạy toàn bộ 39 bài test với Pytest:
```powershell
python -m pytest -v
```

Kết quả: **39 / 39 PASSED** (100% đạt chuẩn chất lượng).

---

## 📡 Danh Mục REST API Endpoints

| Phương thức | Đường dẫn | Chức năng chính |
| :---: | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái hoạt động của backend, version và số bản ghi |
| `GET` | `/model-info` | Lấy danh mục 5 hồ sơ, 9 hãng, 2 nhiên liệu, 6 kiểu xe, khoảng giá trị số |
| `POST` | `/recommend` | Nhận tiêu chí người dùng, lọc ngân sách cứng và xếp hạng Top-K bằng Weighted KNN |
| `POST` | `/interpret` | Bóc tách ngôn ngữ tự nhiên thành JSON tiêu chí chuẩn bằng Gemini NLP / Regex |
| `POST` | `/explain` | Sinh văn bản phân tích, so sánh và tư vấn lý do chọn xe dựa trên kết quả Top-K |

---

## 🔒 Quản Lý Bảo Mật API Key

1. Hệ thống nạp Google Gemini API Key tự động từ biến môi trường `GEMINI_API_KEY` hoặc file `keyapi.txt` tại thư mục gốc.
2. Tầng giao diện người dùng (Streamlit) **không chứa và không yêu cầu nhập API Key**, triệt tiêu nguy cơ lộ lọt secret từ phía Client.
3. API Key tuyệt đối không xuất hiện trong file lịch sử `history.json`, nhật ký terminal hay tài liệu API.
