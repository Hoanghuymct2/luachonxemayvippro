# Hệ Thống Hỗ Trợ Ra Quyết Định (DSS) Gợi Ý Xe Máy

> **Đề tài:** Xây dựng hệ thống hỗ trợ ra quyết định gợi ý xe máy theo nhu cầu người dùng bằng Weighted K-Nearest Neighbors và Google Gemini.  
> **Môn học:** Hệ trợ giúp quyết định (Decision Support Systems - DSS)

---

## 📌 Giới thiệu dự án

Dự án là một hệ thống DSS hoàn chỉnh kết hợp giữa:
1. **Thuật toán định lượng Weighted KNN (K-Nearest Neighbors):** Lọc ngân sách cứng, chuẩn hóa dữ liệu bằng `MinMaxScaler`, mã hóa `One-Hot Encoding` 22 chiều, áp dụng ma trận trọng số theo 5 hồ sơ mục đích sử dụng và tính điểm phù hợp tương đối.
2. **Trí tuệ nhân tạo Google Gemini AI:** Tiếp nhận và bóc tách nhu cầu đời thường (ngôn ngữ tự nhiên) thành bộ tiêu chuẩn kỹ thuật, đồng thời đóng vai trò chuyên gia giải thích kết quả gợi ý và đưa ra lời khuyên thực tế.

---

## 📂 Cấu trúc thư mục

```text
D:\TT/
├── dataset.csv                  # Dữ liệu 225 mẫu xe máy thuộc 9 hãng xe
├── keyapi.txt                   # Chứa Google Gemini API Key (được bảo mật)
├── app.py                       # Giao diện người dùng Streamlit và điều phối luồng
├── recommender.py               # Lõi tính toán Weighted KNN, làm sạch dữ liệu và trọng số
├── gemini_helper.py             # Tích hợp Gemini API, bóc tách JSON và bộ fallback offline
├── history.py                   # Quản lý lưu trữ, nạp và xóa lịch sử tìm kiếm an toàn
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── README.md                    # Hướng dẫn cài đặt và vận hành
├── .gitignore                   # Loại trừ secret và cache khỏi Git
├── .streamlit/
│   └── config.toml              # Cấu hình giao diện Streamlit
├── tests/                       # Bộ kiểm thử tự động với Pytest (31 tests passed)
│   ├── test_data.py             # Kiểm thử đọc, làm sạch và chuẩn hóa dữ liệu
│   ├── test_weights.py          # Kiểm thử hồ sơ trọng số và phân bổ chiều
│   ├── test_recommender.py      # Kiểm thử thuật toán KNN và đối chiếu giải tích tay
│   ├── test_gemini_helper.py    # Kiểm thử Gemini mock và bộ fallback từ khóa
│   └── test_history.py          # Kiểm thử lưu trữ lịch sử và chống rò rỉ secret
├── docs/                        # Tài liệu báo cáo môn học
│   ├── ARCHITECTURE.md          # Kiến trúc chi tiết, sơ đồ luồng Mermaid
│   ├── MATHEMATICAL_MODEL.md    # Cơ sở học thuật DSS và ví dụ tính tay từng bước
│   └── TEST_REPORT.md           # Báo cáo kết quả kiểm thử thực tế và thống kê dữ liệu
├── verify_queries.py            # Kịch bản kiểm chứng 3 truy vấn mẫu trên dữ liệu thật
└── history.json                 # File lưu trữ lịch sử tìm kiếm gần đây
```

---

## 🚀 Hướng dẫn Cài đặt và Chạy trên Windows

### ⚡ Cách đơn giản nhất (Chỉ 1 cú click chuột):
Tôi đã tạo sẵn 2 file chạy tự động trong thư mục dự án:
- **`run.bat`**: **Click đúp chuột vào file này** để khởi chạy ứng dụng ngay lập tức (tự động mở trình duyệt web `http://localhost:8501`, tự động cài thư viện nếu thiếu).
- **`run_test.bat`**: **Click đúp chuột** để chạy toàn bộ 31 bài kiểm thử tự động và xem kết quả.

---

### 💻 Cách chạy qua dòng lệnh (Terminal / PowerShell):
- Hệ điều hành: Windows 10 hoặc Windows 11.
- Phiên bản Python: **Python 3.10, 3.11 hoặc 3.12** (đã được kiểm thử trên Python 3.12.10).

### 2. Mở Terminal tại thư mục dự án
Mở PowerShell hoặc Command Prompt và di chuyển vào thư mục:
```powershell
cd D:\TT
```

### 3. Tạo môi trường ảo (Virtual Environment)
```powershell
python -m venv .venv
```

### 4. Kích hoạt môi trường ảo
- **Cách 1: Sử dụng PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```
*Lưu ý: Nếu PowerShell báo lỗi chặn script (`Execution_Policies`), bạn chỉ cần cho phép phiên hiện tại:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

- **Cách 2: Sử dụng Command Prompt (CMD)**
```cmd
.\.venv\Scripts\activate.bat
```

### 5. Nâng cấp pip và Cài đặt Thư viện Phụ thuộc
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Cấu hình Google Gemini API Key
Hệ thống hỗ trợ nạp key tự động theo thứ tự ưu tiên:
1. **File `keyapi.txt`:** Tạo file `keyapi.txt` tại thư mục gốc dự án và dán API key vào dòng đầu tiên (không có dấu cách thừa).
2. **Biến môi trường (Environment Variable):**
   ```powershell
   $env:GEMINI_API_KEY="AIzaSyYourApiKeyHere"
   ```
3. **Nhập trực tiếp trên Giao diện:** Người dùng có thể nhập key vào trường mật khẩu tại Sidebar của ứng dụng.

*(Lưu ý: Nếu không có API Key, hệ thống vẫn hoạt động 100% nhờ bộ Fallback bằng luật Regex và từ khóa tiếng Việt).*

### 7. Chạy Kiểm Thử Tự Động (Pytest)
Trước khi chạy ứng dụng, hãy kiểm tra toàn bộ 31 test cases:
```powershell
python -m pytest -q
```
Kết quả kỳ vọng: `31 passed in ...s`.

### 8. Khởi chạy Ứng dụng Streamlit
```powershell
streamlit run app.py
```
Sau khi khởi chạy, Streamlit sẽ mở trình duyệt tự động hoặc cung cấp địa chỉ truy cập:
```text
Local URL: http://localhost:8501
Network URL: http://...:8501
```

---

## 📊 5 Hồ Sơ Trọng Số DSS Mặc Định

| Hồ sơ mục đích | Giá(VND) | Dung tích cốp | Kiểu xe | Nhiên liệu | Hãng xe | Công suất | Năm SX | Tình trạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đi làm, đi phố & Gia đình** | 20% | **25%** | 20% | 15% | 10% | 5% | 5% | 0% |
| **Học sinh, Sinh viên** | **35%** | 15% | 20% | 15% | 5% | 5% | 5% | 0% |
| **Chạy dịch vụ (Grab/Be/Ship)** | **30%** | 10% | 15% | 20% | 5% | 15% | 5% | 0% |
| **Đi phượt, Thể thao & Tốc độ** | 10% | 5% | 25% | 10% | 10% | **35%** | 5% | 0% |
| **Thời trang, Sang trọng & Cao cấp** | 10% | 15% | **25%** | 5% | **25%** | 10% | 10% | 0% |

---

## 🛠️ Xử lý sự cố thường gặp (Troubleshooting)

1. **Lỗi `UnicodeEncodeError` khi in ra Terminal Windows:**
   - Console Windows mặc định sử dụng bảng mã `cp1252` hoặc `cp437`.
   - Khắc phục: Chạy lệnh `chcp 65001` trước khi chạy script hoặc thiết lập biến môi trường `$env:PYTHONIOENCODING="utf-8"`.
2. **Model `gemini-2.5-flash` trả về lỗi 404:**
   - Một số tài khoản Google AI mới không hỗ trợ phiên bản preview 2.5. Hệ thống đã tích hợp sẵn cơ chế tự động chuyển sang `gemini-3.1-flash-lite`, không làm gián đoạn trải nghiệm người dùng.
3. **Không tìm thấy file dữ liệu:**
   - Đảm bảo file `dataset.csv` nằm cùng thư mục với `app.py`.

