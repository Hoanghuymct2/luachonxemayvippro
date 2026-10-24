# Hệ Thống Hỗ Trợ Ra Quyết Định (DSS) Gợi Ý Xe Máy

> **Đề tài:** Xây dựng hệ thống hỗ trợ ra quyết định gợi ý xe máy theo nhu cầu người dùng bằng Weighted K-Nearest Neighbors và Google Gemini.  
> **Môn học:** Hệ trợ giúp quyết định (Decision Support Systems - DSS)

---

## 📌 Giới thiệu Dự án

Dự án là một hệ thống DSS hoàn chỉnh giúp người dùng tìm kiếm mẫu xe máy phù hợp nhất với nhu cầu và khả năng tài chính dựa trên:
1. **Thuật toán định lượng Weighted KNN (K-Nearest Neighbors):** 
   - **Lọc trần ngân sách cứng (Hard Budget Ceiling):** Không phạt khoảng cách các xe rẻ hơn ngân sách tối đa của người dùng.
   - **Chuẩn hóa MinMaxScaler & One-Hot Encoding:** Đưa 22 chiều đặc trưng (4 thuộc tính số + 18 chiều phân loại) về cùng thang đo chuẩn $[0, 1]$.
   - **Ma trận trọng số theo 5 hồ sơ mục đích:** Phân bổ đều trọng số theo từng chiều đặc trưng ($\sum w_i = 1.0$).
   - **Quy tắc phá vỡ hòa điểm xác định (Deterministic Tie-breaking):** Ưu tiên năm sản xuất mới hơn, giá bán tiết kiệm hơn và tên xe theo bảng chữ cái.
2. **Trí tuệ nhân tạo Google Gemini AI:**
   - Tiếp nhận câu nói tự nhiên của người dùng, phân tích ngữ nghĩa và trích xuất thành bộ tiêu chí kỹ thuật cụ thể.
   - Đóng vai trò chuyên gia phân tích kết quả tính toán định lượng và đưa ra lời khuyên thực tế chuyên sâu (150 - 220 từ).
   - Tự động chuyển đổi linh hoạt giữa các phiên bản model (`gemini-2.5-flash`, `gemini-3.1-flash-lite`) và có bộ Fallback Regex dự phòng khi offline.

---

## 📂 Cấu trúc Thư mục Tinh Gọn

Dự án được gom gọn vào các file cốt lõi tại thư mục gốc, không phân mảnh thành các thư mục con rườm rà:

```text
D:\TT/
├── dataset.csv            # Dữ liệu gốc 225 mẫu xe máy thuộc 9 hãng xe (Honda, Yamaha, ...)
├── keyapi.txt             # Lưu trữ Google Gemini API Key (bảo mật cục bộ)
├── app.py                 # Giao diện Streamlit DSS trực tiếp, mượt mà, đầy đủ tính năng
├── recommender.py         # Lõi thuật toán Weighted KNN, chuẩn hóa, lọc ngân sách, tie-breaking
├── gemini_helper.py       # Tích hợp Gemini AI & Fallback NLP, bảo mật key
├── history.py             # Quản lý lưu trữ lịch sử tư vấn
├── history.json           # File dữ liệu lịch sử tìm kiếm gần đây
├── api.py                 # (Tùy chọn) 1 file duy nhất cho FastAPI REST API nếu cần dùng API
├── requirements.txt       # Danh sách thư viện phụ thuộc tinh gọn
├── README.md              # Hướng dẫn chi tiết vận hành hệ thống
├── run.bat                # 1 Cú click chuột khởi động ngay ứng dụng
├── .gitignore             # Bỏ qua file rác và key
├── .streamlit/
│   └── config.toml        # Cấu hình giao diện Streamlit
└── docs/
    ├── ARCHITECTURE.md    # Tài liệu kiến trúc hệ thống
    └── MATHEMATICAL_MODEL.md # Tài liệu mô hình toán học & tính tay giải tích
```

---

## 🚀 Hướng Dẫn Vận Hành

### ⚡ Cách 1: Khởi động 1-Click (Đơn giản nhất trên Windows)
- **Click đúp chuột vào file `run.bat`**: Ứng dụng sẽ tự động khởi chạy và mở ngay giao diện Web tại `http://localhost:8501`.

---

### 💻 Cách 2: Khởi chạy bằng Dòng Lệnh (PowerShell)
Mở PowerShell tại thư mục `D:\TT`:

1. **Cài đặt thư viện:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Khởi động ứng dụng Streamlit:**
   ```powershell
   python -m streamlit run app.py
   ```
   *Trình duyệt sẽ tự động mở giao diện tại `http://localhost:8501`.*

3. **(Tùy chọn) Chạy Backend REST API:**
   Nếu bạn muốn thử nghiệm API FastAPI:
   ```powershell
   python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Swagger UI tài liệu tương tác: `http://127.0.0.1:8000/docs`.*

---

## 🔒 Bảo Mật Khóa API
- File `keyapi.txt` chứa Google Gemini API Key được giữ nguyên tại thư mục gốc, hệ thống tự động đọc vào bộ nhớ khi cần.
- Tuyệt đối không in API Key ra màn hình terminal, giao diện người dùng, log hệ thống hay lưu vào `history.json`.
- File `keyapi.txt` đã được đưa vào `.gitignore` để ngăn chặn vô tình commit lên Git.
