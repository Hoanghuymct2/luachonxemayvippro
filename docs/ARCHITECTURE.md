# KIẾN TRÚC HỆ THỐNG HỖ TRỢ RA QUYẾT ĐỊNH (DSS) GỢI Ý XE MÁY

Hệ thống kết hợp thuật toán tính toán định lượng **Weighted K-Nearest Neighbors (KNN)** và mô hình ngôn ngữ lớn **Google Gemini AI**, được tổ chức tinh gọn, trực tiếp, loại bỏ các thành phần phân mảnh dư thừa.

---

## 1. Bối cảnh và Bài toán Hỗ trợ Ra Quyết định (DSS)

### 1.1. Bài toán thực tế
Trên thị trường xe máy Việt Nam hiện nay, người tiêu dùng đứng trước hàng trăm mẫu mã từ nhiều thương hiệu (Honda, Yamaha, Suzuki, VinFast, Vespa, SYM, Yadea, Dat Bike, Piaggio) với các công nghệ động cơ (xăng, điện), kiểu dáng (tay ga, xe số, côn tay, 50cc), tầm giá từ 13 triệu đến hơn 1,2 tỷ đồng. Người mua thường gặp các trở ngại:
- **Quá tải thông tin (Information Overload):** Khó so sánh đồng thời nhiều thuộc tính kỹ thuật.
- **Xung đột mục tiêu (Trade-offs):** Ngân sách có hạn nhưng mong muốn cốp rộng, máy khỏe, kiểu dáng đẹp.
- **Khó diễn đạt bằng thông số kỹ thuật cứng:** Người dùng thường diễn đạt bằng nhu cầu đời sống ("tôi là sinh viên cần xe đi học", "chạy xe ôm công nghệ", "thích đi phượt xa").

### 1.2. Mục tiêu của hệ thống DSS
1. **Lớp giao diện ngôn ngữ tự nhiên:** Tiếp nhận mô tả đời thường và dịch sang bộ tiêu chí kỹ thuật chuẩn xác bằng Gemini AI.
2. **Lớp tính toán định lượng khách quan (Weighted KNN):** Đảm bảo tính khả thi tài chính (lọc trần ngân sách cứng), chuẩn hóa không gian đặc trưng (MinMaxScaler + One-Hot 22 chiều) và xếp hạng Top-K xe gần nhất với hồ sơ ưu tiên.
3. **Lớp giải thích và tư vấn (Gemini):** Đóng vai trò chuyên gia phân tích lý do chọn xe dựa trên kết quả định lượng, so sánh đối chứng và đưa ra lời khuyên thực tế.

---

## 2. Sơ đồ Kiến trúc Tinh Gọn

```mermaid
flowchart TD
    User([Người dùng]) -->|1. Nhập nhu cầu tự nhiên| UI_NLP[Khu vực Phân tích AI Streamlit]
    User -->|2. Tùy chỉnh tiêu chí & chọn Hồ sơ| UI_Form[Form Tiêu chí & Sidebar]
    
    subgraph Natural Language Understanding [gemini_helper.py]
        UI_NLP --> Gemini_Call{Gemini API Khả dụng?}
        Gemini_Call -->|Có| Gemini_NLP[Gemini 2.5/3.1 JSON Parser]
        Gemini_Call -->|Không / Lỗi| Regex_NLP[Rule-based Regex Fallback]
        Gemini_NLP --> PreFill[Điền tự động vào Form tiêu chí]
        Regex_NLP --> PreFill
    end
    
    PreFill --> UI_Form
    
    subgraph Quantitative Decision Engine [recommender.py]
        UI_Form -->|Query dict & Mục đích| Engine[MotoRecommender]
        Engine --> HardFilter[1. Lọc trần ngân sách cứng: Price <= Budget]
        HardFilter -->|Tập ứng viên hợp lệ| Encoding[2. Vector hóa Query: Scaler + One-Hot]
        Encoding --> WeightedDist[3. Weighted Euclidean Distance dw]
        WeightedDist --> TopKSort[4. Sắp xếp khoảng cách tăng dần & Tie-breaking]
        TopKSort --> ScoreCalc[5. Tính Score = round 1/(1+dw)*100]
        ScoreCalc --> BreakdownCalc[6. Phân tích chi tiết mức độ đáp ứng tiêu chí]
    end
    
    subgraph Explanation & Presentation [app.py]
        BreakdownCalc --> UI_Results[Hiển thị Bảng kết quả & Score Bars]
        TopKSort --> Gemini_Exp_Btn{Người dùng bấm Tư vấn AI?}
        Gemini_Exp_Btn -->|Có| Gemini_Exp[Gemini Explanation Generator]
        Gemini_Exp --> UI_Exp[Hiển thị Lời khuyên Chuyên sâu]
    end
    
    BreakdownCalc --> History_Save[(history.json)]
    History_Save --> UI_History[Hiển thị Lịch sử tìm kiếm]
```

---

## 3. Phân chia Trách nhiệm Các File Cốt Lõi

Toàn bộ hệ thống được gom gọn vào các file chính tại thư mục gốc, không phân mảnh:

| File | Trách nhiệm chính |
| :--- | :--- |
| **`app.py`** | Giao diện người dùng Streamlit toàn diện: form nhập liệu, phân tích NLP, thanh đo Score Bar, bảng xếp hạng Top-K, hiển thị lời khuyên AI và lịch sử. Chạy trực tiếp, không cần trung gian. |
| **`recommender.py`** | Lõi tính toán DSS: Làm sạch dữ liệu, chuẩn hóa MinMaxScaler, mã hóa One-Hot, quản lý 5 hồ sơ trọng số, lọc trần ngân sách cứng, phân bổ trọng số đều theo chiều, tính khoảng cách Weighted Euclidean, quy tắc tie-breaking xác định và phân tích mức độ đáp ứng tiêu chí. |
| **`gemini_helper.py`** | Tích hợp Google Gemini: Quản lý SDK `google-genai`, chuyển đổi model linh hoạt (Gemini 2.5 $\rightarrow$ 3.1), trích xuất tiêu chí JSON, sinh lời khuyên giải thích và bộ Fallback Regex khi offline. |
| **`history.py`** | Quản lý lịch sử truy vấn: Lưu trữ và nạp lịch sử từ `history.json`, chống trùng lặp, bảo vệ tuyệt đối không lưu API key. |
| **`api.py`** | (Tùy chọn) Cung cấp REST API chuẩn FastAPI (`/health`, `/model-info`, `/recommend`, `/interpret`, `/explain`) gói gọn trong 1 file duy nhất cho nhu cầu tích hợp API. |
| **`run.bat`** | Script 1-click tiện lợi: khởi động ứng dụng ngay lập tức tại `http://localhost:8501`. |
| **`dataset.csv`** | Dữ liệu gốc 225 mẫu xe máy thuộc 9 hãng xe phổ biến tại Việt Nam. |
| **`keyapi.txt`** | Lưu trữ khóa Google Gemini API Key an toàn ở máy cục bộ. |

---

## 4. Nguyên tắc Phối hợp giữa Weighted KNN và Gemini

1. **Gemini KHÔNG được tự tạo danh sách xe:** Toàn bộ danh sách gợi ý, mức giá, thứ tự xếp hạng và khoảng cách toán học đều do `recommender.py` tính toán trên dữ liệu thật `dataset.csv`.
2. **Gemini đóng vai trò hỗ trợ:**
   - *Bộ biên dịch ngôn ngữ tự nhiên (NLP Parser):* Chuyển đổi câu nói của người dùng thành các tiêu chí kỹ thuật cụ thể.
   - *Chuyên gia phân tích kết quả (Explainer):* Đọc bảng kết quả Top-K đã được tính toán xong để giải thích lý do Top 1 được chọn và đưa ra lời khuyên trải nghiệm thực tế.
3. **Bảo mật API Key:** Key được nạp tự động từ biến môi trường hoặc `keyapi.txt` tại server, không hiển thị trên giao diện người dùng và không commit vào Git.
