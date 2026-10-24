# KIẾN TRÚC HỆ THỐNG HỖ TRỢ RA QUYẾT ĐỊNH (DSS) GỢI Ý XE MÁY

Hệ thống kết hợp thuật toán tính toán định lượng **Weighted K-Nearest Neighbors (KNN)** và mô hình ngôn ngữ lớn **Google Gemini AI**.

---

## 1. Bối cảnh và Bài toán Hỗ trợ Ra Quyết định (DSS)

### 1.1. Bài toán thực tế
Trên thị trường xe máy Việt Nam hiện nay, người tiêu dùng đứng trước hàng trăm mẫu mã từ nhiều thương hiệu (Honda, Yamaha, Suzuki, VinFast, Vespa, SYM, Yadea, Dat Bike, Piaggio) với các công nghệ động cơ (xăng, điện), kiểu dáng (tay ga, xe số, côn tay, 50cc), tầm giá từ 13 triệu đến hơn 1,2 tỷ đồng. Người mua thường gặp các trở ngại:
- **Quá tải thông tin (Information Overload):** Khó so sánh đồng thời nhiều thuộc tính kỹ thuật.
- **Xung đột mục tiêu (Trade-offs):** Ngân sách có hạn nhưng mong muốn cốp rộng, máy khỏe, kiểu dáng đẹp.
- **Khó diễn đạt bằng thông số kỹ thuật cứng:** Người dùng thường diễn đạt bằng nhu cầu đời sống ("tôi là sinh viên cần xe đi học", "chạy xe ôm công nghệ", "thích đi phượt xa").

### 1.2. Mục tiêu của hệ thống DSS
Xây dựng một hệ trợ giúp quyết định toàn diện:
1. **Lớp giao diện ngôn ngữ tự nhiên:** Tiếp nhận mô tả đời thường và dịch sang bộ tiêu chí kỹ thuật chuẩn xác.
2. **Lớp tính toán định lượng khách quan (Weighted KNN):** Đảm bảo tính khả thi tài chính (lọc ngân sách cứng), chuẩn hóa không gian đặc trưng và xếp hạng Top-K xe gần nhất với hồ sơ ưu tiên.
3. **Lớp giải thích và tư vấn (Gemini):** Đóng vai trò chuyên gia phân tích lý do chọn xe dựa trên kết quả định lượng, so sánh đối chứng và đưa ra lời khuyên thực tế.

---

## 2. Người dùng, Đầu vào và Đầu ra

```text
+-------------------------------------------------------------------------+
|                              NGƯỜI DÙNG                                 |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                                ĐẦU VÀO                                  |
| - Văn bản nhu cầu tự nhiên (Natural text prompt)                        |
| - Hồ sơ mục đích (1 trong 5 profiles)                                    |
| - Ngân sách tối đa (Triệu VNĐ) [Lọc cứng]                               |
| - Các tiêu chí kỹ thuật: Dung tích cốp, Công suất, Hãng, Kiểu xe, v.v.  |
| - Số lượng xe cần gợi ý K (1 - 20)                                      |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                          HỆ THỐNG XỬ LÝ (DSS)                           |
| 1. Gemini / Fallback NLP Engine (Bóc tách tiêu chí)                     |
| 2. Recommender Engine (MinMaxScaler, One-Hot, Hard Filter, Weighted KNN)|
| 3. Explanation Engine (Phân tích tiêu chí + Tư vấn Gemini)              |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                                 ĐẦU RA                                  |
| - Bảng xếp hạng Top-K xe tối ưu nhất (Thứ hạng, Tên, Giá, Thông số)     |
| - Điểm số phù hợp tương đối (Score %) và Khoảng cách Euclid             |
| - Báo cáo phân tích mức độ đáp ứng từng tiêu chí kèm Score Bar màu sắc  |
| - Lời khuyên chuyên sâu từ Gemini AI (150 - 220 từ)                     |
| - Lịch sử tìm kiếm được lưu trữ an toàn                                 |
+-------------------------------------------------------------------------+
```

---

## 3. Sơ đồ Kiến trúc Hệ thống

```mermaid
flowchart TD
    User([Người dùng]) -->|1. Nhập nhu cầu tự nhiên| UI_NLP[Khu vực Phân tích AI Streamlit]
    User -->|2. Tùy chỉnh tiêu chí & chọn Hồ sơ| UI_Form[Form Tiêu chí & Sidebar]
    
    subgraph Natural Language Understanding
        UI_NLP --> Gemini_Call{Gemini API Khả dụng?}
        Gemini_Call -->|Có| Gemini_NLP[Gemini 2.5/3.1 JSON Parser]
        Gemini_Call -->|Không / Lỗi| Regex_NLP[Rule-based Regex Fallback]
        Gemini_NLP --> PreFill[Điền tự động vào Form tiêu chí]
        Regex_NLP --> PreFill
    end
    
    PreFill --> UI_Form
    
    subgraph Quantitative Decision Engine [Weighted KNN Core]
        UI_Form -->|Query dict & Mục đích| Engine[MotoRecommender]
        Engine --> HardFilter[1. Lọc ngân sách cứng: Price <= Budget]
        HardFilter -->|Tập ứng viên hợp lệ| Encoding[2. Vector hóa Query: Scaler + One-Hot]
        Encoding --> WeightedDist[3. Weighted Euclidean Distance dw]
        WeightedDist --> TopKSort[4. Sắp xếp khoảng cách tăng dần & Cắt Top-K]
        TopKSort --> ScoreCalc[5. Tính Score = round(1/(1+dw)*100, 1)]
        ScoreCalc --> BreakdownCalc[6. Phân tích chi tiết mức độ đáp ứng tiêu chí]
    end
    
    subgraph Explanation & Consultation
        BreakdownCalc --> UI_Results[Hiển thị Bảng kết quả & Score Bars]
        TopKSort --> Gemini_Exp_Btn{Người dùng bấm Tư vấn AI?}
        Gemini_Exp_Btn -->|Có| Check_API{Gemini API?}
        Check_API -->|Sẵn sàng| Gemini_Exp[Gemini Explanation Generator]
        Check_API -->|Offline / Lỗi| Fallback_Exp[Rule-based DSS Explanation]
        Gemini_Exp --> UI_Exp[Hiển thị Lời khuyên Chuyên sâu]
        Fallback_Exp --> UI_Exp
    end
    
    BreakdownCalc --> History_Save[(history.json)]
    History_Save --> UI_History[Hiển thị Lịch sử tìm kiếm]
```

---

## 4. Phân chia Trách nhiệm Các Module

Hệ thống được thiết kế theo nguyên lý phân tách trách nhiệm (Separation of Concerns - SoC):

| Module | File | Trách nhiệm chính |
| :--- | :--- | :--- |
| **Giao diện & Điều phối** | `app.py` | Quản lý vòng đời Streamlit, Session State, tiếp nhận sự kiện, hiển thị bảng biểu, thanh đo Score Bar, bảng điều khiển Sidebar. |
| **Lõi Tính toán DSS** | `recommender.py` | Làm sạch dữ liệu, chuẩn hóa MinMaxScaler, mã hóa One-Hot, quản lý 5 hồ sơ trọng số, phân bổ trọng số từng chiều, lọc ngân sách cứng, tính khoảng cách Weighted Euclidean và phân tích tiêu chí. |
| **Tích hợp Gemini AI** | `gemini_helper.py` | Quản lý kết nối Gemini SDK, chuyển nhu cầu tự nhiên sang JSON tiêu chuẩn, sinh lời khuyên giải thích, cơ chế tự động chuyển đổi model (2.5 -> 3.1) và bộ fallback regex toàn diện khi offline. |
| **Quản lý Lịch sử** | `history.py` | Lưu trữ, nạp và xóa lịch sử truy vấn từ `history.json`, kiểm tra trùng lặp trên Streamlit rerun, bảo vệ dữ liệu nhạy cảm (không lưu API key). |
| **Kiểm thử tự động** | `tests/` | Bộ kiểm thử toàn diện với Pytest, cô lập môi trường bằng mock, kiểm tra giải tích tính tay độc lập cho mô hình toán học. |
| **Tài liệu học thuật** | `docs/` | Lưu trữ chi tiết kiến trúc, mô hình toán học giải tích và báo cáo kết quả kiểm thử. |

---

## 5. Nguyên tắc Phối hợp giữa Weighted KNN và Gemini

Một nguyên tắc cốt lõi của đề tài DSS này là **sự tách bạch giữa phép tính định lượng và mô hình ngôn ngữ**:
1. **Gemini KHÔNG được tự tạo danh sách xe:** Mô hình ngôn ngữ lớn (LLM) dễ bị hiện tượng ảo giác (hallucination), bịa thông số kỹ thuật hoặc đưa ra các dòng xe không có thật trên thị trường.
2. **Gemini KHÔNG được thay đổi thứ tự xếp hạng:** Thứ tự Top-K xe, khoảng cách Euclid ($d_w$) và điểm số phù hợp (%) hoàn toàn do thuật toán toán học trong `recommender.py` tính toán trên dữ liệu thật.
3. **Gemini chỉ đóng vai trò:**
   - *Bộ biên dịch ngôn ngữ tự nhiên (NLP Parser):* Chuyển câu nói của người dùng thành cấu trúc tiêu chí JSON.
   - *Chuyên gia phân tích kết quả (Explainer):* Đọc bảng kết quả đã tính toán xong để giải thích lý do Top 1 được chọn và đưa ra lời khuyên trải nghiệm thực tế.

---

## 6. Các Quyết định Thiết kế Trọng yếu

1. **Khớp nối ma trận One-Hot cố định:**
   - Danh sách cột One-Hot được xây dựng từ tập dữ liệu sạch ban đầu (18 cột phân loại) và giữ cố định.
   - Khi người dùng truy vấn, vector truy vấn $q$ được đối chiếu chính xác với danh sách này, loại bỏ rủi ro lệch chiều đặc trưng giữa dữ liệu và truy vấn.
2. **Phân bổ trọng số đều theo số chiều One-Hot ($w_{g,j} = W_g / m$):**
   - Giúp các thuộc tính có nhiều lớp phân loại (như Hãng xe có 9 giá trị) không bị thổi phồng tầm ảnh hưởng so với các thuộc tính ít lớp (như Nhiên liệu có 2 giá trị).
3. **Lọc ngân sách cứng (Hard Budget Filter):**
   - Trong bài toán mua sắm tài sản lớn như xe máy, ngân sách là một ràng buộc cứng (hard constraint). Nếu một chiếc xe có giá 100 triệu, dù kiểu dáng hay công suất có hợp đến đâu cũng không thể gợi ý cho người chỉ có 40 triệu. Do đó, lọc cứng trước khi chạy KNN là quyết định mang tính thực tiễn cao.
4. **Bảo mật API Key nhiều lớp:**
   - Không hardcode API key trong mã nguồn.
   - Tự động nạp theo thứ tự ưu tiên: UI Password Input $\rightarrow$ Biến môi trường $\rightarrow$ File `keyapi.txt`.
   - Loại trừ `keyapi.txt` và file môi trường khỏi Git qua `.gitignore`.

---

## 7. Rủi ro, Giới hạn và Hướng Phát triển

### 7.1. Giới hạn hiện tại
- Dataset hiện tại chứa 225 mẫu xe thuộc các năm 2022 - 2024. Khi thị trường ra mắt các mẫu xe 2025 - 2026, cần cập nhật file CSV.
- Truy vấn hiện tại hỗ trợ một lựa chọn duy nhất cho mỗi thuộc tính phân loại (chưa hỗ trợ điều kiện OR phức hợp trong form lọc).

### 7.2. Hướng phát triển trong tương lai
- Bổ sung chỉ số chi phí lăn bánh theo từng tỉnh/thành phố (thuế trước bạ, phí cấp biển số).
- Bổ sung tiêu chí đánh giá trạm sạc điện theo tọa độ địa lý người dùng đối với xe máy điện.
- Hỗ trợ so sánh trực tiếp bảng thông số (side-by-side comparison) giữa 2 xe bất kỳ trong Top-K.

