"""
Hệ Thống Hỗ Trợ Ra Quyết Định (DSS) Gợi Ý Xe Máy
Kết hợp Thuật toán Định lượng Weighted KNN và Mô hình Ngôn ngữ Google Gemini AI.
Chạy trực tiếp với Streamlit: streamlit run app.py
"""

import html
import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from recommender import (
    ALL_GROUPS,
    CATEGORICAL_COLS,
    DEFAULT_PURPOSE,
    NUMERIC_COLS,
    PURPOSE_DESCRIPTIONS,
    PURPOSE_WEIGHTS,
    MotoRecommender,
    load_and_clean_data,
)
import gemini_helper as gh
import history as hist

# ---------------------------------------------------------------------------
# 1. Cấu hình trang Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gợi ý xe máy theo nhu cầu (DSS: KNN + Gemini)",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Khởi tạo recommender cache
@st.cache_resource
def get_recommender() -> Optional[MotoRecommender]:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        return MotoRecommender(csv_path)
    except Exception as e:
        st.error(f"Lỗi khởi tạo dữ liệu xe: {e}")
        return None


recommender = get_recommender()

# ---------------------------------------------------------------------------
# 2. Quản lý Session State
# ---------------------------------------------------------------------------
if "search_query" not in st.session_state:
    st.session_state.search_query = {}
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "current_purpose" not in st.session_state:
    st.session_state.current_purpose = DEFAULT_PURPOSE
if "gemini_explanation" not in st.session_state:
    st.session_state.gemini_explanation = None
if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = None


# ---------------------------------------------------------------------------
# 3. Hàm hiển thị Score Bar trực quan (an toàn XSS)
# ---------------------------------------------------------------------------
def show_score_bar(label: str, score: float) -> None:
    score_clamped = max(0.0, min(100.0, float(score)))
    if score_clamped >= 80.0:
        bar_color = "#2E7D32"  # Xanh lá đậm
        badge_bg = "#E8F5E9"
        badge_text = "#1B5E20"
    elif score_clamped >= 50.0:
        bar_color = "#F57F17"  # Vàng cam
        badge_bg = "#FFF8E1"
        badge_text = "#E65100"
    else:
        bar_color = "#D32F2F"  # Đỏ
        badge_bg = "#FFEBEE"
        badge_text = "#B71C1C"

    safe_label = html.escape(str(label))
    bar_html = f"""
    <div style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-weight: 600; font-size: 13px; color: #1E293B;">{safe_label}</span>
            <span style="background-color: {badge_bg}; color: {badge_text}; font-weight: bold; font-size: 12px; padding: 2px 8px; border-radius: 8px;">
                {score_clamped:.1f}%
            </span>
        </div>
        <div style="width: 100%; background-color: #E2E8F0; border-radius: 6px; height: 8px; overflow: hidden;">
            <div style="width: {score_clamped}%; background-color: {bar_color}; height: 8px; border-radius: 6px;"></div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Sidebar: Cấu hình hệ thống & Bộ trọng số DSS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛵 Motorcycle DSS")
    st.caption("Hệ Thống Hỗ Trợ Quyết Định Gợi Ý Xe Máy")

    st.markdown("---")
    st.subheader("🤖 Trạng Thái Trí Tuệ Nhân Tạo (Gemini)")
    model_status = gh.get_model_status()
    if model_status["ready"]:
        st.success(f"🟢 {model_status['message']}")
    else:
        st.info(f"⚪ {model_status['message']}")

    st.markdown("---")
    st.subheader("🎯 Hồ Sơ Mục Đích Sử Dụng")
    purposes_list = list(PURPOSE_WEIGHTS.keys())
    selected_purpose = st.selectbox(
        "Chọn hồ sơ nhu cầu:",
        options=purposes_list,
        index=purposes_list.index(st.session_state.current_purpose),
        key="sidebar_purpose_select",
    )
    st.session_state.current_purpose = selected_purpose

    purpose_desc = PURPOSE_DESCRIPTIONS.get(selected_purpose, "")
    st.info(f"💡 **Đặc trưng:** {purpose_desc}")

    st.markdown("##### ⚖️ Ma trận trọng số thuộc tính:")
    current_weights = PURPOSE_WEIGHTS[selected_purpose]
    sorted_weights = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)

    weights_df = pd.DataFrame(
        [{"Thuộc tính": k, "Trọng số": f"{int(v * 100)}%"} for k, v in sorted_weights]
    )
    st.dataframe(weights_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.caption("Khóa API Gemini được lưu trữ an toàn tại Server qua `keyapi.txt`.")


# ---------------------------------------------------------------------------
# 5. Khu vực Giao diện Chính
# ---------------------------------------------------------------------------
st.title("🛵 HỆ THỐNG HỖ TRỢ RA QUYẾT ĐỊNH GỢI Ý XE MÁY")
st.markdown(
    "Hệ thống kết hợp **Thuật toán Định lượng Weighted KNN** (lọc ngân sách, tính khoảng cách Euclid, xếp hạng minh bạch) "
    "và **Mô hình Ngôn ngữ Lớn Google Gemini AI** (bóc tách nhu cầu và giải thích kết quả)."
)

if recommender is None:
    st.error("❌ Không thể nạp dữ liệu từ `dataset.csv`. Vui lòng kiểm tra lại file dữ liệu.")
    st.stop()

# ---------------------------------------------------------------------------
# BƯỚC 1: Phân tích Nhu cầu Bằng Ngôn ngữ Tự nhiên (Gemini NLP)
# ---------------------------------------------------------------------------
st.markdown("### 1️⃣ Nhập Nhu Cầu Tự Nhiên (Gemini AI hỗ trợ)")
with st.container():
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        user_text = st.text_input(
            "Mô tả mong muốn của bạn bằng lời văn tự nhiên:",
            placeholder="Ví dụ: Tôi là sinh viên có khoảng 25 triệu cần mua xe số Honda đi học tiết kiệm xăng",
            key="natural_user_input",
        )
    with col_btn:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Phân tích AI", use_container_width=True)

    if analyze_btn and user_text.strip():
        with st.spinner("Gemini đang phân tích nhu cầu và trích xuất tiêu chí..."):
            extracted = gh.parse_need(user_text)
            st.session_state.extracted_info = extracted

            # Cập nhật purpose nếu AI phát hiện
            detected_purpose = extracted.get("purpose")
            if detected_purpose in PURPOSE_WEIGHTS:
                st.session_state.current_purpose = detected_purpose

            st.success("✅ Đã trích xuất xong tiêu chí! Dữ liệu đã được tự động điền vào form bên dưới.")

    if st.session_state.extracted_info:
        ext = st.session_state.extracted_info
        st.caption(
            f"ℹ️ **Kết quả trích xuất ({ext.get('source', 'gemini')}):** "
            f"Mục đích: `{ext.get('purpose')}` | Ngân sách: `{ext.get('max_budget') or ext.get('Giá(VND)')}` | "
            f"Hãng: `{ext.get('brand') or ext.get('Hãng')}` | Kiểu xe: `{ext.get('body_style') or ext.get('Kiểu xe')}` | "
            f"Nhiên liệu: `{ext.get('fuel') or ext.get('Nhiên liệu')}`"
        )

# ---------------------------------------------------------------------------
# BƯỚC 2: Form Tinh chỉnh Tiêu chí Kỹ thuật
# ---------------------------------------------------------------------------
st.markdown("### 2️⃣ Tinh Chỉnh Tiêu Chí Tìm Kiếm")

ext = st.session_state.extracted_info or {}

# Xác định giá trị mặc định cho form
default_budget = 0.0
raw_b = ext.get("max_budget") or ext.get("Giá(VND)")
if raw_b is not None:
    try:
        default_budget = float(raw_b) / 1_000_000.0
    except (ValueError, TypeError):
        default_budget = 0.0

brands_list = ["Không yêu cầu"] + recommender.unique_values("Hãng")
fuels_list = ["Không yêu cầu"] + recommender.unique_values("Nhiên liệu")
body_styles_list = ["Không yêu cầu"] + recommender.unique_values("Kiểu xe")

brand_default = ext.get("brand") or ext.get("Hãng") or "Không yêu cầu"
fuel_default = ext.get("fuel") or ext.get("Nhiên liệu") or "Không yêu cầu"
body_default = ext.get("body_style") or ext.get("Kiểu xe") or "Không yêu cầu"

with st.form("filter_criteria_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        budget_input_m = st.number_input(
            "Ngân sách tối đa (Triệu VNĐ):",
            min_value=0.0,
            max_value=1500.0,
            value=float(default_budget),
            step=1.0,
            help="Hệ thống sẽ lọc cứng những xe có giá <= mức ngân sách này.",
        )
        selected_brand = st.selectbox(
            "Hãng sản xuất:",
            options=brands_list,
            index=brands_list.index(brand_default) if brand_default in brands_list else 0,
        )
        target_year = st.selectbox(
            "Năm sản xuất:",
            options=["Không yêu cầu", 2024, 2023, 2022],
            index=0,
        )

    with col2:
        selected_fuel = st.selectbox(
            "Loại nhiên liệu:",
            options=fuels_list,
            index=fuels_list.index(fuel_default) if fuel_default in fuels_list else 0,
        )
        selected_body = st.selectbox(
            "Kiểu dáng xe:",
            options=body_styles_list,
            index=body_styles_list.index(body_default) if body_default in body_styles_list else 0,
        )
        min_trunk = st.number_input(
            "Dung tích cốp tối thiểu (Lít):",
            min_value=0.0,
            max_value=50.0,
            value=float(ext.get("min_trunk") or ext.get("Dung tích cốp (L)") or 0.0),
            step=1.0,
        )

    with col3:
        min_power = st.number_input(
            "Công suất động cơ mong muốn (kW):",
            min_value=0.0,
            max_value=150.0,
            value=float(ext.get("min_power") or ext.get("Công suất tối đa (Kw)") or 0.0),
            step=0.5,
        )
        k_results = st.slider(
            "Số lượng xe gợi ý (K):",
            min_value=1,
            max_value=15,
            value=int(ext.get("k", 5)),
        )
        mandatory_fuel_check = st.checkbox(
            "Ràng buộc bắt buộc nhiên liệu đã chọn",
            value=False,
            help="Nếu tích chọn, chỉ gợi ý đúng loại nhiên liệu bạn chọn, loại bỏ hoàn toàn loại khác.",
        )

    submit_btn = st.form_submit_button("🚀 Gợi Ý Xe Máy Tối Ưu", use_container_width=True)

# ---------------------------------------------------------------------------
# BƯỚC 3: Tính toán Weighted KNN & Hiển thị Kết quả
# ---------------------------------------------------------------------------
if submit_btn:
    # Xây dựng query dictionary
    budget_vnd = (budget_input_m * 1_000_000.0) if budget_input_m > 0 else None

    query_data: Dict[str, Any] = {
        "max_budget": budget_vnd,
        "is_budget_only": True,
        "prefer_economy": True,
        "Hãng": selected_brand if selected_brand != "Không yêu cầu" else None,
        "Nhiên liệu": selected_fuel if selected_fuel != "Không yêu cầu" else None,
        "Kiểu xe": selected_body if selected_body != "Không yêu cầu" else None,
        "Công suất tối đa (Kw)": min_power if min_power > 0 else None,
        "Dung tích cốp (L)": min_trunk if min_trunk > 0 else None,
        "Năm sản xuất": float(target_year) if target_year != "Không yêu cầu" else None,
    }
    if mandatory_fuel_check and selected_fuel != "Không yêu cầu":
        query_data["mandatory_fuel"] = selected_fuel

    st.session_state.search_query = query_data
    st.session_state.gemini_explanation = None

    active_weights = PURPOSE_WEIGHTS[st.session_state.current_purpose]

    with st.spinner("Đang tính toán ma trận khoảng cách Weighted KNN..."):
        try:
            results_df = recommender.recommend(
                query=query_data,
                weights=active_weights,
                k=k_results,
            )
            st.session_state.search_results = results_df

            # Lưu vào lịch sử tìm kiếm an toàn
            top_names = results_df["Tên xe"].tolist() if not results_df.empty else []
            hist.save_search(
                purpose=st.session_state.current_purpose,
                query=query_data,
                weights=active_weights,
                top_moto_names=top_names,
            )
        except Exception as e:
            st.error(f"Lỗi khi thực hiện thuật toán: {e}")
            st.session_state.search_results = None

# Hiển thị kết quả đã lưu trong Session State
if st.session_state.search_results is not None:
    results_df = st.session_state.search_results
    st.markdown("---")
    st.markdown("### 3️⃣ Bảng Xếp Hạng Kết Quả Gợi Ý (DSS Weighted KNN)")

    if results_df.empty:
        st.warning("⚠️ Không tìm thấy mẫu xe nào thỏa mãn mức ngân sách hoặc các tiêu chí bắt buộc của bạn.")
        st.info("💡 **Gợi ý:** Hãy thử nâng mức ngân sách hoặc nới lỏng các ràng buộc về hãng/nhiên liệu.")
    else:
        st.success(f"🎉 Đã tìm thấy **{len(results_df)}** mẫu xe phù hợp nhất theo hồ sơ **{st.session_state.current_purpose}**!")

        # Bảng tổng quan rút gọn
        view_cols = [
            "Thứ hạng", "Tên xe", "Hãng", "Giá(VND)", "Kiểu xe",
            "Nhiên liệu", "Dung tích cốp (L)", "Công suất tối đa (Kw)",
            "Năm sản xuất", "Độ phù hợp (%)", "Khoảng cách KNN"
        ]
        display_df = results_df[view_cols].copy()
        display_df["Giá(VND)"] = display_df["Giá(VND)"].apply(lambda x: f"{int(x):,} VNĐ".replace(",", "."))
        display_df["Độ phù hợp (%)"] = display_df["Độ phù hợp (%)"].apply(lambda x: f"{x:.1f}%")
        display_df["Năm sản xuất"] = display_df["Năm sản xuất"].astype(str)

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Thẻ chi tiết từng xe
        st.markdown("#### 🔍 Chi Tiết Mức Độ Đáp Ứng Từng Tiêu Chí")
        for _, row in results_df.iterrows():
            rank = row["Thứ hạng"]
            name = row["Tên xe"]
            score = row["Độ phù hợp (%)"]
            price_formatted = f"{int(row['Giá(VND)']):,} VNĐ".replace(",", ".")
            distance = row["Khoảng cách KNN"]

            with st.expander(f"🏅 Top {rank}: {name} — Giá: {price_formatted} — Điểm phù hợp: {score:.1f}%", expanded=(rank == 1)):
                col_info, col_breakdown = st.columns([1, 2])

                with col_info:
                    st.markdown(f"**Hãng:** {row['Hãng']}")
                    st.markdown(f"**Kiểu xe:** {row['Kiểu xe']}")
                    st.markdown(f"**Nhiên liệu:** {row['Nhiên liệu']}")
                    st.markdown(f"**Cốp xe:** {row['Dung tích cốp (L)']} L")
                    st.markdown(f"**Công suất:** {row['Công suất tối đa (Kw)']} kW")
                    st.markdown(f"**Năm sản xuất:** {int(row['Năm sản xuất'])}")
                    st.markdown(f"**Khoảng cách Euclid $d_w$:** `{distance:.4f}`")

                with col_breakdown:
                    show_score_bar("Điểm Phù Hợp Tương Đối (Score)", score)
                    st.markdown("##### Mức độ đáp ứng tiêu chí đã chọn:")
                    breakdowns = row["Chi tiết tiêu chí"]
                    if isinstance(breakdowns, list) and len(breakdowns) > 0:
                        for b in breakdowns:
                            if isinstance(b, dict):
                                msg = b.get("message", "")
                                is_good = b.get("is_good", True)
                                if is_good:
                                    st.markdown(f"<span style='color: #2E7D32;'>{msg}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color: #C62828;'>{msg}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"- {b}")
                    else:
                        st.caption("Xe thỏa mãn trần ngân sách cho phép.")

        # -------------------------------------------------------------------
        # BƯỚC 4: Giải thích & Lời khuyên Chuyên sâu từ Gemini AI
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 4️⃣ Lời Khuyên & Tư Vấn Chuyên Sâu từ Gemini AI")

        col_exp_btn, _ = st.columns([2, 3])
        with col_exp_btn:
            ask_ai_btn = st.button("💡 Nhận Tư Vấn Chuyên Sâu từ AI", use_container_width=True)

        if ask_ai_btn:
            with st.spinner("Gemini đang đọc dữ liệu bảng tính định lượng để viết báo cáo tư vấn..."):
                user_context = user_text if user_text.strip() else f"Tìm xe theo mục đích {st.session_state.current_purpose}"
                explanation, src = gh.explain_results(
                    top_candidates=results_df,
                    user_need=user_context,
                    purpose=st.session_state.current_purpose,
                )
                st.session_state.gemini_explanation = (explanation, src)

        if st.session_state.gemini_explanation:
            exp_text, src = st.session_state.gemini_explanation
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border-left: 5px solid #2563EB; padding: 16px 20px; border-radius: 6px; margin-top: 10px;">
                <div style="font-weight: 600; font-size: 14px; color: #1E40AF; margin-bottom: 8px;">
                    🤖 ĐÁNH GIÁ CHUYÊN MÔN ({src.upper()} ENGINE):
                </div>
                <div style="font-size: 14px; line-height: 1.6; color: #334155; white-space: pre-wrap;">{html.escape(exp_text)}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# BƯỚC 5: Lịch sử Tìm kiếm & Khám phá Dữ liệu
# ---------------------------------------------------------------------------
st.markdown("---")
col_hist, col_data = st.columns(2)

with col_hist:
    with st.expander("🕒 Xem Lịch Sử Tìm Kiếm"):
        history_records = hist.load_history()
        if not history_records:
            st.info("Chưa có lịch sử tìm kiếm nào.")
        else:
            st.markdown(f"Tổng số lượt tìm kiếm gần đây: **{len(history_records)}**")
            for h in reversed(history_records[-10:]):
                st.markdown(f"**Thời gian:** `{h.get('timestamp')}` | **Mục đích:** `{h.get('muc_dich')}`")
                st.caption(f"Top xe gợi ý: {', '.join(h.get('danh_sach_xe', []))}")
                st.markdown("---")
            if st.button("🗑️ Xóa sạch lịch sử tìm kiếm"):
                hist.clear_history()
                st.rerun()

with col_data:
    with st.expander("📊 Thống Kê Tập Dữ Liệu Thực Tế"):
        stats = recommender.get_stats()
        st.write(f"- **Tổng số mẫu xe:** {stats['total_rows']}")
        st.write(f"- **Phân bố Hãng xe:** {stats['brands_count']}")
        st.write(f"- **Phân bố Nhiên liệu:** {stats['fuels_count']}")
        st.write(f"- **Khoảng giá:** {int(stats['price_min']):,} - {int(stats['price_max']):,} VNĐ".replace(",", "."))
        st.write(f"- **Dung tích cốp:** {stats['trunk_min']}L - {stats['trunk_max']}L")
        st.write(f"- **Số chiều đặc trưng:** {stats['feature_dims']} chiều (MinMaxScaler + One-Hot)")
