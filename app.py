"""
Application: Hệ thống Hỗ trợ Ra Quyết định (DSS) Gợi ý Xe Máy
Kết hợp Thuật toán Định lượng Weighted KNN và Mô hình Ngôn ngữ Google Gemini.
"""

import html
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as pd_st
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
    page_title="Gợi ý xe máy theo nhu cầu (KNN + Gemini)",
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
# 3. Hàm hiển thị Score Bar an toàn
# ---------------------------------------------------------------------------
def show_score_bar(label: str, score: float) -> None:
    """Hiển thị thanh đo độ phù hợp với màu sắc tương ứng (an toàn XSS)."""
    score_clamped = max(0.0, min(100.0, float(score)))
    if score_clamped >= 80.0:
        bar_color = "#4CAF50"  # Xanh lá
        badge_bg = "#E8F5E9"
        badge_text = "#2E7D32"
    elif score_clamped >= 50.0:
        bar_color = "#FFC400"  # Vàng cam
        badge_bg = "#FFF8E1"
        badge_text = "#F57F17"
    else:
        bar_color = "#F44336"  # Đỏ
        badge_bg = "#FFEBEE"
        badge_text = "#C62828"

    safe_label = html.escape(str(label))
    bar_html = f"""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-weight: 500; font-size: 13px; color: #334155;">{safe_label}</span>
            <span style="background-color: {badge_bg}; color: {badge_text}; font-weight: bold; font-size: 12px; padding: 2px 8px; border-radius: 10px;">
                {score_clamped:.1f}%
            </span>
        </div>
        <div style="width: 100%; background-color: #E2E8F0; border-radius: 6px; height: 8px; overflow: hidden;">
            <div style="width: {score_clamped}%; background-color: {bar_color}; height: 8px; border-radius: 6px; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Sidebar: Cấu hình hệ thống & Bộ trọng số DSS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛵 Motorcycle DSS")
    st.caption("Hệ Hỗ Trợ Quyết Định Mua Xe Máy")

    st.markdown("---")
    st.subheader("🔑 Google Gemini API")
    user_api_key = st.text_input(
        "Nhập API Key (tùy chọn)",
        type="password",
        placeholder="AIzaSy...",
        help="Mặc định hệ thống tự đọc từ keyapi.txt hoặc biến môi trường GEMINI_API_KEY.",
    )

    # Hiển thị trạng thái kết nối Gemini
    model_status = gh.get_model_status(user_api_key)
    if model_status["ready"]:
        st.success(f"🟢 Gemini: {model_status['message']}")
    else:
        st.info(f"⚪ Gemini: {model_status['message']}")

    st.markdown("---")
    st.subheader("🎯 Mục đích sử dụng")
    
    purposes_list = list(PURPOSE_WEIGHTS.keys())
    selected_purpose = st.selectbox(
        "Chọn hồ sơ nhu cầu:",
        options=purposes_list,
        index=purposes_list.index(st.session_state.current_purpose),
        key="sidebar_purpose_select",
    )
    st.session_state.current_purpose = selected_purpose

    # Hiển thị mô tả mục đích
    purpose_desc = PURPOSE_DESCRIPTIONS.get(selected_purpose, "")
    st.info(f"💡 **Đặc trưng hồ sơ:** {purpose_desc}")

    # Bảng trọng số 8 thuộc tính
    st.markdown("##### ⚖️ Ma trận trọng số thuộc tính:")
    current_weights = PURPOSE_WEIGHTS[selected_purpose]
    sorted_weights = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)

    weights_df = pd.DataFrame(
        [{"Thuộc tính": k, "Trọng số": f"{int(v * 100)}%"} for k, v in sorted_weights]
    )
    st.dataframe(weights_df, hide_index=True, use_container_width=True)

    # Biểu đồ progress bars mức độ ưu tiên
    st.caption("Tỷ trọng đóng góp vào khoảng cách KNN:")
    for attr, w in sorted_weights:
        if w > 0:
            st.progress(w, text=f"{attr} ({int(w * 100)}%)")

    st.markdown("---")
    if recommender:
        stats = recommender.get_stats()
        with st.expander("📊 Thống kê Dataset thực tế"):
            st.write(f"• Tổng số mẫu xe: **{stats['total_rows']}**")
            st.write(f"• Số chiều đặc trưng: **{stats['feature_dims']}**")
            st.write(f"• Giá thấp nhất: **{int(stats['price_min']):,} VNĐ**".replace(",", "."))
            st.write(f"• Giá cao nhất: **{int(stats['price_max']):,} VNĐ**".replace(",", "."))
            st.write(f"• Cốp xe lớn nhất: **{stats['trunk_max']} L**")
            st.write(f"• Công suất lớn nhất: **{stats['power_max']} kW**")


# ---------------------------------------------------------------------------
# 5. Tiêu đề chính ứng dụng
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%); padding: 22px 28px; border-radius: 12px; margin-bottom: 25px; color: white;">
        <h2 style="margin: 0; font-size: 26px; color: white;">Hệ Hỗ Trợ Ra Quyết Định Gợi Ý Xe Máy</h2>
        <p style="margin: 6px 0 0 0; opacity: 0.92; font-size: 15px;">
            Mô hình kết hợp <strong>Weighted K-Nearest Neighbors (KNN)</strong> định lượng và 
            <strong>Google Gemini AI</strong> phân tích nhu cầu tự nhiên & giải thích chuyên sâu.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if recommender is None:
    st.error("⚠️ Không tìm thấy file `dataset.csv` hoặc dữ liệu bị lỗi. Vui lòng kiểm tra lại thư mục!")
    st.stop()

# ---------------------------------------------------------------------------
# 6. Khu vực 1: Phân tích nhu cầu tự nhiên bằng Gemini
# ---------------------------------------------------------------------------
st.subheader("✨ Phân tích nhu cầu tự nhiên bằng AI")
st.markdown(
    "Nhập câu mô tả nhu cầu thực tế của bạn bằng ngôn ngữ tự nhiên. "
    "Hệ thống sẽ tự động bóc tách các tiêu chí kỹ thuật và áp dụng vào bộ lọc."
)

col_nlp1, col_nlp2 = st.columns([4, 1])
with col_nlp1:
    user_prompt_text = st.text_area(
        "Mô tả nhu cầu:",
        placeholder="Ví dụ: Em là sinh viên cần tìm xe tay ga hoặc xe điện đi học, cốp rộng tầm 25 lít đựng vừa balo, giá dưới 45 triệu.",
        height=85,
        label_visibility="collapsed",
    )
with col_nlp2:
    st.write("")
    analyze_btn = st.button("✨ Phân tích nhu cầu", use_container_width=True, type="primary")

if analyze_btn:
    if not user_prompt_text.strip():
        st.warning("Vui lòng nhập câu mô tả nhu cầu của bạn.")
    else:
        with st.spinner("Đang phân tích nhu cầu bằng AI..."):
            options_dict = {
                "Hãng": recommender.unique_values("Hãng"),
                "Nhiên liệu": recommender.unique_values("Nhiên liệu"),
                "Kiểu xe": recommender.unique_values("Kiểu xe"),
                "Tình trạng": recommender.unique_values("Tình trạng"),
            }
            parsed = gh.parse_need(user_prompt_text, options_dict, api_key=user_api_key)
            st.session_state.extracted_info = parsed
            
            # Cập nhật mục đích
            if parsed.get("purpose") in PURPOSE_WEIGHTS:
                st.session_state.current_purpose = parsed["purpose"]

if st.session_state.extracted_info:
    info = st.session_state.extracted_info
    src_badge = info.get("source", "fallback")
    st.success(f"✅ Đã phân tích thành công (Nguồn: **{src_badge}**)")
    
    # Hiển thị các tiêu chí đã bóc tách
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    with col_t1:
        st.metric("🎯 Mục đích", info.get("purpose") or "Mặc định")
    with col_t2:
        b_val = f"{info.get('budget_trieu')} tr" if info.get("budget_trieu") else "Không rõ"
        st.metric("💰 Ngân sách tối đa", b_val)
    with col_t3:
        trunk_val = f"{info.get('trunk_liters')} L" if info.get("trunk_liters") else "Không rõ"
        if info.get("inferred_trunk"):
            trunk_val += " (gợi ý)"
        st.metric("🎒 Dung tích cốp", trunk_val)
    with col_t4:
        st.metric("🛵 Kiểu xe", info.get("body") or "Không yêu cầu")
    with col_t5:
        st.metric("⛽ Nhiên liệu", info.get("fuel") or "Không yêu cầu")

st.markdown("---")

# ---------------------------------------------------------------------------
# 7. Khu vực 2: Form chọn tiêu chí tìm kiếm định lượng
# ---------------------------------------------------------------------------
st.subheader("🔎 Tiêu chí tìm xe chi tiết")
st.caption("Các tiêu chí chưa bật sẽ không bị ép buộc, hệ thống sẽ tự động tối ưu theo khoảng cách trung tính.")

with st.form("search_form"):
    # Hàng 1: Ngân sách, Cốp, Công suất
    col_c1, col_c2, col_c3 = st.columns(3)
    
    # Giá trị mặc định nếu có từ AI
    ext = st.session_state.extracted_info or {}
    default_budget_active = ext.get("budget_trieu") is not None
    default_budget_val = float(ext.get("budget_trieu") or 50.0)

    default_trunk_active = ext.get("trunk_liters") is not None
    default_trunk_val = float(ext.get("trunk_liters") or 20.0)

    default_power_active = ext.get("power_kw") is not None
    default_power_val = float(ext.get("power_kw") or 8.0)

    with col_c1:
        use_budget = st.checkbox("Áp dụng Ngân sách tối đa", value=default_budget_active)
        budget_slider = st.slider(
            "Ngân sách tối đa (Triệu VNĐ):",
            min_value=15.0,
            max_value=300.0,
            value=min(300.0, max(15.0, default_budget_val)),
            step=1.0,
            disabled=not use_budget,
            help="Hệ thống sẽ lọc cứng loại bỏ các xe vượt quá mức giá này.",
        )

    with col_c2:
        use_trunk = st.checkbox("Yêu cầu Dung tích cốp", value=default_trunk_active)
        trunk_slider = st.slider(
            "Dung tích cốp mong muốn (Lít):",
            min_value=0.0,
            max_value=50.0,
            value=min(50.0, max(0.0, default_trunk_val)),
            step=1.0,
            disabled=not use_trunk,
        )

    with col_c3:
        use_power = st.checkbox("Yêu cầu Công suất động cơ", value=default_power_active)
        power_slider = st.slider(
            "Công suất tối đa (Kw):",
            min_value=0.5,
            max_value=100.0,
            value=min(100.0, max(0.5, default_power_val)),
            step=0.5,
            disabled=not use_power,
        )

    # Hàng 2: Hãng, Kiểu xe, Nhiên liệu, Năm sản xuất, Tình trạng
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    all_brands = ["Không yêu cầu"] + recommender.unique_values("Hãng")
    default_brand_idx = 0
    if ext.get("brand") in all_brands:
        default_brand_idx = all_brands.index(ext.get("brand"))

    with col_f1:
        selected_brand = st.selectbox("Hãng xe:", options=all_brands, index=default_brand_idx)

    all_bodies = ["Không yêu cầu"] + recommender.unique_values("Kiểu xe")
    default_body_idx = 0
    if ext.get("body") in all_bodies:
        default_body_idx = all_bodies.index(ext.get("body"))

    with col_f2:
        selected_body = st.selectbox("Kiểu xe:", options=all_bodies, index=default_body_idx)

    all_fuels = ["Không yêu cầu"] + recommender.unique_values("Nhiên liệu")
    default_fuel_idx = 0
    if ext.get("fuel") in all_fuels:
        default_fuel_idx = all_fuels.index(ext.get("fuel"))

    with col_f3:
        selected_fuel = st.selectbox("Nhiên liệu:", options=all_fuels, index=default_fuel_idx)

    all_years = ["Không yêu cầu"] + [str(y) for y in sorted(recommender.df["Năm sản xuất"].astype(int).unique(), reverse=True)]
    with col_f4:
        selected_year = st.selectbox("Năm sản xuất:", options=all_years, index=0)

    # Hàng 3: Số lượng K kết quả
    col_k1, col_k2 = st.columns([2, 2])
    with col_k1:
        k_results = st.slider("Số lượng xe gợi ý (K):", min_value=1, max_value=20, value=5)
    with col_k2:
        st.write("")
        st.write("")
        submit_search = st.form_submit_button("🔍 Tìm xe máy phù hợp", use_container_width=True, type="primary")

if submit_search:
    # Xây dựng vector truy vấn query dict
    active_query = {}
    if use_budget:
        active_query["Giá(VND)"] = budget_slider * 1_000_000.0
    if use_trunk:
        active_query["Dung tích cốp (L)"] = trunk_slider
    if use_power:
        active_query["Công suất tối đa (Kw)"] = power_slider
    if selected_brand != "Không yêu cầu":
        active_query["Hãng"] = selected_brand
    if selected_body != "Không yêu cầu":
        active_query["Kiểu xe"] = selected_body
    if selected_fuel != "Không yêu cầu":
        active_query["Nhiên liệu"] = selected_fuel
    if selected_year != "Không yêu cầu":
        active_query["Năm sản xuất"] = float(selected_year)
    
    st.session_state.search_query = active_query
    
    # Thực thi thuật toán Weighted KNN
    weights_to_use = PURPOSE_WEIGHTS[st.session_state.current_purpose]
    with st.spinner("Đang tính toán khoảng cách Weighted KNN..."):
        results_df = recommender.recommend(
            query=active_query,
            weights=weights_to_use,
            k=k_results,
        )
        st.session_state.search_results = results_df
        st.session_state.gemini_explanation = None  # Reset giải thích cũ

        # Lưu lịch sử tìm kiếm an toàn
        if not results_df.empty:
            top_names = results_df["Tên xe"].tolist()
            hist.save_search(
                purpose=st.session_state.current_purpose,
                query=active_query,
                weights=weights_to_use,
                top_moto_names=top_names,
            )

# ---------------------------------------------------------------------------
# 8. Khu vực 3: Bảng kết quả và Giải thích chi tiết
# ---------------------------------------------------------------------------
if st.session_state.search_results is not None:
    results_df = st.session_state.search_results
    st.markdown("---")
    st.subheader("🏆 Kết quả gợi ý (Weighted KNN Ranking)")

    if results_df.empty:
        st.warning(
            "⚠️ Không tìm thấy mẫu xe nào thỏa mãn mức ngân sách hoặc bộ tiêu chí cứng đã chọn!\n\n"
            "**Gợi ý xử lý:**\n"
            "- Hãy thử nâng mức ngân sách tối đa.\n"
            "- Bỏ bớt các điều kiện lọc Hãng hoặc Kiểu xe để thuật toán tìm kiếm trên không gian rộng hơn."
        )
    else:
        # Bảng tóm tắt kết quả
        display_df = results_df[[
            "Thứ hạng", "Tên xe", "Hãng", "Giá(VND)", "Kiểu xe",
            "Nhiên liệu", "Dung tích cốp (L)", "Công suất tối đa (Kw)",
            "Năm sản xuất", "Độ phù hợp (%)"
        ]].copy()

        # Định dạng cột hiển thị
        display_df["Giá (VNĐ)"] = display_df["Giá(VND)"].apply(lambda x: f"{int(x):,} đ".replace(",", "."))
        display_df["Cốp (L)"] = display_df["Dung tích cốp (L)"].apply(lambda x: f"{x} L")
        display_df["Công suất (kW)"] = display_df["Công suất tối đa (Kw)"].apply(lambda x: f"{x} kW")
        display_df["Điểm phù hợp"] = display_df["Độ phù hợp (%)"].apply(lambda x: f"{x:.1f}%")

        table_cols = [
            "Thứ hạng", "Tên xe", "Hãng", "Giá (VNĐ)", "Kiểu xe",
            "Nhiên liệu", "Cốp (L)", "Công suất (kW)", "Năm sản xuất", "Điểm phù hợp"
        ]
        st.dataframe(
            display_df[table_cols],
            hide_index=True,
            use_container_width=True,
        )

        st.caption("ℹ️ *Điểm phù hợp (%) là điểm tương đối quy đổi từ khoảng cách Weighted Euclidean Distance: Score = round(1 / (1 + d_w) * 100, 1).*")

        # Nút yêu cầu Gemini giải thích
        st.markdown("#### 🤖 Tư vấn chuyên sâu")
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.write("Nhận bản tóm tắt phân tích DSS, so sánh Top 1 và lời khuyên thực tế:")
        with col_exp2:
            explain_btn = st.button("🤖 Nhờ Gemini tư vấn", use_container_width=True)

        if explain_btn:
            with st.spinner("Gemini đang phân tích và tạo lời khuyên chuyên sâu..."):
                weights_to_use = PURPOSE_WEIGHTS[st.session_state.current_purpose]
                explanation = gh.explain_results(
                    purpose=st.session_state.current_purpose,
                    weights=weights_to_use,
                    query=st.session_state.search_query,
                    result=results_df,
                    api_key=user_api_key,
                )
                st.session_state.gemini_explanation = explanation

        if st.session_state.gemini_explanation:
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border-left: 4px solid #0284C7; padding: 16px 20px; border-radius: 6px; margin: 15px 0;">
                    <h5 style="color: #0369A1; margin-top: 0;">💡 Phân tích & Lời khuyên của Chuyên gia DSS:</h5>
                    <div style="color: #1E293B; line-height: 1.6; font-size: 14.5px;">
                        {st.session_state.gemini_explanation.replace(chr(10), '<br>')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Chi tiết từng mẫu xe (Expanders)
        st.markdown("#### 📋 Phân tích tiêu chí chi tiết từng xe:")
        for idx, row in results_df.iterrows():
            is_top1 = (row["Thứ hạng"] == 1)
            header_title = (
                f"Top {row['Thứ hạng']}: {row['Tên xe']} — "
                f"Độ phù hợp: {row['Độ phù hợp (%)']}% "
                f"(Khoảng cách KNN: {row['Khoảng cách KNN']})"
            )
            with st.expander(header_title, expanded=is_top1):
                col_spec1, col_spec2 = st.columns([1, 1])
                with col_spec1:
                    st.markdown("**Thông số kỹ thuật:**")
                    st.write(f"• **Hãng:** {row['Hãng']}")
                    st.write(f"• **Giá:** {int(row['Giá(VND)']):,} VNĐ".replace(",", "."))
                    st.write(f"• **Kiểu xe:** {row['Kiểu xe']}")
                    st.write(f"• **Nhiên liệu:** {row['Nhiên liệu']}")
                    st.write(f"• **Dung tích cốp:** {row['Dung tích cốp (L)']} Lít")
                    st.write(f"• **Công suất:** {row['Công suất tối đa (Kw)']} kW")
                    st.write(f"• **Năm sản xuất:** {row['Năm sản xuất']}")

                with col_spec2:
                    st.markdown("**Mức độ đáp ứng tiêu chí:**")
                    breakdown = row["Chi tiết tiêu chí"]
                    if not breakdown:
                        st.write("*(Không có tiêu chí cụ thể nào được nhập, xe được xếp hạng theo hồ sơ trọng số)*")
                    else:
                        for crit in breakdown:
                            show_score_bar(f"{crit['column']}", crit["match_pct"])
                            if crit["is_good"]:
                                st.markdown(f"<span style='color: #2E7D32; font-size: 13px;'>{crit['message']}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color: #C62828; font-size: 13px;'>{crit['message']}</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 9. Khu vực 4: Lịch sử tìm kiếm
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🕒 Lịch sử tìm kiếm gần đây")

history_list = hist.load_history()
if not history_list:
    st.info("Chưa có lượt tìm kiếm nào được lưu.")
else:
    col_h_table, col_h_clear = st.columns([4, 1])
    with col_h_clear:
        if st.button("🗑️ Xóa toàn bộ lịch sử", use_container_width=True):
            hist.clear_history()
            st.rerun()

    with col_h_table:
        recent_10 = list(reversed(history_list[-10:]))
        formatted_history = []
        for r in recent_10:
            top_bikes = ", ".join(r.get("danh_sach_xe", [])[:3])
            query_items = []
            for k, v in r.get("tieu_chi", {}).items():
                if k == "Giá(VND)":
                    query_items.append(f"Giá <= {int(v):,} đ".replace(",", "."))
                else:
                    query_items.append(f"{k}: {v}")
            q_str = "; ".join(query_items) if query_items else "Theo mục đích"

            formatted_history.append({
                "Thời gian": r.get("timestamp"),
                "Mục đích": r.get("muc_dich"),
                "Tiêu chí lọc": q_str,
                "Top gợi ý": top_bikes,
            })

        st.dataframe(pd.DataFrame(formatted_history), hide_index=True, use_container_width=True)

