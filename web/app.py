"""
Streamlit Web Presentation Layer for Motorcycle DSS.
Communicates with the FastAPI backend via REST API (Client-Server architecture).
"""

import html
import os
import sys
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# Cấu hình URL của Backend API (có thể override qua biến môi trường API_URL)
API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Gợi ý xe máy theo nhu cầu (DSS: KNN + Gemini)",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1. Các hàm gọi API Backend
# ---------------------------------------------------------------------------
def api_get_health() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_get_model_info() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_recommend(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{API_BASE_URL}/recommend", json=payload, timeout=25)
        if r.status_code == 200:
            return r.json()
        st.error(f"Lỗi API ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Không thể kết nối đến Backend API ({API_BASE_URL}): {e}")
    return None


def api_interpret(text: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{API_BASE_URL}/interpret", json={"text": text}, timeout=20)
        if r.status_code == 200:
            return r.json()
        st.error(f"Lỗi diễn giải ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Không thể kết nối đến Backend API: {e}")
    return None


def api_explain(purpose: str, results: List[Dict[str, Any]], query: Dict[str, Any]) -> Optional[str]:
    try:
        payload = {"purpose": purpose, "results": results, "query": query}
        r = requests.post(f"{API_BASE_URL}/explain", json=payload, timeout=25)
        if r.status_code == 200:
            return r.json().get("explanation")
        st.error(f"Lỗi tạo giải thích: {r.text}")
    except Exception as e:
        st.error(f"Không thể kết nối đến Backend API: {e}")
    return None


# ---------------------------------------------------------------------------
# 2. Quản lý Session State
# ---------------------------------------------------------------------------
if "search_query" not in st.session_state:
    st.session_state.search_query = {}
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "current_purpose" not in st.session_state:
    st.session_state.current_purpose = "Đi làm, đi phố & Gia đình"
if "gemini_explanation" not in st.session_state:
    st.session_state.gemini_explanation = None
if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = None
if "history_list" not in st.session_state:
    st.session_state.history_list = []


# ---------------------------------------------------------------------------
# 3. Hiển thị Score Bar an toàn
# ---------------------------------------------------------------------------
def show_score_bar(label: str, score: float) -> None:
    score_clamped = max(0.0, min(100.0, float(score)))
    if score_clamped >= 80.0:
        bar_color = "#4CAF50"
        badge_bg = "#E8F5E9"
        badge_text = "#2E7D32"
    elif score_clamped >= 50.0:
        bar_color = "#FFC400"
        badge_bg = "#FFF8E1"
        badge_text = "#F57F17"
    else:
        bar_color = "#F44336"
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
            <div style="width: {score_clamped}%; background-color: {bar_color}; height: 8px; border-radius: 6px;"></div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Kiểm tra trạng thái Backend API
# ---------------------------------------------------------------------------
health_info = api_get_health()
model_info = api_get_model_info() if health_info else None

# ---------------------------------------------------------------------------
# 5. Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛵 Motorcycle DSS")
    st.caption("Kiến trúc Client - Server (FastAPI + Streamlit)")

    st.markdown("---")
    st.subheader("🔌 Trạng thái Hệ thống")
    if health_info:
        st.success(f"🟢 Backend API: Sẵn sàng ({health_info.get('total_motorcycles', 0)} mẫu xe)")
    else:
        st.error(f"🔴 Backend API ({API_BASE_URL}): Chưa kết nối")
        st.info("💡 Hãy đảm bảo FastAPI đang chạy bằng lệnh:\n`uvicorn api.main:app --port 8000`")

    st.markdown("---")
    st.subheader("🎯 Mục đích sử dụng")

    fallback_purposes = [
        "Đi làm, đi phố & Gia đình",
        "Học sinh, Sinh viên",
        "Chạy dịch vụ (Grab/Be/Giao hàng)",
        "Đi phượt, Thể thao & Đam mê tốc độ",
        "Thời trang, Sang trọng & Cao cấp"
    ]
    purposes_list = model_info.get("purpose_profiles", fallback_purposes) if model_info else fallback_purposes

    curr_idx = purposes_list.index(st.session_state.current_purpose) if st.session_state.current_purpose in purposes_list else 0
    selected_purpose = st.selectbox(
        "Chọn hồ sơ nhu cầu:",
        options=purposes_list,
        index=curr_idx,
        key="sidebar_purpose_select",
    )
    st.session_state.current_purpose = selected_purpose

    # Trọng số hồ sơ
    purpose_weights_map = {
        "Đi làm, đi phố & Gia đình": {"Cốp xe (L)": 0.25, "Giá xe": 0.20, "Kiểu xe": 0.20, "Nhiên liệu": 0.15, "Hãng xe": 0.10, "Công suất": 0.05, "Năm SX": 0.05},
        "Học sinh, Sinh viên": {"Giá xe": 0.35, "Kiểu xe": 0.20, "Cốp xe (L)": 0.15, "Nhiên liệu": 0.15, "Hãng xe": 0.05, "Công suất": 0.05, "Năm SX": 0.05},
        "Chạy dịch vụ (Grab/Be/Giao hàng)": {"Giá xe": 0.30, "Nhiên liệu": 0.20, "Công suất": 0.15, "Kiểu xe": 0.15, "Cốp xe (L)": 0.10, "Hãng xe": 0.05, "Năm SX": 0.05},
        "Đi phượt, Thể thao & Đam mê tốc độ": {"Công suất": 0.35, "Kiểu xe": 0.25, "Giá xe": 0.10, "Nhiên liệu": 0.10, "Hãng xe": 0.10, "Cốp xe (L)": 0.05, "Năm SX": 0.05},
        "Thời trang, Sang trọng & Cao cấp": {"Hãng xe": 0.25, "Kiểu xe": 0.25, "Cốp xe (L)": 0.15, "Giá xe": 0.10, "Công suất": 0.10, "Năm SX": 0.10, "Nhiên liệu": 0.05},
    }
    cur_weights = purpose_weights_map.get(selected_purpose, {})

    st.markdown("##### ⚖️ Trọng số ưu tiên:")
    for attr, w in sorted(cur_weights.items(), key=lambda x: x[1], reverse=True):
        st.progress(w, text=f"{attr} ({int(w * 100)}%)")

    if model_info:
        with st.expander("📊 Metadata Mô hình (Từ API)"):
            st.write(f"• Tên mô hình: **{model_info.get('model_name')}**")
            st.write(f"• Phiên bản: **{model_info.get('version')}**")
            st.write(f"• Số chiều đặc trưng: **{model_info.get('feature_dimensions')}**")
            p_stats = model_info.get("price_stats", {})
            st.write(f"• Giá min: **{int(p_stats.get('min_vnd', 0)):,} đ**".replace(",", "."))
            st.write(f"• Giá max: **{int(p_stats.get('max_vnd', 0)):,} đ**".replace(",", "."))


# ---------------------------------------------------------------------------
# 6. Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%); padding: 22px 28px; border-radius: 12px; margin-bottom: 25px; color: white;">
        <h2 style="margin: 0; font-size: 26px; color: white;">Hệ Hỗ Trợ Ra Quyết Định Gợi Ý Xe Máy</h2>
        <p style="margin: 6px 0 0 0; opacity: 0.92; font-size: 15px;">
            Mô hình kết hợp <strong>Weighted KNN</strong> định lượng & <strong>Google Gemini AI</strong> phân tích nhu cầu tự nhiên qua <strong>FastAPI</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 7. Khu vực 1: Diễn giải nhu cầu tự nhiên bằng Gemini
# ---------------------------------------------------------------------------
st.subheader("✨ Phân tích nhu cầu tự nhiên bằng AI")
st.markdown("Nhập nhu cầu tìm xe bằng tiếng Việt để hệ thống phân tích thành tiêu chí chuẩn:")

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
    analyze_btn = st.button("✨ Phân tích AI", use_container_width=True, type="primary")

if analyze_btn:
    if not user_prompt_text.strip():
        st.warning("Vui lòng nhập câu mô tả nhu cầu của bạn.")
    else:
        with st.spinner("Đang gửi yêu cầu đến Backend API để phân tích..."):
            parsed = api_interpret(user_prompt_text)
            if parsed:
                st.session_state.extracted_info = parsed
                if parsed.get("purpose") in purposes_list:
                    st.session_state.current_purpose = parsed["purpose"]

if st.session_state.extracted_info:
    info = st.session_state.extracted_info
    src_badge = info.get("source", "fallback")
    st.success(f"✅ Đã phân tích thành công (Nguồn: **{src_badge}**)")

    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    with col_t1:
        st.metric("🎯 Mục đích", info.get("purpose") or "Mặc định")
    with col_t2:
        b_val = f"{info.get('budget_trieu')} tr" if info.get("budget_trieu") else "Không rõ"
        st.metric("💰 Ngân sách trần", b_val)
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
# 8. Khu vực 2: Form chọn tiêu chí tìm xe chi tiết
# ---------------------------------------------------------------------------
st.subheader("🔎 Tiêu chí tìm xe chi tiết")

categories_map = model_info.get("valid_categories", {}) if model_info else {}
brand_options = ["Không yêu cầu"] + categories_map.get("Hãng", ["Honda", "Yamaha", "VinFast", "Vespa", "SYM", "Suzuki", "Piaggio", "Yadea", "Dat Bike"])
body_options = ["Không yêu cầu"] + categories_map.get("Kiểu xe", ["Tay ga", "Xe số", "Xe côn tay", "Xe điện", "Tay ga (50cc)", "Xe số (50cc)"])
fuel_options = ["Không yêu cầu"] + categories_map.get("Nhiên liệu", ["Xăng", "Điện"])

ext = st.session_state.extracted_info or {}

with st.form("search_form"):
    col_c1, col_c2, col_c3 = st.columns(3)

    default_budget_active = ext.get("budget_trieu") is not None
    default_budget_val = float(ext.get("budget_trieu") or 50.0)

    default_trunk_active = ext.get("trunk_liters") is not None
    default_trunk_val = float(ext.get("trunk_liters") or 20.0)

    default_power_active = ext.get("power_kw") is not None
    default_power_val = float(ext.get("power_kw") or 8.0)

    with col_c1:
        use_budget = st.checkbox("Áp dụng Ngân sách tối đa (Lọc trần)", value=default_budget_active)
        budget_slider = st.slider(
            "Ngân sách tối đa (Triệu VNĐ):",
            min_value=15.0,
            max_value=300.0,
            value=min(300.0, max(15.0, default_budget_val)),
            step=1.0,
            disabled=not use_budget,
            help="Hệ thống lọc bỏ hoàn toàn các xe có giá cao hơn mức này. Không phạt xe rẻ hơn.",
        )
        prefer_economy = st.checkbox("Ưu tiên xe tiết kiệm hơn trong ngân sách", value=True)

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

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    default_brand_idx = brand_options.index(ext.get("brand")) if ext.get("brand") in brand_options else 0
    with col_f1:
        selected_brand = st.selectbox("Hãng xe:", options=brand_options, index=default_brand_idx)

    default_body_idx = body_options.index(ext.get("body")) if ext.get("body") in body_options else 0
    with col_f2:
        selected_body = st.selectbox("Kiểu xe:", options=body_options, index=default_body_idx)

    default_fuel_idx = fuel_options.index(ext.get("fuel")) if ext.get("fuel") in fuel_options else 0
    with col_f3:
        selected_fuel = st.selectbox("Nhiên liệu:", options=fuel_options, index=default_fuel_idx)

    year_options = ["Không yêu cầu", "2024", "2023", "2022"]
    with col_f4:
        selected_year = st.selectbox("Năm sản xuất:", options=year_options, index=0)

    col_k1, col_k2 = st.columns([2, 2])
    with col_k1:
        k_results = st.slider("Số lượng xe gợi ý (K):", min_value=1, max_value=20, value=5)
    with col_k2:
        st.write("")
        st.write("")
        submit_btn = st.form_submit_button("🔍 Tìm xe máy phù hợp", use_container_width=True, type="primary")

if submit_btn:
    payload = {
        "purpose": st.session_state.current_purpose,
        "max_budget": (budget_slider * 1_000_000.0) if use_budget else None,
        "prefer_economy": prefer_economy,
        "trunk_liters": trunk_slider if use_trunk else None,
        "power_kw": power_slider if use_power else None,
        "brand": selected_brand if selected_brand != "Không yêu cầu" else None,
        "body": selected_body if selected_body != "Không yêu cầu" else None,
        "fuel": selected_fuel if selected_fuel != "Không yêu cầu" else None,
        "year": int(selected_year) if selected_year != "Không yêu cầu" else None,
        "k": k_results,
        "with_explanation": False,
    }

    st.session_state.search_query = payload

    with st.spinner("Đang gửi yêu cầu đến FastAPI Backend để tính toán Weighted KNN..."):
        resp = api_recommend(payload)
        if resp:
            st.session_state.search_results = resp.get("results", [])
            st.session_state.gemini_explanation = None

            # Lưu vào lịch sử cục bộ
            top_names = [item["name"] for item in resp.get("results", [])]
            st.session_state.history_list.append({
                "purpose": st.session_state.current_purpose,
                "filters": payload,
                "top_names": top_names,
            })

# ---------------------------------------------------------------------------
# 9. Khu vực 3: Bảng kết quả và Giải thích
# ---------------------------------------------------------------------------
if st.session_state.search_results is not None:
    results_list = st.session_state.search_results
    st.markdown("---")
    st.subheader("🏆 Kết quả gợi ý (Weighted KNN)")

    if len(results_list) == 0:
        st.warning("⚠️ Không tìm thấy mẫu xe nào thỏa mãn mức ngân sách hoặc bộ tiêu chí cứng đã chọn.")
    else:
        # Bảng dữ liệu tóm tắt
        import pandas as pd
        table_rows = []
        for r in results_list:
            table_rows.append({
                "Thứ hạng": r["rank"],
                "Tên xe": r["name"],
                "Hãng": r["brand"],
                "Giá": r["price_formatted"],
                "Kiểu xe": r["body"],
                "Nhiên liệu": r["fuel"],
                "Cốp (L)": f"{r['trunk_liters']} L",
                "Công suất": f"{r['power_kw']} kW",
                "Năm SX": r["year"],
                "Độ phù hợp": f"{r['fit_score']:.1f}%",
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        st.caption("ℹ️ *Điểm phù hợp (%) là điểm tương đối quy đổi từ khoảng cách Weighted Euclidean: Score = round(1 / (1 + d_w) * 100, 1).*")

        # Nút yêu cầu tư vấn Gemini
        st.markdown("#### 🤖 Tư vấn chuyên sâu")
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.write("Nhận bản tóm tắt phân tích DSS, so sánh Top 1 và lời khuyên thực tế từ AI:")
        with col_exp2:
            explain_btn = st.button("🤖 Nhờ Gemini tư vấn", use_container_width=True)

        if explain_btn:
            with st.spinner("Đang gửi yêu cầu đến Backend API để tạo giải thích..."):
                exp_text = api_explain(
                    purpose=st.session_state.current_purpose,
                    results=results_list,
                    query=st.session_state.search_query
                )
                if exp_text:
                    st.session_state.gemini_explanation = exp_text

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

        # Chi tiết từng mẫu xe
        st.markdown("#### 📋 Phân tích tiêu chí chi tiết từng xe:")
        for idx, item in enumerate(results_list):
            is_top1 = (item["rank"] == 1)
            header_title = f"Top {item['rank']}: {item['name']} — Độ phù hợp: {item['fit_score']:.1f}% (Khoảng cách KNN: {item['knn_distance']})"
            with st.expander(header_title, expanded=is_top1):
                col_spec1, col_spec2 = st.columns([1, 1])
                with col_spec1:
                    st.markdown("**Thông số kỹ thuật:**")
                    st.write(f"• **Hãng:** {item['brand']}")
                    st.write(f"• **Giá:** {item['price_formatted']}")
                    st.write(f"• **Kiểu xe:** {item['body']}")
                    st.write(f"• **Nhiên liệu:** {item['fuel']}")
                    st.write(f"• **Dung tích cốp:** {item['trunk_liters']} Lít")
                    st.write(f"• **Công suất:** {item['power_kw']} kW")
                    st.write(f"• **Năm sản xuất:** {item['year']}")

                with col_spec2:
                    st.markdown("**Mức độ đáp ứng tiêu chí:**")
                    breakdown = item.get("criteria_breakdown", [])
                    if not breakdown:
                        st.write("*(Được xếp hạng tự động theo hồ sơ trọng số)*")
                    else:
                        for crit in breakdown:
                            show_score_bar(f"{crit['column']}", crit["match_pct"])
                            if crit["is_good"]:
                                st.markdown(f"<span style='color: #2E7D32; font-size: 13px;'>{crit['message']}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color: #C62828; font-size: 13px;'>{crit['message']}</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 10. Khu vực 4: Lịch sử tìm kiếm
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🕒 Lịch sử tìm kiếm gần đây (Phiên hiện tại)")
if not st.session_state.history_list:
    st.info("Chưa có lượt tìm kiếm nào trong phiên làm việc này.")
else:
    hist_table = []
    for h in reversed(st.session_state.history_list[-10:]):
        flt = h.get("filters", {})
        budget_str = f"<= {int(flt['max_budget']):,} đ".replace(",", ".") if flt.get("max_budget") else "Không giới hạn"
        hist_table.append({
            "Mục đích": h.get("purpose"),
            "Ngân sách trần": budget_str,
            "Top gợi ý": ", ".join(h.get("top_names", [])[:3]),
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(hist_table), hide_index=True, use_container_width=True)
