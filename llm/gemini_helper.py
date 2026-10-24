"""
Module llm/gemini_helper.py
Integration with Google Gemini API for natural language need parsing and result explanation,
with robust fallbacks, error handling, and server-side security guarantees.
"""

import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from core.recommender import (
    DEFAULT_PURPOSE,
    PURPOSE_DESCRIPTIONS,
    PURPOSE_WEIGHTS,
)

DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-flash-latest"]

# Trạng thái model trong phiên làm việc
ACTIVE_WORKING_MODEL: Optional[str] = None
MODEL_STATUS_MESSAGE: str = "Chưa kiểm tra"


def get_gemini_api_key(provided_key: Optional[str] = None) -> Optional[str]:
    """
    Lấy API key an toàn theo thứ tự ưu tiên:
    1. Key truyền từ cấu hình server (nếu có).
    2. Biến môi trường GEMINI_API_KEY.
    3. File keyapi.txt tại thư mục gốc dự án.
    
    Tuyệt đối không log hoặc in key ra terminal.
    """
    if provided_key and str(provided_key).strip():
        return str(provided_key).strip()

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and str(env_key).strip():
        return str(env_key).strip()

    # Thử tìm file keyapi.txt ở root hoặc các cấp thư mục
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(root_dir, "keyapi.txt"),
        os.path.join(os.getcwd(), "keyapi.txt"),
        "keyapi.txt"
    ]
    for key_file in candidates:
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception:
                pass

    return None


def get_gemini_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Khởi tạo Client của google-genai nếu key hợp lệ."""
    if not HAS_GENAI:
        return None
    key = get_gemini_api_key(api_key)
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def get_model_status(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Kiểm tra trạng thái kết nối và model Gemini tại Server."""
    key = get_gemini_api_key(api_key)
    if not HAS_GENAI:
        return {
            "ready": False,
            "status": "warning",
            "message": "Chưa cài đặt thư viện google-genai",
            "model": None,
        }
    if not key:
        return {
            "ready": False,
            "status": "info",
            "message": "Chưa cấu hình API Key (đang dùng Fallback)",
            "model": None,
        }

    global ACTIVE_WORKING_MODEL
    if ACTIVE_WORKING_MODEL:
        return {
            "ready": True,
            "status": "success",
            "message": f"Sẵn sàng ({ACTIVE_WORKING_MODEL})",
            "model": ACTIVE_WORKING_MODEL,
        }

    return {
        "ready": True,
        "status": "success",
        "message": f"Sẵn sàng (Mặc định: {DEFAULT_MODEL})",
        "model": DEFAULT_MODEL,
    }


def _call_gemini_with_fallback(
    client: Any,
    contents: str,
    preferred_model: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Gọi Gemini API với model ưu tiên và tự động chuyển đổi sang fallback nếu model bị 404/deprecated.
    """
    global ACTIVE_WORKING_MODEL, MODEL_STATUS_MESSAGE

    if preferred_model:
        models_to_try = [preferred_model]
    elif ACTIVE_WORKING_MODEL:
        models_to_try = [ACTIVE_WORKING_MODEL]
    else:
        models_to_try = [DEFAULT_MODEL]
        for fm in FALLBACK_MODELS:
            if fm not in models_to_try:
                models_to_try.append(fm)

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            if response and response.text:
                ACTIVE_WORKING_MODEL = model_name
                MODEL_STATUS_MESSAGE = f"Đang sử dụng {model_name}"
                return response.text.strip(), model_name
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "404" in err_str or "no longer available" in err_str:
                continue
            else:
                break

    return None, None


# ---------------------------------------------------------------------------
# 1. Fallback phân tích nhu cầu bằng Regex và Từ khóa tiếng Việt
# ---------------------------------------------------------------------------
def _parse_fallback(text: str, options: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Phân tích nhu cầu tự nhiên bằng luật Regex và từ khóa tiếng Việt.
    Dùng khi không có API key hoặc Gemini API gặp lỗi.
    """
    lower = text.lower()

    # 1. Nhận diện Mục đích sử dụng
    if re.search(r"\b(sinh viên|học sinh|đi học)\b", lower):
        purpose = "Học sinh, Sinh viên"
    elif re.search(r"\b(dịch vụ|grab|be|giao hàng|ship|chạy xe ôm|shipper)\b", lower):
        purpose = "Chạy dịch vụ (Grab/Be/Giao hàng)"
    elif re.search(r"\b(phượt|thể thao|tốc độ|phân khối|bốc|mạnh mẽ|đua)\b", lower):
        purpose = "Đi phượt, Thể thao & Đam mê tốc độ"
    elif re.search(r"\b(thời trang|sang trọng|cao cấp|dạo phố|sang|quý phái|thanh lịch)\b", lower):
        purpose = "Thời trang, Sang trọng & Cao cấp"
    else:
        purpose = "Đi làm, đi phố & Gia đình"

    # 2. Ngân sách (triệu, tr, củ)
    budget_trieu = None
    budget_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:triệu|tr|củ)\b", lower)
    if budget_match:
        try:
            budget_trieu = float(budget_match.group(1).replace(",", "."))
        except ValueError:
            budget_trieu = None

    # 3. Dung tích cốp (lít, L)
    trunk_liters = None
    inferred_trunk = False
    trunk_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:lít|lit|l)\b", lower)
    if trunk_match:
        try:
            trunk_liters = float(trunk_match.group(1).replace(",", "."))
        except ValueError:
            trunk_liters = None
    elif "cốp rộng" in lower or "cốp to" in lower or "cốp bự" in lower or "đựng vừa balo" in lower:
        trunk_liters = 25.0
        inferred_trunk = True

    # 4. Công suất (kW)
    power_kw = None
    power_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:kw)\b", lower)
    if power_match:
        try:
            power_kw = float(power_match.group(1).replace(",", "."))
        except ValueError:
            power_kw = None

    # 5. Năm sản xuất
    year = None
    year_match = re.search(r"\b(202[0-9])\b", lower)
    if year_match:
        try:
            year = int(year_match.group(1))
        except ValueError:
            year = None

    # 6. Hãng xe
    brand = None
    valid_brands = options.get("Hãng", [])
    for b in valid_brands:
        if b.lower() in lower:
            brand = b
            break

    # 7. Nhiên liệu
    fuel = None
    if "điện" in lower or "xe điện" in lower or "pin" in lower:
        fuel = "Điện"
    elif "xăng" in lower:
        fuel = "Xăng"

    # 8. Kiểu xe
    body = None
    if "50cc" in lower:
        if "ga" in lower:
            body = "Tay ga (50cc)"
        elif "số" in lower:
            body = "Xe số (50cc)"
    elif "tay ga" in lower or "xe ga" in lower:
        body = "Tay ga"
    elif "côn tay" in lower or "xe côn" in lower or "tay côn" in lower:
        body = "Xe côn tay"
    elif "xe điện" in lower:
        body = "Xe điện"
    elif "xe số" in lower:
        body = "Xe số"

    if fuel and fuel not in options.get("Nhiên liệu", []):
        fuel = None
    if body and body not in options.get("Kiểu xe", []):
        body = None

    budget_vnd = budget_trieu * 1_000_000.0 if budget_trieu is not None else None

    return {
        "purpose": purpose,
        "budget_trieu": budget_trieu,
        "budget_vnd": budget_vnd,
        "power_kw": power_kw,
        "trunk_liters": trunk_liters,
        "year": year,
        "brand": brand,
        "fuel": fuel,
        "body": body,
        "condition": "Mới",
        "source": "fallback",
        "inferred_trunk": inferred_trunk,
    }


# ---------------------------------------------------------------------------
# 2. Hàm phân tích nhu cầu bằng Gemini (với fallback)
# ---------------------------------------------------------------------------
def parse_need(
    text: str,
    options: Dict[str, List[str]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phân tích nhu cầu tự nhiên của người dùng thành bộ tiêu chí chuẩn hóa.
    Nếu Gemini khả dụng và thành công, trả về kết quả từ Gemini.
    Nếu thất bại hoặc không có API key, tự động chuyển sang _parse_fallback.
    """
    text = (text or "").strip()
    if not text:
        return _parse_fallback("", options)

    client = get_gemini_client(api_key)
    if not client:
        return _parse_fallback(text, options)

    purposes_str = "; ".join(PURPOSE_WEIGHTS.keys())
    brands_str = ", ".join(options.get("Hãng", []))
    fuels_str = ", ".join(options.get("Nhiên liệu", []))
    body_str = ", ".join(options.get("Kiểu xe", []))

    prompt = f"""Bạn là chuyên gia tư vấn xe máy. Hãy phân tích nhu cầu tự nhiên sau của người dùng và trích xuất thành định dạng JSON chuẩn.

QUY TẮC BẮT BUỘC:
1. "purpose" BẮT BUỘC phải là một trong 5 mục đích sau:
   [{purposes_str}]
2. "brand" phải nằm trong danh sách: [{brands_str}] hoặc null nếu người dùng không yêu cầu hãng cụ thể.
3. "fuel" phải nằm trong danh sách: [{fuels_str}] hoặc null.
4. "body" (Kiểu xe) phải nằm trong danh sách: [{body_str}] hoặc null.
5. "condition" luôn là "Mới" hoặc null.
6. "budget_trieu": Ngân sách tối đa tính bằng TRIỆU VND (dạng số thực, vd: 45 triệu -> 45.0) hoặc null nếu không nói.
7. "power_kw": Công suất tối đa mong muốn (Kw) hoặc null.
8. "trunk_liters": Dung tích cốp mong muốn (Lít) hoặc null.
9. "year": Năm sản xuất mong muốn (vd: 2024) hoặc null.
10. Nếu người dùng đề cập nhiều lựa chọn như "xe ga hoặc xe điện", hãy ưu tiên kiểu xe cụ thể đầu tiên được nhắc đến hoặc để null nếu không chắc chắn.

Nhu cầu của người dùng:
"{text}"

Chỉ trả về DUY NHẤT một khối JSON hợp lệ theo đúng cấu trúc mẫu sau (không kèm giải thích bên ngoài):
{{
  "purpose": "Đi làm, đi phố & Gia đình",
  "budget_trieu": 45.0,
  "power_kw": null,
  "trunk_liters": 25.0,
  "year": null,
  "brand": null,
  "fuel": null,
  "body": "Tay ga",
  "condition": "Mới"
}}
"""
    try:
        raw_response, used_model = _call_gemini_with_fallback(
            client=client,
            contents=prompt,
            preferred_model=model
        )

        if not raw_response:
            return _parse_fallback(text, options)

        clean_json_str = raw_response.strip()
        json_match = re.search(r"\{.*\}", clean_json_str, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)

        data = json.loads(clean_json_str)

        purpose = data.get("purpose")
        if purpose not in PURPOSE_WEIGHTS:
            purpose = DEFAULT_PURPOSE
            for p in PURPOSE_WEIGHTS.keys():
                if str(data.get("purpose", "")).lower() in p.lower():
                    purpose = p
                    break

        brand = data.get("brand")
        if brand and brand not in options.get("Hãng", []):
            brand = None

        fuel = data.get("fuel")
        if fuel and fuel not in options.get("Nhiên liệu", []):
            fuel = None

        body = data.get("body")
        if body and body not in options.get("Kiểu xe", []):
            body = None

        def _safe_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                fv = float(v)
                if np.isnan(fv) or np.isinf(fv) or fv < 0:
                    return None
                return fv
            except (ValueError, TypeError):
                return None

        budget_trieu = _safe_float(data.get("budget_trieu"))
        power_kw = _safe_float(data.get("power_kw"))
        trunk_liters = _safe_float(data.get("trunk_liters"))

        year_val = data.get("year")
        year = None
        if year_val is not None:
            try:
                year = int(year_val)
                if year < 1990 or year > 2030:
                    year = None
            except (ValueError, TypeError):
                year = None

        budget_vnd = budget_trieu * 1_000_000.0 if budget_trieu is not None else None

        return {
            "purpose": purpose,
            "budget_trieu": budget_trieu,
            "budget_vnd": budget_vnd,
            "power_kw": power_kw,
            "trunk_liters": trunk_liters,
            "year": year,
            "brand": brand,
            "fuel": fuel,
            "body": body,
            "condition": "Mới",
            "source": f"gemini ({used_model})",
            "inferred_trunk": False,
        }
    except Exception:
        return _parse_fallback(text, options)


# ---------------------------------------------------------------------------
# 3. Hàm giải thích fallback (Dựa trên dữ liệu thực tế)
# ---------------------------------------------------------------------------
def _explain_fallback(
    purpose: str,
    weights: Dict[str, float],
    query: Dict[str, Any],
    result: Union[pd.DataFrame, List[Dict[str, Any]]]
) -> str:
    """Giải thích kết quả dựa trên dữ liệu định lượng thật khi không có Gemini."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "Không có mẫu xe nào thỏa mãn mức ngân sách và tiêu chí của bạn. Hãy thử nới rộng ngân sách hoặc giảm bớt các điều kiện lọc phân loại."
        top1 = result.iloc[0].to_dict()
        top2 = result.iloc[1].to_dict() if len(result) > 1 else None
    elif isinstance(result, list):
        if not result:
            return "Không có mẫu xe nào thỏa mãn mức ngân sách và tiêu chí của bạn. Hãy thử nới rộng ngân sách hoặc giảm bớt các điều kiện lọc phân loại."
        top1 = result[0]
        top2 = result[1] if len(result) > 1 else None
    else:
        return "Không có mẫu xe phù hợp."

    top1_name = top1.get("Tên xe") or top1.get("name", "Xe đề xuất")
    top1_score = top1.get("Độ phù hợp (%)") or top1.get("fit_score", 0.0)
    top1_price_val = top1.get("Giá(VND)") or top1.get("price", 0)
    top1_price = f"{int(top1_price_val):,} VNĐ".replace(",", ".")
    top1_trunk = f"{top1.get('Dung tích cốp (L)') or top1.get('trunk_liters', 0)} L"
    top1_power = f"{top1.get('Công suất tối đa (Kw)') or top1.get('power_kw', 0)} kW"

    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top_factor, top_factor_w = sorted_weights[0]

    explanation = (
        f"Theo thuật toán Weighted KNN với mục đích '{purpose}', mẫu xe **{top1_name}** dẫn đầu bảng gợi ý "
        f"với độ phù hợp **{top1_score}%**.\n\n"
        f"Lý do mẫu xe này đạt thứ hạng cao nhất là nhờ sự tối ưu vượt trội ở tiêu chí trọng số lớn nhất: "
        f"**{top_factor}** (chiếm {int(top_factor_w * 100)}% tổng trọng số). "
        f"Mẫu xe sở hữu mức giá {top1_price}, dung tích cốp {top1_trunk} và công suất {top1_power}, "
        f"đáp ứng cân bằng giữa khả năng chi trả và hiệu năng thực tế."
    )

    if top2:
        top2_name = top2.get("Tên xe") or top2.get("name", "")
        top2_price_val = top2.get("Giá(VND)") or top2.get("price", 0)
        top2_price = f"{int(top2_price_val):,} VNĐ".replace(",", ".")
        top2_score = top2.get("Độ phù hợp (%)") or top2.get("fit_score", 0.0)
        explanation += (
            f"\n\nSo với phương án kế tiếp là **{top2_name}** ({top2_score}%, giá {top2_price}), "
            f"mẫu xe Top 1 có khoảng cách đặc trưng tổng thể ngắn hơn và sát với nhu cầu ưu tiên của bạn hơn."
        )

    fuel_type = str(top1.get("Nhiên liệu") or top1.get("fuel", "")).lower()
    if "điện" in fuel_type:
        explanation += (
            "\n\n💡 **Lời khuyên thực tế:** Đây là dòng xe điện, bạn nên kiểm tra hạ tầng sạc pin tại nơi ở/làm việc "
            "và quãng đường di chuyển trung bình mỗi ngày trước khi đưa ra quyết định mua."
        )
    else:
        explanation += (
            "\n\n💡 **Lời khuyên thực tế:** Bạn nên đến trực tiếp đại lý trải nghiệm lái thử, "
            "kiểm tra chiều cao yên xe và tư thế ngồi để đảm bảo sự thoải mái tối đa khi điều khiển."
        )

    return explanation


# ---------------------------------------------------------------------------
# 4. Hàm giải thích kết quả gợi ý
# ---------------------------------------------------------------------------
def explain_results(
    purpose: str,
    weights: Dict[str, float],
    query: Dict[str, Any],
    result: Union[pd.DataFrame, List[Dict[str, Any]]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Tạo đoạn văn giải thích kết quả gợi ý (150 - 220 từ) bằng tiếng Việt.
    Dữ liệu hoàn toàn xuất phát từ kết quả thật của Weighted KNN.
    """
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return _explain_fallback(purpose, weights, query, result)
        rows = [r.to_dict() for _, r in result.iterrows()]
    elif isinstance(result, list):
        if not result:
            return _explain_fallback(purpose, weights, query, result)
        rows = result
    else:
        return _explain_fallback(purpose, weights, query, result)

    client = get_gemini_client(api_key)
    if not client:
        return _explain_fallback(purpose, weights, query, rows)

    top1 = rows[0]
    top1_name = top1.get("Tên xe") or top1.get("name")
    top1_brand = top1.get("Hãng") or top1.get("brand")
    top1_price = int(top1.get("Giá(VND)") or top1.get("price", 0))
    top1_body = top1.get("Kiểu xe") or top1.get("body")
    top1_fuel = top1.get("Nhiên liệu") or top1.get("fuel")
    top1_trunk = top1.get("Dung tích cốp (L)") or top1.get("trunk_liters")
    top1_power = top1.get("Công suất tối đa (Kw)") or top1.get("power_kw")
    top1_score = top1.get("Độ phù hợp (%)") or top1.get("fit_score")

    top1_info = (
        f"Top 1: {top1_name} (Hãng: {top1_brand}, Giá: {top1_price:,} VND, "
        f"Kiểu: {top1_body}, Nhiên liệu: {top1_fuel}, Cốp: {top1_trunk}L, "
        f"Công suất: {top1_power}kW, Điểm phù hợp: {top1_score}%)"
    )

    other_info = []
    for i in range(1, min(3, len(rows))):
        r = rows[i]
        r_name = r.get("Tên xe") or r.get("name")
        r_price = int(r.get("Giá(VND)") or r.get("price", 0))
        r_trunk = r.get("Dung tích cốp (L)") or r.get("trunk_liters")
        r_power = r.get("Công suất tối đa (Kw)") or r.get("power_kw")
        r_score = r.get("Độ phù hợp (%)") or r.get("fit_score")
        other_info.append(
            f"Top {i+1}: {r_name} (Giá: {r_price:,} VND, Cốp: {r_trunk}L, Công suất: {r_power}kW, Điểm: {r_score}%)"
        )
    others_str = "\n".join(other_info) if other_info else "Không có xe phụ."

    query_desc = []
    for k, v in query.items():
        if v is not None and str(v).strip() not in ["", "None", "Không yêu cầu"]:
            query_desc.append(f"{k}: {v}")
    query_str = ", ".join(query_desc) if query_desc else "Không có tiêu chí cứng."

    top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    weights_str = ", ".join([f"{k} ({int(w*100)}%)" for k, w in top_weights])

    prompt = f"""Bạn là chuyên gia tư vấn hệ thống hỗ trợ ra quyết định (DSS) mua xe máy.
Hãy viết một đoạn văn giải thích kết quả gợi ý cho người dùng bằng tiếng Việt (độ dài khoảng 150 - 220 từ).

DỮ LIỆU ĐỊNH LƯỢNG THỰC TẾ:
- Mục đích sử dụng: {purpose}
- Các tiêu chí ưu tiên hàng đầu theo bộ trọng số: {weights_str}
- Tiêu chí người dùng yêu cầu: {query_str}
- {top1_info}
- Các xe đứng kế tiếp:
{others_str}

CÁC QUY TẮC NGHIÊM NGẶT:
1. KHÔNG ĐƯỢC THAY ĐỔI thứ hạng hoặc điểm số do thuật toán Weighted KNN tính toán.
2. KHÔNG ĐƯỢC TỰ BỊA thêm các thông số kỹ thuật như mức tiêu hao xăng (lít/100km), tầm hoạt động pin (km), chi phí bảo dưỡng... nếu không có trong dữ liệu trên.
3. Giải thích rõ lý do Top 1 được chọn: dựa trên điểm phù hợp KNN, các tiêu chí người dùng nhập và tiêu chí trọng số cao của mục đích.
4. So sánh ngắn gọn với xe đứng tiếp theo (nếu có).
5. Đưa ra 1 lời khuyên thực tế phù hợp (thử chiều cao yên, tư thế ngồi hoặc hạ tầng sạc nếu là xe điện).
6. Không tuyên bố xe là tốt nhất thị trường, chỉ kết luận xe là phù hợp nhất theo dữ liệu và tiêu chí hiện tại.
"""
    raw_response, used_model = _call_gemini_with_fallback(
        client=client,
        contents=prompt,
        preferred_model=model
    )

    if raw_response and len(raw_response.strip()) >= 50:
        return raw_response.strip()

    return _explain_fallback(purpose, weights, query, rows)
