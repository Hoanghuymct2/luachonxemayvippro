"""
Module history.py
Manages search history persistence in history.json safely and reliably.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")


def _clean_query_for_storage(query: Dict[str, Any]) -> Dict[str, Any]:
    """Loại bỏ các trường None, rỗng và bảo đảm không lưu bất kỳ secret nào."""
    cleaned = {}
    for k, v in query.items():
        if k in ["api_key", "key", "secret", "token"]:
            continue
        if v is not None and str(v).strip() not in ["", "None", "Không yêu cầu"]:
            cleaned[k] = v
    return cleaned


def load_history(file_path: str = HISTORY_FILE) -> List[Dict[str, Any]]:
    """
    Đọc lịch sử tìm kiếm từ file JSON.
    Trả về danh sách rỗng nếu file không tồn tại hoặc dữ liệu lỗi.
    """
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        # Nếu file hỏng, không làm sập ứng dụng, trả về rỗng
        return []


def save_search(
    purpose: str,
    query: Dict[str, Any],
    weights: Dict[str, float],
    top_moto_names: List[str],
    file_path: str = HISTORY_FILE
) -> None:
    """
    Lưu một lượt tìm kiếm vào file JSON.
    
    Quy tắc:
    - Lưu timestamp, mục đích, tiêu chí, trọng số, danh sách tên xe.
    - Không lưu API key hoặc thông tin nhạy cảm.
    - Chống lưu trùng lặp khi Streamlit rerun với cùng kết quả.
    - Giữ tối đa 50 bản ghi gần nhất.
    """
    cleaned_query = _clean_query_for_storage(query)
    
    current_history = load_history(file_path)

    # Chống lưu trùng lặp khi Streamlit rerun
    if current_history:
        last = current_history[-1]
        if (
            last.get("muc_dich") == purpose and
            last.get("tieu_chi") == cleaned_query and
            last.get("danh_sach_xe") == top_moto_names
        ):
            # Lượt tìm kiếm hoàn toàn trùng lặp với lần gần nhất, bỏ qua
            return

    new_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "muc_dich": purpose,
        "tieu_chi": cleaned_query,
        "bo_trong_so": weights,
        "danh_sach_xe": top_moto_names,
    }

    current_history.append(new_record)

    # Giữ tối đa 50 bản ghi
    if len(current_history) > 50:
        current_history = current_history[-50:]

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(current_history, f, ensure_ascii=False, indent=2)
    except Exception:
        # Xử lý lỗi ghi file an toàn
        pass


def clear_history(file_path: str = HISTORY_FILE) -> None:
    """Xóa toàn bộ lịch sử tìm kiếm."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    except Exception:
        pass

