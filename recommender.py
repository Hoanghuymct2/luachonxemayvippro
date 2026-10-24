"""
Module recommender.py
Core DSS Motorcycle Recommender using Weighted K-Nearest Neighbors (KNN).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# 1. Hằng số danh sách đặc trưng
# ---------------------------------------------------------------------------
NUMERIC_COLS: List[str] = [
    "Giá(VND)",
    "Công suất tối đa (Kw)",
    "Dung tích cốp (L)",
    "Năm sản xuất",
]

CATEGORICAL_COLS: List[str] = [
    "Hãng",
    "Nhiên liệu",
    "Kiểu xe",
    "Tình trạng",
]

ALL_GROUPS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS

# ---------------------------------------------------------------------------
# 2. Năm hồ sơ trọng số theo mục đích sử dụng (DSS Profiles)
# ---------------------------------------------------------------------------
PURPOSE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Đi làm, đi phố & Gia đình": {
        "Giá(VND)": 0.20,
        "Dung tích cốp (L)": 0.25,
        "Kiểu xe": 0.20,
        "Nhiên liệu": 0.15,
        "Hãng": 0.10,
        "Công suất tối đa (Kw)": 0.05,
        "Năm sản xuất": 0.05,
        "Tình trạng": 0.00,
    },
    "Học sinh, Sinh viên": {
        "Giá(VND)": 0.35,
        "Kiểu xe": 0.20,
        "Dung tích cốp (L)": 0.15,
        "Nhiên liệu": 0.15,
        "Hãng": 0.05,
        "Công suất tối đa (Kw)": 0.05,
        "Năm sản xuất": 0.05,
        "Tình trạng": 0.00,
    },
    "Chạy dịch vụ (Grab/Be/Giao hàng)": {
        "Giá(VND)": 0.30,
        "Nhiên liệu": 0.20,
        "Công suất tối đa (Kw)": 0.15,
        "Kiểu xe": 0.15,
        "Dung tích cốp (L)": 0.10,
        "Hãng": 0.05,
        "Năm sản xuất": 0.05,
        "Tình trạng": 0.00,
    },
    "Đi phượt, Thể thao & Đam mê tốc độ": {
        "Công suất tối đa (Kw)": 0.35,
        "Kiểu xe": 0.25,
        "Giá(VND)": 0.10,
        "Nhiên liệu": 0.10,
        "Hãng": 0.10,
        "Dung tích cốp (L)": 0.05,
        "Năm sản xuất": 0.05,
        "Tình trạng": 0.00,
    },
    "Thời trang, Sang trọng & Cao cấp": {
        "Hãng": 0.25,
        "Kiểu xe": 0.25,
        "Dung tích cốp (L)": 0.15,
        "Giá(VND)": 0.10,
        "Công suất tối đa (Kw)": 0.10,
        "Năm sản xuất": 0.10,
        "Nhiên liệu": 0.05,
        "Tình trạng": 0.00,
    },
}

DEFAULT_PURPOSE: str = "Đi làm, đi phố & Gia đình"

PURPOSE_DESCRIPTIONS: Dict[str, str] = {
    "Đi làm, đi phố & Gia đình": "Ưu tiên cốp xe rộng rãi chứa đồ, mức giá vừa phải, kiểu dáng linh hoạt và tiết kiệm nhiên liệu cho nhu cầu sinh hoạt hàng ngày.",
    "Học sinh, Sinh viên": "Ưu tiên giá thành hợp lý, tiết kiệm chi phí nhiên liệu, kiểu dáng gọn nhẹ hoặc xe 50cc/xe điện phù hợp lứa tuổi.",
    "Chạy dịch vụ (Grab/Be/Giao hàng)": "Ưu tiên chi phí đầu tư thấp, tiết kiệm nhiên liệu tối đa, máy bền bỉ, công suất ổn định để vận hành đường dài cả ngày.",
    "Đi phượt, Thể thao & Đam mê tốc độ": "Ưu tiên động cơ mạnh mẽ, công suất lớn, khung sườn đầm chắc và kiểu dáng xe côn tay thể thao linh hoạt trên xa lộ.",
    "Thời trang, Sang trọng & Cao cấp": "Ưu tiên thương hiệu danh tiếng, thiết kế thanh lịch tôn dáng người lái, mức độ hoàn thiện cao cấp và tiện nghi hiện đại.",
}


# ---------------------------------------------------------------------------
# 3. Hàm chuẩn hóa dữ liệu đầu vào
# ---------------------------------------------------------------------------
def _parse_price_value(val: object) -> float:
    """Chuẩn hóa giá trị giá tiền sang số thực float (VND)."""
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    
    # Xử lý dấu phân cách hàng nghìn và thập phân
    if "." in s and "," in s:
        # Định dạng 15.000.000,50
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        if s.count(".") > 1:
            # Định dạng 166.190.000 hoặc 1.231.000.000
            s = s.replace(".", "")
        else:
            # 1 dấu chấm duy nhất: kiểm tra nếu là phân cách hàng nghìn (vd 50.000) hay thập phân
            parts = s.split(".")
            if len(parts[1]) == 3 and float(parts[0]) < 1000 and len(parts[0]) <= 3:
                s = parts[0] + parts[1]
    elif "," in s:
        if s.count(",") > 1:
            s = s.replace(",", "")
        else:
            parts = s.split(",")
            if len(parts[1]) == 3 and float(parts[0]) < 1000:
                s = parts[0] + parts[1]
            else:
                s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_trunk_value(val: object) -> float:
    """Chuẩn hóa dung tích cốp sang số thực float (Lít)."""
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().upper().replace("L", "").strip()
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_power_value(val: object) -> float:
    """Chuẩn hóa công suất tối đa sang số thực float (Kw)."""
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().upper().replace("KW", "").strip()
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_and_clean_data(path: str = "dataset.csv") -> pd.DataFrame:
    """
    Đọc và làm sạch dữ liệu xe máy từ file CSV.
    
    Quy tắc:
    - Đọc encoding UTF-8, UTF-8-BOM hoặc fallback.
    - Forward fill tên Hãng bị trống ở các dòng con, loại bỏ 'nan'/'none'.
    - Chuẩn hóa Model: thêm tiền tố Hãng nếu chưa bắt đầu bằng Hãng (không phân biệt hoa thường).
    - Chuẩn hóa Giá(VND), Công suất tối đa (Kw), Dung tích cốp (L), Năm sản xuất.
    - Chuẩn hóa các cột phân loại.
    - Kiểm tra loại bỏ giá trị bất hợp lý (âm, ngoài khoảng).
    - Trả về DataFrame sạch, không sửa đổi dữ liệu gốc.
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "cp1258", "latin1"]
    raw_df = None
    last_err = None
    for enc in encodings_to_try:
        try:
            raw_df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            continue
            
    if raw_df is None:
        raise ValueError(f"Không thể đọc file CSV '{path}'. Lỗi: {last_err}")

    # Tạo bản sao độc lập
    df = raw_df.copy()
    
    # Chuẩn hóa tên cột
    df.columns = [str(c).strip() for c in df.columns]
    
    # Kiểm tra các cột bắt buộc tối thiểu
    expected_cols = [
        "Hãng", "Model", "Giá(VND)", "Công suất tối đa (Kw)",
        "Nhiên liệu", "Kiểu xe", "Dung tích cốp", "Tình trạng", "Năm sản xuất"
    ]
    missing_required = [c for c in expected_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {missing_required}")

    # 1. Xử lý Hãng (Forward fill & loại bỏ nan)
    df["Hãng"] = df["Hãng"].replace(r"^\s*$", np.nan, regex=True)
    df["Hãng"] = df["Hãng"].ffill()
    df = df[df["Hãng"].notna() & (~df["Hãng"].astype(str).str.lower().isin(["nan", "none", ""]))]
    df["Hãng"] = df["Hãng"].astype(str).str.strip()

    # 2. Xử lý Model: Thêm tiền tố Hãng nếu chưa có
    def _fix_model_name(row: pd.Series) -> str:
        brand = str(row["Hãng"]).strip()
        model = str(row["Model"]).strip()
        if not model.lower().startswith(brand.lower()):
            return f"{brand} {model}"
        return model

    df["Model"] = df.apply(_fix_model_name, axis=1)

    # 3. Chuẩn hóa Giá(VND)
    df["Giá(VND)"] = df["Giá(VND)"].apply(_parse_price_value)

    # 4. Chuẩn hóa Công suất tối đa (Kw)
    df["Công suất tối đa (Kw)"] = df["Công suất tối đa (Kw)"].apply(_parse_power_value)

    # 5. Chuẩn hóa Dung tích cốp (L)
    df["Dung tích cốp (L)"] = df["Dung tích cốp"].apply(_parse_trunk_value)

    # 6. Chuẩn hóa Năm sản xuất
    df["Năm sản xuất"] = pd.to_numeric(df["Năm sản xuất"], errors="coerce").astype("Int64")

    # 7. Chuẩn hóa chuỗi phân loại
    for col in ["Hãng", "Nhiên liệu", "Kiểu xe", "Tình trạng"]:
        df[col] = df[col].astype(str).str.strip()

    # 8. Lọc các giá trị bất hợp lý
    valid_mask = (
        (df["Giá(VND)"] > 0) &
        (df["Công suất tối đa (Kw)"] >= 0) &
        (df["Dung tích cốp (L)"] >= 0) &
        (df["Năm sản xuất"].notna()) &
        (df["Năm sản xuất"] >= 1990) &
        (df["Năm sản xuất"] <= 2030)
    )
    df = df[valid_mask].copy()

    # Chuyển năm sản xuất sang float/int chuẩn
    df["Năm sản xuất"] = df["Năm sản xuất"].astype(float)

    # 9. Loại bỏ trùng lặp hoàn toàn nếu có
    df = df.drop_duplicates(subset=["Model", "Năm sản xuất", "Giá(VND)"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 4. Hàm phân bổ trọng số cho từng chiều (Per-dimension weights)
# ---------------------------------------------------------------------------
def _per_dimension_weights(
    group_weights: Dict[str, float],
    cat_columns_map: Dict[str, List[str]]
) -> np.ndarray:
    """
    Phân bổ trọng số của 8 nhóm thuộc tính cho từng chiều của ma trận đặc trưng.
    
    Quy tắc:
    - Với nhóm số (1 chiều): w = W_g
    - Với nhóm phân loại (m chiều One-Hot): w_j = W_g / m
    - Tổng trọng số nhóm = 1.0 (sai số 1e-9)
    - Tổng trọng số các chiều = 1.0 (sai số 1e-9)
    """
    # Kiểm tra đủ 8 nhóm thuộc tính
    missing_groups = [g for g in ALL_GROUPS if g not in group_weights]
    if missing_groups:
        raise ValueError(f"Thiếu trọng số cho các nhóm: {missing_groups}")

    # Kiểm tra không âm
    for g, w in group_weights.items():
        if w < 0:
            raise ValueError(f"Trọng số của nhóm '{g}' không được âm ({w})")

    # Kiểm tra tổng trọng số nhóm
    group_sum = sum(group_weights[g] for g in ALL_GROUPS)
    if abs(group_sum - 1.0) > 1e-9:
        raise ValueError(f"Tổng trọng số các nhóm phải bằng 1.0 (hiện tại: {group_sum:.10f})")

    dim_weights = []

    # 1. Bốn đặc trưng số
    for num_col in NUMERIC_COLS:
        dim_weights.append(float(group_weights[num_col]))

    # 2. Bốn đặc trưng phân loại One-Hot
    for cat_col in CATEGORICAL_COLS:
        one_hot_cols = cat_columns_map.get(cat_col, [])
        m = len(one_hot_cols)
        w_g = float(group_weights[cat_col])
        if m > 0:
            per_dim = w_g / m
            dim_weights.extend([per_dim] * m)

    dim_arr = np.array(dim_weights, dtype=np.float64)
    total_dim_sum = float(np.sum(dim_arr))
    if abs(total_dim_sum - 1.0) > 1e-9:
        raise ValueError(f"Tổng trọng số phân bổ theo chiều không bằng 1.0: {total_dim_sum:.10f}")

    return dim_arr


# ---------------------------------------------------------------------------
# 5. Phân tích đáp ứng từng tiêu chí (_criteria_breakdown)
# ---------------------------------------------------------------------------
def _criteria_breakdown(
    query: Dict[str, object],
    car_row: pd.Series,
    numeric_mins: Optional[Dict[str, float]] = None,
    numeric_maxs: Optional[Dict[str, float]] = None,
) -> List[Dict[str, object]]:
    """
    Phân tích mức độ đáp ứng từng tiêu chí mà người dùng đã chọn.
    Chỉ phân tích các tiêu chí khác None và khác 'Không yêu cầu'.
    """
    if numeric_mins is None:
        numeric_mins = {
            "Giá(VND)": 12900000.0,
            "Công suất tối đa (Kw)": 0.5,
            "Dung tích cốp (L)": 0.0,
            "Năm sản xuất": 2022.0,
        }
    if numeric_maxs is None:
        numeric_maxs = {
            "Giá(VND)": 1231000000.0,
            "Công suất tối đa (Kw)": 140.0,
            "Dung tích cốp (L)": 50.0,
            "Năm sản xuất": 2024.0,
        }

    breakdown = []

    # 1. Thuộc tính số
    for col in NUMERIC_COLS:
        target_val = query.get(col)
        if target_val is None or str(target_val).strip() in ["", "None"]:
            continue
        
        try:
            t_val = float(target_val)
        except (ValueError, TypeError):
            continue

        car_val = float(car_row[col])
        min_v = numeric_mins.get(col, 0.0)
        max_v = numeric_maxs.get(col, 1.0)
        
        if max_v == min_v:
            match_pct = 100.0
        else:
            t_norm = (t_val - min_v) / (max_v - min_v)
            c_norm = (car_val - min_v) / (max_v - min_v)
            match_pct = round(max(0.0, (1.0 - abs(c_norm - t_norm))) * 100.0, 1)

        # Định dạng hiển thị
        if col == "Giá(VND)":
            car_text = f"{int(car_val):,} VNĐ".replace(",", ".")
            target_text = f"{int(t_val):,} VNĐ".replace(",", ".")
        elif col == "Công suất tối đa (Kw)":
            car_text = f"{car_val} kW"
            target_text = f"{t_val} kW"
        elif col == "Dung tích cốp (L)":
            car_text = f"{car_val} L"
            target_text = f"{t_val} L"
        elif col == "Năm sản xuất":
            car_text = f"năm {int(car_val)}"
            target_text = f"năm {int(t_val)}"
        else:
            car_text = str(car_val)
            target_text = str(t_val)

        is_good = match_pct >= 80.0
        if is_good:
            msg = f"✓ {col}: xe có {car_text}, rất sát yêu cầu ({target_text})."
        else:
            msg = f"✗ {col}: xe có {car_text}, lệch khá nhiều so với yêu cầu ({target_text})."

        breakdown.append({
            "column": col,
            "type": "numeric",
            "target": target_text,
            "car_value": car_text,
            "match_pct": match_pct,
            "is_good": is_good,
            "message": msg,
        })

    # 2. Thuộc tính phân loại
    for col in CATEGORICAL_COLS:
        target_val = query.get(col)
        if target_val is None or str(target_val).strip() in ["", "None", "Không yêu cầu"]:
            continue

        target_str = str(target_val).strip()
        car_str = str(car_row[col]).strip()
        is_match = (target_str.lower() == car_str.lower())
        match_pct = 100.0 if is_match else 0.0

        if is_match:
            msg = f"✓ {col}: đúng yêu cầu ({target_str})."
        else:
            msg = f"✗ {col}: xe là '{car_str}', khác với yêu cầu '{target_str}'."

        breakdown.append({
            "column": col,
            "type": "categorical",
            "target": target_str,
            "car_value": car_str,
            "match_pct": match_pct,
            "is_good": is_match,
            "message": msg,
        })

    return breakdown


# ---------------------------------------------------------------------------
# 6. Lớp MotoRecommender
# ---------------------------------------------------------------------------
class MotoRecommender:
    """
    Hệ thống hỗ trợ ra quyết định gợi ý xe máy bằng Weighted KNN.
    """
    def __init__(self, data_or_path: Union[pd.DataFrame, str] = "dataset.csv"):
        if isinstance(data_or_path, str):
            self.df = load_and_clean_data(data_or_path)
        elif isinstance(data_or_path, pd.DataFrame):
            self.df = data_or_path.copy().reset_index(drop=True)
        else:
            raise TypeError("data_or_path phải là đường dẫn str hoặc pd.DataFrame")

        if self.df.empty:
            raise ValueError("Dataset trống sau khi đọc và làm sạch!")

        # Fit MinMaxScaler trên 4 đặc trưng số của toàn bộ tập dữ liệu hợp lệ
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.df[NUMERIC_COLS].values)

        # Lưu min/max phục vụ kiểm tra và giải thích
        self.numeric_mins = {col: float(self.scaler.data_min_[i]) for i, col in enumerate(NUMERIC_COLS)}
        self.numeric_maxs = {col: float(self.scaler.data_max_[i]) for i, col in enumerate(NUMERIC_COLS)}

        # Xây dựng danh sách cột One-Hot cố định từ dữ liệu sạch
        self.cat_dummies = pd.get_dummies(self.df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, dtype=float)
        self.one_hot_feature_names: List[str] = list(self.cat_dummies.columns)

        # Lưu bản đồ các cột One-Hot thuộc từng nhóm phân loại
        self.cat_columns_map: Dict[str, List[str]] = {}
        for cat_col in CATEGORICAL_COLS:
            prefix = f"{cat_col}_"
            self.cat_columns_map[cat_col] = [c for c in self.one_hot_feature_names if c.startswith(prefix)]

        # Toàn bộ danh sách tên đặc trưng: 4 số + M cột One-Hot
        self.all_feature_names: List[str] = NUMERIC_COLS + self.one_hot_feature_names

        # Tính ma trận đặc trưng chuẩn hóa P cho toàn bộ dataset
        scaled_numeric = self.scaler.transform(self.df[NUMERIC_COLS].values)
        self.P = np.hstack([scaled_numeric, self.cat_dummies.values]).astype(np.float64)

        # Lưu danh sách giá trị phân loại hợp lệ của từng cột
        self.valid_categories: Dict[str, List[str]] = {}
        for col in CATEGORICAL_COLS:
            self.valid_categories[col] = sorted(list(self.df[col].dropna().unique()))

    def unique_values(self, col: str) -> List[str]:
        """Lấy danh sách các giá trị phân loại duy nhất của cột."""
        if col in self.valid_categories:
            return self.valid_categories[col]
        if col in self.df.columns:
            return sorted(list(self.df[col].dropna().unique()))
        return []

    def get_stats(self) -> Dict[str, object]:
        """Thống kê chi tiết về dataset phục vụ báo cáo và kiểm tra."""
        return {
            "total_rows": len(self.df),
            "brands_count": self.df["Hãng"].value_counts().to_dict(),
            "fuels_count": self.df["Nhiên liệu"].value_counts().to_dict(),
            "body_types_count": self.df["Kiểu xe"].value_counts().to_dict(),
            "years_count": self.df["Năm sản xuất"].astype(int).value_counts().to_dict(),
            "price_min": float(self.df["Giá(VND)"].min()),
            "price_max": float(self.df["Giá(VND)"].max()),
            "price_median": float(self.df["Giá(VND)"].median()),
            "power_min": float(self.df["Công suất tối đa (Kw)"].min()),
            "power_max": float(self.df["Công suất tối đa (Kw)"].max()),
            "power_median": float(self.df["Công suất tối đa (Kw)"].median()),
            "trunk_min": float(self.df["Dung tích cốp (L)"].min()),
            "trunk_max": float(self.df["Dung tích cốp (L)"].max()),
            "trunk_median": float(self.df["Dung tích cốp (L)"].median()),
            "feature_dims": len(self.all_feature_names),
        }

    def _encode_query(self, query: Dict[str, object]) -> np.ndarray:
        """
        Mã hóa vector truy vấn q cùng không gian đặc trưng với P.
        
        Quy tắc:
        - Tiêu chí số không cung cấp (None): mã hóa thành 0.5 (trung tính).
        - Tiêu chí số có giá trị: chuẩn hóa qua scaler đã fit.
        - Tiêu chí phân loại không cung cấp / 'Không yêu cầu': tất cả One-Hot = 0.
        - Tiêu chí phân loại có giá trị: One-Hot tương ứng = 1, còn lại = 0.
        - Giá trị phân loại không hợp lệ: raise ValueError rõ ràng.
        """
        # 1. Mã hóa đặc trưng số
        numeric_vec = []
        for i, col in enumerate(NUMERIC_COLS):
            val = query.get(col)
            if val is None or str(val).strip() in ["", "None"]:
                # Giá trị trung tính 0.5
                numeric_vec.append(0.5)
            else:
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Giá trị truy vấn cho cột '{col}' phải là số, nhận được: {val}")
                
                min_v = self.numeric_mins[col]
                max_v = self.numeric_maxs[col]
                if max_v == min_v:
                    scaled = 0.5
                else:
                    scaled = (num_val - min_v) / (max_v - min_v)
                numeric_vec.append(scaled)

        # 2. Mã hóa đặc trưng phân loại One-Hot
        cat_vec = []
        for cat_col in CATEGORICAL_COLS:
            val = query.get(cat_col)
            one_hot_cols = self.cat_columns_map.get(cat_col, [])
            
            if val is None or str(val).strip() in ["", "None", "Không yêu cầu"]:
                # Không yêu cầu -> toàn bộ 0
                cat_vec.extend([0.0] * len(one_hot_cols))
            else:
                str_val = str(val).strip()
                target_col_name = f"{cat_col}_{str_val}"
                if target_col_name not in one_hot_cols:
                    valid_opts = self.valid_categories.get(cat_col, [])
                    raise ValueError(
                        f"Giá trị phân loại '{str_val}' không hợp lệ cho cột '{cat_col}'. "
                        f"Các giá trị hợp lệ: {valid_opts}"
                    )
                for c in one_hot_cols:
                    cat_vec.append(1.0 if c == target_col_name else 0.0)

        full_q = np.array(numeric_vec + cat_vec, dtype=np.float64)
        return full_q

    def _criteria_breakdown(self, query: Dict[str, object], car_row: pd.Series) -> List[Dict[str, object]]:
        """Phân tích mức độ đáp ứng từng tiêu chí mà người dùng đã chọn."""
        return _criteria_breakdown(query, car_row, self.numeric_mins, self.numeric_maxs)

    def recommend(
        self,
        query: Dict[str, object],
        weights: Optional[Dict[str, float]] = None,
        k: int = 5
    ) -> pd.DataFrame:
        """
        Thực hiện gợi ý Top-K xe máy theo Weighted KNN.
        
        Quy trình:
        1. Kiểm tra k hợp lệ.
        2. Kiểm tra bộ trọng số hoặc dùng mặc định.
        3. Phân bổ trọng số từng chiều (per-dimension).
        4. Lọc ngân sách cứng (Hard Budget Filter).
        5. Mã hóa vector truy vấn.
        6. Tính khoảng cách Weighted Euclidean Distance.
        7. Sắp xếp khoảng cách tăng dần, lấy Top-K.
        8. Tính điểm phù hợp tương đối Score = round(1 / (1 + d_w) * 100, 1).
        9. Phân tích chi tiết mức độ đáp ứng từng tiêu chí.
        """
        if k <= 0:
            raise ValueError(f"Số lượng kết quả k phải > 0, nhận được: {k}")

        if weights is None:
            weights = PURPOSE_WEIGHTS[DEFAULT_PURPOSE]

        # Tính trọng số theo chiều
        dim_weights = _per_dimension_weights(weights, self.cat_columns_map)

        # 1. Lọc ngân sách cứng
        budget = query.get("Giá(VND)")
        if budget is None:
            budget = query.get("budget")
        
        candidate_indices = np.arange(len(self.df))
        if budget is not None:
            try:
                b_val = float(budget)
                if b_val > 0:
                    budget_mask = (self.df["Giá(VND)"].values <= b_val)
                    candidate_indices = np.where(budget_mask)[0]
            except (ValueError, TypeError):
                pass

        if len(candidate_indices) == 0:
            # Không có xe nào thỏa mãn ngân sách
            cols = [
                "Thứ hạng", "Tên xe", "Hãng", "Giá(VND)", "Kiểu xe",
                "Nhiên liệu", "Dung tích cốp (L)", "Công suất tối đa (Kw)",
                "Năm sản xuất", "Độ phù hợp (%)", "Khoảng cách KNN", "Chi tiết tiêu chí"
            ]
            return pd.DataFrame(columns=cols)

        # 2. Mã hóa query
        q = self._encode_query(query)

        # 3. Trích xuất ma trận con cho các ứng viên thỏa mãn ngân sách
        P_cand = self.P[candidate_indices]

        # 4. Tính khoảng cách Weighted Euclidean
        diff = P_cand - q
        dist_sq = np.dot(diff ** 2, dim_weights)
        # Đảm bảo không âm do sai số dấu phẩy động
        dist_sq = np.maximum(dist_sq, 0.0)
        distances = np.sqrt(dist_sq)

        # 5. Sắp xếp lấy Top-K
        num_to_take = min(k, len(candidate_indices))
        sorted_cand_idx = np.argsort(distances)[:num_to_take]

        selected_indices = candidate_indices[sorted_cand_idx]
        selected_distances = distances[sorted_cand_idx]

        results = []
        for rank, (orig_idx, dist) in enumerate(zip(selected_indices, selected_distances), start=1):
            row = self.df.iloc[orig_idx]
            dist_val = float(dist)
            score = round((1.0 / (1.0 + dist_val)) * 100.0, 1)
            
            # Phân tích tiêu chí
            breakdown = self._criteria_breakdown(query, row)

            results.append({
                "Thứ hạng": rank,
                "Tên xe": str(row["Model"]),
                "Hãng": str(row["Hãng"]),
                "Giá(VND)": float(row["Giá(VND)"]),
                "Kiểu xe": str(row["Kiểu xe"]),
                "Nhiên liệu": str(row["Nhiên liệu"]),
                "Dung tích cốp (L)": float(row["Dung tích cốp (L)"]),
                "Công suất tối đa (Kw)": float(row["Công suất tối đa (Kw)"]),
                "Năm sản xuất": int(row["Năm sản xuất"]),
                "Độ phù hợp (%)": score,
                "Khoảng cách KNN": round(dist_val, 4),
                "Chi tiết tiêu chí": breakdown,
            })

        return pd.DataFrame(results)


# Alias tương thích ngược
CarRecommender = MotoRecommender
