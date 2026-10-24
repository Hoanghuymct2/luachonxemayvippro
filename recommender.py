"""
Module core/recommender.py
Decision Support System (DSS) Core Recommender using Weighted K-Nearest Neighbors (KNN).
Provides transparent Multi-Criteria Decision Making, hard budget constraints,
active criteria weighting, and deterministic ranking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

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

    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        if s.count(".") > 1:
            s = s.replace(".", "")
        else:
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


def load_and_clean_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Đọc và làm sạch dữ liệu xe máy từ file CSV.
    Tự động tìm kiếm tại data/dataset.csv hoặc dataset.csv.
    """
    if path is None:
        candidates = ["data/dataset.csv", "dataset.csv"]
        for cand in candidates:
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            path = "dataset.csv"

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

    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    expected_cols = [
        "Hãng", "Model", "Giá(VND)", "Công suất tối đa (Kw)",
        "Nhiên liệu", "Kiểu xe", "Dung tích cốp", "Tình trạng", "Năm sản xuất"
    ]
    missing_required = [c for c in expected_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {missing_required}")

    # 1. Forward-fill Hãng và loại bỏ nan
    df["Hãng"] = df["Hãng"].replace(r"^\s*$", np.nan, regex=True)
    df["Hãng"] = df["Hãng"].ffill()
    df = df[df["Hãng"].notna() & (~df["Hãng"].astype(str).str.lower().isin(["nan", "none", ""]))]
    df["Hãng"] = df["Hãng"].astype(str).str.strip()

    # 2. Xử lý tiền tố Model
    def _fix_model_name(row: pd.Series) -> str:
        brand = str(row["Hãng"]).strip()
        model = str(row["Model"]).strip()
        if not model.lower().startswith(brand.lower()):
            return f"{brand} {model}"
        return model

    df["Model"] = df.apply(_fix_model_name, axis=1)

    # 3. Chuẩn hóa các cột số
    df["Giá(VND)"] = df["Giá(VND)"].apply(_parse_price_value)
    df["Công suất tối đa (Kw)"] = df["Công suất tối đa (Kw)"].apply(_parse_power_value)
    df["Dung tích cốp (L)"] = df["Dung tích cốp"].apply(_parse_trunk_value)
    df["Năm sản xuất"] = pd.to_numeric(df["Năm sản xuất"], errors="coerce").astype("Int64")

    # 4. Chuẩn hóa chuỗi phân loại
    for col in ["Hãng", "Nhiên liệu", "Kiểu xe", "Tình trạng"]:
        df[col] = df[col].astype(str).str.strip()

    # 5. Lọc bất hợp lý
    valid_mask = (
        (df["Giá(VND)"] > 0) &
        (df["Công suất tối đa (Kw)"] >= 0) &
        (df["Dung tích cốp (L)"] >= 0) &
        (df["Năm sản xuất"].notna()) &
        (df["Năm sản xuất"] >= 1990) &
        (df["Năm sản xuất"] <= 2030)
    )
    df = df[valid_mask].copy()
    df["Năm sản xuất"] = df["Năm sản xuất"].astype(float)

    # 6. Loại bỏ trùng lặp hoàn toàn
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
    Mỗi chiều phân loại One-Hot nhận w_j = W_g / m.
    """
    missing_groups = [g for g in ALL_GROUPS if g not in group_weights]
    if missing_groups:
        raise ValueError(f"Thiếu trọng số cho các nhóm: {missing_groups}")

    for g, w in group_weights.items():
        if w < 0:
            raise ValueError(f"Trọng số của nhóm '{g}' không được âm ({w})")

    group_sum = sum(group_weights[g] for g in ALL_GROUPS)
    if abs(group_sum - 1.0) > 1e-9:
        raise ValueError(f"Tổng trọng số các nhóm phải bằng 1.0 (hiện tại: {group_sum:.10f})")

    dim_weights = []
    for num_col in NUMERIC_COLS:
        dim_weights.append(float(group_weights[num_col]))

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
# 5. Phân tích mức độ đáp ứng tiêu chí (_criteria_breakdown)
# ---------------------------------------------------------------------------
def _criteria_breakdown(
    query: Dict[str, object],
    car_row: pd.Series,
    numeric_mins: Optional[Dict[str, float]] = None,
    numeric_maxs: Optional[Dict[str, float]] = None,
) -> List[Dict[str, object]]:
    """
    Phân tích mức độ đáp ứng từng tiêu chí mà người dùng đã chọn.
    Chỉ phân tích các tiêu chí được người dùng chỉ định.
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

    # 1. Tiêu chí ngân sách tối đa
    max_budget = query.get("max_budget")
    if max_budget is None and query.get("is_budget_only") and query.get("Giá(VND)"):
        max_budget = query.get("Giá(VND)")
    
    if max_budget is not None and not query.get("target_price"):
        car_val = float(car_row["Giá(VND)"])
        target_b = float(max_budget)
        car_text = f"{int(car_val):,} VNĐ".replace(",", ".")
        b_text = f"{int(target_b):,} VNĐ".replace(",", ".")
        if car_val <= target_b:
            savings = target_b - car_val
            savings_text = f"{int(savings):,} VNĐ".replace(",", ".")
            breakdown.append({
                "column": "Ngân sách",
                "type": "budget_filter",
                "target": f"Tối đa {b_text}",
                "car_value": car_text,
                "match_pct": 100.0,
                "is_good": True,
                "message": f"✓ Ngân sách: {car_text} nằm trong trần cho phép ({b_text}), tiết kiệm {savings_text}.",
            })

    # 2. Thuộc tính số
    for col in NUMERIC_COLS:
        # Nếu col là Giá(VND) nhưng chỉ là max_budget thì không phân tích độ gần trừ khi có target_price
        if col == "Giá(VND)":
            target_val = query.get("target_price") or (query.get("Giá(VND)") if not query.get("is_budget_only") else None)
        else:
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

    # 3. Thuộc tính phân loại
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
    Đảm bảo phân tách rạch ròi giữa bộ lọc ngân sách cứng và điểm tương đồng mềm.
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

        # Fit MinMaxScaler trên 4 đặc trưng số
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.df[NUMERIC_COLS].values)

        self.numeric_mins = {col: float(self.scaler.data_min_[i]) for i, col in enumerate(NUMERIC_COLS)}
        self.numeric_maxs = {col: float(self.scaler.data_max_[i]) for i, col in enumerate(NUMERIC_COLS)}

        # Xây dựng danh sách cột One-Hot cố định từ dữ liệu sạch
        self.cat_dummies = pd.get_dummies(self.df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, dtype=float)
        self.one_hot_feature_names: List[str] = list(self.cat_dummies.columns)

        self.cat_columns_map: Dict[str, List[str]] = {}
        for cat_col in CATEGORICAL_COLS:
            prefix = f"{cat_col}_"
            self.cat_columns_map[cat_col] = [c for c in self.one_hot_feature_names if c.startswith(prefix)]

        self.all_feature_names: List[str] = NUMERIC_COLS + self.one_hot_feature_names

        # Ma trận đặc trưng P cho toàn bộ dataset
        scaled_numeric = self.scaler.transform(self.df[NUMERIC_COLS].values)
        self.P = np.hstack([scaled_numeric, self.cat_dummies.values]).astype(np.float64)

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
        """Thống kê chi tiết về dataset phục vụ báo cáo và API model-info."""
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
            "features_list": self.all_feature_names,
        }

    def _encode_query(
        self,
        query: Dict[str, object],
        active_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Mã hóa vector truy vấn q cùng không gian đặc trưng với P.
        Nếu active_features được chỉ định, các chiều không active sẽ được đặt bằng 0.0.
        """
        # 1. Mã hóa đặc trưng số
        numeric_vec = []
        for i, col in enumerate(NUMERIC_COLS):
            val = query.get(col)
            # Nếu người dùng truyền target_price cho cột Giá(VND)
            if col == "Giá(VND)" and query.get("target_price") is not None:
                val = query.get("target_price")

            if val is None or str(val).strip() in ["", "None"]:
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
        
        Quy trình xử lý:
        1. Lọc cứng ngân sách: min_budget <= Price <= max_budget.
        2. Tách bạch ngân sách trần khỏi điểm tương đồng (không phạt xe rẻ hơn).
        3. Phân bổ trọng số từng chiều hợp lệ.
        4. Tính khoảng cách Weighted Euclidean.
        5. Xếp hạng ổn định có Tie-breaking (khoảng cách tăng dần, năm mới hơn, giá rẻ hơn).
        6. Tính điểm phù hợp tương đối Score = round(1 / (1 + d_w) * 100, 1).
        7. Giải thích tiêu chí chi tiết.
        """
        if k <= 0:
            raise ValueError(f"Số lượng kết quả k phải > 0, nhận được: {k}")

        if weights is None:
            weights = PURPOSE_WEIGHTS[DEFAULT_PURPOSE]

        # Kiểm tra tính hợp lệ của trọng số
        dim_weights = _per_dimension_weights(weights, self.cat_columns_map)

        # -------------------------------------------------------------------
        # BƯỚC 1: LỌC CỨNG (HARD FILTERS)
        # -------------------------------------------------------------------
        candidate_mask = np.ones(len(self.df), dtype=bool)

        # Phân biệt rõ ràng giữa max_budget (trần lọc) và target_price (giá mục tiêu)
        max_budget = query.get("max_budget")
        target_price = query.get("target_price")

        # Tương thích ngược:
        # Nếu truyền "Giá(VND)" hoặc "budget":
        # - Nếu có cờ is_budget_only (hoặc gọi qua API với max_budget): chỉ là trần lọc cứng
        # - Nếu không (vd: unit test toán học truyền query={"Giá(VND)": 50M}): vừa là trần vừa là target_price
        raw_price_val = query.get("Giá(VND)") if query.get("Giá(VND)") is not None else query.get("budget")
        if max_budget is None and raw_price_val is not None:
            max_budget = raw_price_val
            if target_price is None and not query.get("is_budget_only", False):
                target_price = raw_price_val

        if max_budget is not None:
            try:
                b_max = float(max_budget)
                if b_max > 0:
                    candidate_mask = candidate_mask & (self.df["Giá(VND)"].values <= b_max)
            except (ValueError, TypeError):
                pass

        # Ngân sách tối thiểu (nếu có)
        min_budget = query.get("min_budget")
        if min_budget is not None:
            try:
                b_min = float(min_budget)
                if b_min > 0:
                    candidate_mask = candidate_mask & (self.df["Giá(VND)"].values >= b_min)
            except (ValueError, TypeError):
                pass

        # Lọc cứng bắt buộc khác nếu có chỉ định (vd: mandatory_fuel)
        if query.get("mandatory_fuel") and query.get("mandatory_fuel") != "Không yêu cầu":
            candidate_mask = candidate_mask & (self.df["Nhiên liệu"].values == str(query["mandatory_fuel"]))

        candidate_indices = np.where(candidate_mask)[0]

        cols = [
            "Thứ hạng", "Tên xe", "Hãng", "Giá(VND)", "Kiểu xe",
            "Nhiên liệu", "Dung tích cốp (L)", "Công suất tối đa (Kw)",
            "Năm sản xuất", "Độ phù hợp (%)", "Khoảng cách KNN", "Chi tiết tiêu chí"
        ]

        if len(candidate_indices) == 0:
            return pd.DataFrame(columns=cols)

        # -------------------------------------------------------------------
        # BƯỚC 2: TÁCH BẠCH NGÂN SÁCH TRẦN & ĐO ĐỘ TƯƠNG ĐỒNG
        # -------------------------------------------------------------------
        eff_dim_weights = dim_weights.copy()
        query_for_encoding = query.copy()

        if target_price is not None:
            query_for_encoding["Giá(VND)"] = float(target_price)
        else:
            # Người dùng chỉ đặt trần ngân sách mà không chỉ định giá mong muốn
            # Không phạt xe rẻ hơn; ưu tiên xe kinh tế nếu prefer_economy
            if query.get("prefer_economy", True) and max_budget is not None:
                cheapest_in_cand = float(np.min(self.df.iloc[candidate_indices]["Giá(VND)"].values))
                query_for_encoding["Giá(VND)"] = cheapest_in_cand
            else:
                query_for_encoding["Giá(VND)"] = None

        # Mã hóa query
        q = self._encode_query(query_for_encoding)

        # -------------------------------------------------------------------
        # BƯỚC 3: TÍNH KHOẢNG CÁCH WEIGHTED EUCLIDEAN
        # -------------------------------------------------------------------
        P_cand = self.P[candidate_indices]
        diff = P_cand - q
        dist_sq = np.dot(diff ** 2, eff_dim_weights)
        dist_sq = np.maximum(dist_sq, 0.0)
        distances = np.sqrt(dist_sq)

        # -------------------------------------------------------------------
        # BƯỚC 4: SẮP XẾP CÓ TIE-BREAKING
        # -------------------------------------------------------------------
        # Quy tắc Tie-breaking:
        # 1) Khoảng cách KNN tăng dần (độ tương đồng cao nhất)
        # 2) Năm sản xuất giảm dần (xe đời mới hơn)
        # 3) Giá tiền tăng dần (xe tiết kiệm hơn)
        # 4) Tên model theo thứ tự bảng chữ cái
        cand_df = self.df.iloc[candidate_indices].copy()
        cand_df["_dist"] = distances
        cand_df["_orig_idx"] = candidate_indices

        cand_df = cand_df.sort_values(
            by=["_dist", "Năm sản xuất", "Giá(VND)", "Model"],
            ascending=[True, False, True, True]
        )

        num_to_take = min(k, len(cand_df))
        top_cand_df = cand_df.iloc[:num_to_take]

        # -------------------------------------------------------------------
        # BƯỚC 5: TẠO BẢNG KẾT QUẢ VÀ GIẢI THÍCH
        # -------------------------------------------------------------------
        results = []
        for rank, (_, row) in enumerate(top_cand_df.iterrows(), start=1):
            dist_val = float(row["_dist"])
            score = round((1.0 / (1.0 + dist_val)) * 100.0, 1)

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


CarRecommender = MotoRecommender
