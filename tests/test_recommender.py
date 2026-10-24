"""
Unit tests for the Weighted KNN Decision Support Model (test_recommender.py).
Includes an independent synthetic dataset with hand-calculated expected values.
"""

import math
import numpy as np
import pandas as pd
import pytest

from recommender import (
    ALL_GROUPS,
    MotoRecommender,
    _criteria_breakdown,
)


@pytest.fixture
def synthetic_recommender():
    """Tạo một tập dữ liệu giả lập độc lập với các thông số được định sẵn để kiểm chứng bằng tay."""
    data = {
        "Hãng": ["Honda", "Yamaha", "VinFast"],
        "Model": ["Honda Wave", "Yamaha Grande", "VinFast Evo"],
        "Giá(VND)": [20_000_000.0, 50_000_000.0, 30_000_000.0],
        "Công suất tối đa (Kw)": [6.0, 6.0, 6.0],  # Min=Max=6.0
        "Dung tích cốp (L)": [10.0, 30.0, 20.0],   # Min=10, Max=30
        "Năm sản xuất": [2024.0, 2024.0, 2024.0],  # Min=Max=2024.0
        "Nhiên liệu": ["Xăng", "Xăng", "Điện"],
        "Kiểu xe": ["Xe số", "Tay ga", "Xe điện"],
        "Tình trạng": ["Mới", "Mới", "Mới"],
    }
    df = pd.DataFrame(data)
    return MotoRecommender(df)


def test_independent_mathematical_hand_calculation(synthetic_recommender):
    """
    Kiểm tra công thức Weighted Euclidean Distance và Điểm phù hợp
    bằng phép tính giải tích tính tay độc lập (không gọi lại code nội bộ).
    
    Thông số truy vấn:
    - Giá: 50.000.000 VNĐ
    - Kiểu xe: "Tay ga"
    
    Trọng số:
    - Giá(VND): 0.40
    - Kiểu xe: 0.60 (chia đều 3 chiều One-Hot -> mỗi chiều 0.20)
    - 6 thuộc tính khác: 0.00
    
    Kết quả tính tay:
    1. Yamaha Grande: d_w = 0.0, Score = 100.0%
    2. VinFast Evo: d_w = sqrt(4/9 * 0.4 + 2 * 0.2) = sqrt(0.577777...) ≈ 0.7601, Score = 56.8%
    3. Honda Wave: d_w = sqrt(1.0 * 0.4 + 2 * 0.2) = sqrt(0.8) ≈ 0.8944, Score = 52.8%
    """
    rec = synthetic_recommender

    weights = {g: 0.0 for g in ALL_GROUPS}
    weights["Giá(VND)"] = 0.40
    weights["Kiểu xe"] = 0.60

    query = {
        "Giá(VND)": 50_000_000.0,
        "Kiểu xe": "Tay ga",
    }

    res = rec.recommend(query=query, weights=weights, k=3)

    assert len(res) == 3

    # 1. Top 1: Yamaha Grande
    row1 = res.iloc[0]
    assert row1["Tên xe"] == "Yamaha Grande"
    assert abs(row1["Khoảng cách KNN"] - 0.0) < 1e-4
    assert row1["Độ phù hợp (%)"] == 100.0

    # 2. Top 2: VinFast Evo
    row2 = res.iloc[1]
    assert row2["Tên xe"] == "VinFast Evo"
    expected_dist_evo = math.sqrt((4.0 / 9.0) * 0.40 + 0.40)  # ≈ 0.760117
    expected_score_evo = round((1.0 / (1.0 + expected_dist_evo)) * 100.0, 1)  # 56.8%
    assert abs(row2["Khoảng cách KNN"] - round(expected_dist_evo, 4)) < 1e-4
    assert row2["Độ phù hợp (%)"] == expected_score_evo

    # 3. Top 3: Honda Wave
    row3 = res.iloc[2]
    assert row3["Tên xe"] == "Honda Wave"
    expected_dist_wave = math.sqrt(1.0 * 0.40 + 0.40)  # = sqrt(0.8) ≈ 0.894427
    expected_score_wave = round((1.0 / (1.0 + expected_dist_wave)) * 100.0, 1)  # 52.8%
    assert abs(row3["Khoảng cách KNN"] - round(expected_dist_wave, 4)) < 1e-4
    assert row3["Độ phù hợp (%)"] == expected_score_wave

    # Kiểm tra thứ tự khoảng cách tăng dần
    assert res.iloc[0]["Khoảng cách KNN"] <= res.iloc[1]["Khoảng cách KNN"] <= res.iloc[2]["Khoảng cách KNN"]


def test_hard_budget_filter(synthetic_recommender):
    """Kiểm tra bộ lọc ngân sách cứng: xe vượt ngân sách tuyệt đối không xuất hiện."""
    rec = synthetic_recommender

    weights = {g: 0.0 for g in ALL_GROUPS}
    weights["Giá(VND)"] = 0.50
    weights["Kiểu xe"] = 0.50

    # Ngân sách 35 triệu: Yamaha Grande (50 triệu) phải bị loại bỏ hoàn toàn
    query = {"Giá(VND)": 35_000_000.0}
    res = rec.recommend(query=query, weights=weights, k=5)

    assert len(res) == 2
    for p in res["Giá(VND)"]:
        assert p <= 35_000_000.0
    assert "Yamaha Grande" not in res["Tên xe"].values


def test_empty_results_when_budget_too_low(synthetic_recommender):
    """Kiểm tra khi ngân sách thấp hơn tất cả xe trong hệ thống -> trả về kết quả rỗng."""
    rec = synthetic_recommender
    query = {"Giá(VND)": 10_000_000.0}  # Không có xe nào dưới 10M
    res = rec.recommend(query=query, k=5)
    assert len(res) == 0
    assert isinstance(res, pd.DataFrame)


def test_k_larger_than_candidate_count(synthetic_recommender):
    """Kiểm tra k lớn hơn số xe thỏa mãn: trả về đúng số lượng xe còn lại."""
    rec = synthetic_recommender
    query = {"Giá(VND)": 35_000_000.0}  # Còn 2 xe
    res = rec.recommend(query=query, k=10)
    assert len(res) == 2


def test_invalid_categorical_query_raises_error(synthetic_recommender):
    """Kiểm tra truy vấn phân loại chứa giá trị ngoài từ vựng phải báo lỗi rõ ràng."""
    rec = synthetic_recommender
    query = {"Kiểu xe": "Máy bay"}
    with pytest.raises(ValueError, match="không hợp lệ cho cột 'Kiểu xe'"):
        rec.recommend(query=query)


def test_invalid_k_raises_error(synthetic_recommender):
    """Kiểm tra k <= 0 báo lỗi."""
    rec = synthetic_recommender
    with pytest.raises(ValueError, match="k phải > 0"):
        rec.recommend(query={}, k=0)


def test_criteria_breakdown_messages(synthetic_recommender):
    """Kiểm tra các thông báo phân tích tiêu chí chi tiết đúng quy chuẩn."""
    rec = synthetic_recommender
    row = rec.df.iloc[0]  # Honda Wave: Giá 20M, Kiểu xe 'Xe số'

    # 1. Trùng khớp chính xác
    query_match = {"Giá(VND)": 20_000_000.0, "Kiểu xe": "Xe số"}
    bd = rec._criteria_breakdown(query_match, row)
    assert len(bd) == 2
    assert bd[0]["is_good"] is True
    assert "✓ Giá(VND):" in bd[0]["message"]
    assert bd[1]["is_good"] is True
    assert "✓ Kiểu xe: đúng yêu cầu (Xe số)." in bd[1]["message"]

    # 2. Không khớp
    query_mismatch = {"Kiểu xe": "Tay ga"}
    bd2 = rec._criteria_breakdown(query_mismatch, row)
    assert len(bd2) == 1
    assert bd2[0]["is_good"] is False
    assert "✗ Kiểu xe: xe là 'Xe số', khác với yêu cầu 'Tay ga'." in bd2[0]["message"]

