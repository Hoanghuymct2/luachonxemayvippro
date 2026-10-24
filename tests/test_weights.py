"""
Unit tests for purpose weights and per-dimension allocation (test_weights.py).
"""

import numpy as np
import pytest

from recommender import (
    ALL_GROUPS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    PURPOSE_DESCRIPTIONS,
    PURPOSE_WEIGHTS,
    _per_dimension_weights,
)


def test_purpose_weights_sum_to_one():
    """Kiểm tra cả 5 hồ sơ trọng số đều có tổng bằng 1.0 (sai số < 1e-9)."""
    assert len(PURPOSE_WEIGHTS) == 5

    for name, weights in PURPOSE_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"Hồ sơ '{name}' có tổng trọng số = {total} != 1.0"


def test_purpose_weights_structure():
    """Kiểm tra cả 5 hồ sơ có đủ 8 nhóm thuộc tính và không âm."""
    for name, weights in PURPOSE_WEIGHTS.items():
        # Đủ 8 nhóm
        assert set(weights.keys()) == set(ALL_GROUPS), f"Hồ sơ '{name}' thiếu hoặc thừa nhóm!"
        # Không có trọng số âm
        for k, v in weights.items():
            assert v >= 0.0, f"Hồ sơ '{name}' có trọng số âm tại {k}: {v}"


def test_purpose_descriptions_coverage():
    """Kiểm tra có đủ mô tả tiếng Việt cho cả 5 hồ sơ."""
    assert len(PURPOSE_DESCRIPTIONS) == 5
    for name in PURPOSE_WEIGHTS.keys():
        assert name in PURPOSE_DESCRIPTIONS
        desc = PURPOSE_DESCRIPTIONS[name]
        assert isinstance(desc, str) and len(desc.strip()) > 10


def test_per_dimension_weights_calculation():
    """Kiểm tra hàm phân bổ trọng số cho từng chiều ma trận đặc trưng."""
    cat_map = {
        "Hãng": ["Hãng_Honda", "Hãng_Yamaha"],  # m = 2
        "Nhiên liệu": ["Nhiên liệu_Xăng", "Nhiên liệu_Điện"],  # m = 2
        "Kiểu xe": ["Kiểu xe_Tay ga", "Kiểu xe_Xe số", "Kiểu xe_Xe côn tay"],  # m = 3
        "Tình trạng": ["Tình trạng_Mới"],  # m = 1
    }
    # 4 số + 2 + 2 + 3 + 1 = 12 chiều
    profile = PURPOSE_WEIGHTS["Đi làm, đi phố & Gia đình"]
    dim_w = _per_dimension_weights(profile, cat_map)

    assert len(dim_w) == 12
    # Tổng trọng số các chiều phải bằng đúng 1.0 (sai số < 1e-9)
    assert abs(np.sum(dim_w) - 1.0) < 1e-9

    # Kiểm tra phân bổ đều cho One-Hot của Kiểu xe (m=3)
    # Kiểu xe có trọng số nhóm = 0.20 -> mỗi chiều = 0.20 / 3
    ki_w = profile["Kiểu xe"] / 3.0
    # Vị trí của Kiểu xe trong vector: 4 số + 2 Hãng + 2 Nhiên liệu = index 8, 9, 10
    assert abs(dim_w[8] - ki_w) < 1e-9
    assert abs(dim_w[9] - ki_w) < 1e-9
    assert abs(dim_w[10] - ki_w) < 1e-9


def test_per_dimension_weights_validation_errors():
    """Kiểm tra các trường hợp bắt lỗi khi trọng số không hợp lệ."""
    cat_map = {"Hãng": ["Hãng_A"], "Nhiên liệu": ["Nhiên liệu_B"], "Kiểu xe": ["Kiểu xe_C"], "Tình trạng": ["Tình trạng_D"]}

    # 1. Thiếu nhóm
    incomplete = {"Giá(VND)": 1.0}
    with pytest.raises(ValueError, match="Thiếu trọng số"):
        _per_dimension_weights(incomplete, cat_map)

    # 2. Tổng trọng số != 1.0
    bad_sum = {g: 0.1 for g in ALL_GROUPS}  # 8 * 0.1 = 0.8 != 1.0
    with pytest.raises(ValueError, match="Tổng trọng số các nhóm phải bằng 1.0"):
        _per_dimension_weights(bad_sum, cat_map)

    # 3. Trọng số âm
    bad_neg = {g: 0.2 for g in ALL_GROUPS}
    bad_neg["Giá(VND)"] = -0.1
    bad_neg["Hãng"] = 0.5
    with pytest.raises(ValueError, match="không được âm"):
        _per_dimension_weights(bad_neg, cat_map)

