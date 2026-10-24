"""
Unit tests for data reading, cleaning, and preprocessing (test_data.py).
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from recommender import (
    ALL_GROUPS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    _parse_power_value,
    _parse_price_value,
    _parse_trunk_value,
    load_and_clean_data,
)


def test_parse_price_formats():
    """Kiểm tra parse các định dạng giá tiền khác nhau."""
    assert _parse_price_value(15000000) == 15000000.0
    assert _parse_price_value("166.190.000") == 166190000.0
    assert _parse_price_value("1.231.000.000") == 1231000000.0
    assert _parse_price_value("41.931.818") == 41931818.0
    assert _parse_price_value("50000000") == 50000000.0
    assert _parse_price_value("50,000,000") == 50000000.0
    assert _parse_price_value("15.000.000,50") == 15000000.5
    assert np.isnan(_parse_price_value(None))
    assert np.isnan(_parse_price_value(""))


def test_parse_trunk_formats():
    """Kiểm tra parse các định dạng dung tích cốp."""
    assert _parse_trunk_value("28 L") == 28.0
    assert _parse_trunk_value("22,5 L") == 22.5
    assert _parse_trunk_value("0 L") == 0.0
    assert _parse_trunk_value(37) == 37.0
    assert _parse_trunk_value("18.5 l") == 18.5
    assert np.isnan(_parse_trunk_value(None))


def test_parse_power_formats():
    """Kiểm tra parse các định dạng công suất tối đa."""
    assert _parse_power_value(21.5) == 21.5
    assert _parse_power_value("21,5") == 21.5
    assert _parse_power_value("11.3 kW") == 11.3
    assert _parse_power_value("0.5") == 0.5
    assert np.isnan(_parse_power_value(None))


def test_load_and_clean_real_dataset():
    """Kiểm tra đọc và làm sạch dataset.csv thực tế trong workspace."""
    csv_path = "dataset.csv"
    if not os.path.exists(csv_path):
        pytest.skip("dataset.csv không tồn tại trong thư mục kiểm thử.")

    df = load_and_clean_data(csv_path)

    # 1. Số dòng hợp lệ
    assert len(df) == 225

    # 2. Hãng được forward-fill đúng, không còn nan/none
    assert df["Hãng"].isnull().sum() == 0
    assert "nan" not in df["Hãng"].str.lower().values
    assert len(df["Hãng"].unique()) == 9

    # 3. Model prefix: mọi model đều bắt đầu bằng tên Hãng
    for _, row in df.iterrows():
        b = str(row["Hãng"]).strip().lower()
        m = str(row["Model"]).strip().lower()
        assert m.startswith(b), f"Model '{row['Model']}' không bắt đầu bằng Hãng '{row['Hãng']}'"
        # Không có tiền tố lặp lại (vd: Honda Honda)
        assert not m.startswith(f"{b} {b}"), f"Model '{row['Model']}' bị lặp tiền tố hãng!"

    # 4. Kiểm tra các cột số
    assert df["Giá(VND)"].isnull().sum() == 0
    assert (df["Giá(VND)"] > 0).all()
    assert (df["Công suất tối đa (Kw)"] >= 0).all()
    assert (df["Dung tích cốp (L)"] >= 0).all()
    assert (df["Năm sản xuất"] >= 2020).all()

    # 5. Không phát sinh NaN hoặc inf
    for col in NUMERIC_COLS:
        assert not np.isinf(df[col]).any()
        assert not df[col].isnull().any()


def test_clean_data_handles_missing_and_corrupt():
    """Kiểm tra hàm làm sạch dữ liệu xử lý an toàn khi có dòng lỗi/bất thường."""
    dummy_csv = """Hãng,Model,Giá(VND),Công suất tối đa (Kw),Nhiên liệu,Kiểu xe,Dung tích cốp,Tình trạng,Năm sản xuất
Honda,Vision,31.000.000,6.59,Xăng,Tay ga,16 L,Mới,2024
,Air Blade,56.000.000,11.2,Xăng,Tay ga,23 L,Mới,2024
,Xe lỗi giá,-10.000,5.0,Xăng,Tay ga,10 L,Mới,2024
,Xe lỗi công suất,40.000.000,-2.0,Xăng,Tay ga,10 L,Mới,2024
,Xe lỗi năm,40.000.000,5.0,Xăng,Tay ga,10 L,Mới,1800
"""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".csv") as f:
        f.write(dummy_csv)
        temp_path = f.name

    try:
        cleaned_df = load_and_clean_data(temp_path)
        # Các dòng lỗi bị loại bỏ, chỉ còn 2 dòng hợp lệ
        assert len(cleaned_df) == 2
        assert "Honda Vision" in cleaned_df["Model"].values
        assert "Honda Air Blade" in cleaned_df["Model"].values
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

