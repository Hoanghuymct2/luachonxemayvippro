"""
Unit tests for Gemini AI integration and Fallback mechanisms (test_gemini_helper.py).
Fully isolated with unittest.mock; does not consume Gemini API quota.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

import gemini_helper as gh
from recommender import PURPOSE_WEIGHTS


@pytest.fixture
def mock_options():
    return {
        "Hãng": ["Honda", "Yamaha", "VinFast", "Vespa", "SYM", "Suzuki"],
        "Nhiên liệu": ["Xăng", "Điện"],
        "Kiểu xe": ["Tay ga", "Xe số", "Xe điện", "Xe côn tay", "Tay ga (50cc)", "Xe số (50cc)"],
        "Tình trạng": ["Mới"],
    }


def test_fallback_purpose_keywords(mock_options):
    """Kiểm tra nhận diện mục đích sử dụng bằng từ khóa tiếng Việt trong fallback."""
    # 1. Học sinh, Sinh viên
    r1 = gh._parse_fallback("Em là sinh viên năm nhất cần tìm xe đi học", mock_options)
    assert r1["purpose"] == "Học sinh, Sinh viên"

    # 2. Dịch vụ (Grab/Be)
    r2 = gh._parse_fallback("Tìm xe số bền bỉ chạy grab giao hàng tiết kiệm xăng", mock_options)
    assert r2["purpose"] == "Chạy dịch vụ (Grab/Be/Giao hàng)"

    # 3. Phượt, Thể thao
    r3 = gh._parse_fallback("Cần xe côn tay đi phượt tốc độ máy bốc mạnh mẽ", mock_options)
    assert r3["purpose"] == "Đi phượt, Thể thao & Đam mê tốc độ"

    # 4. Thời trang, Sang trọng
    r4 = gh._parse_fallback("Cần xe thời trang sang trọng cao cấp dạo phố", mock_options)
    assert r4["purpose"] == "Thời trang, Sang trọng & Cao cấp"

    # 5. Mặc định
    r5 = gh._parse_fallback("Tìm mua xe máy đi lại bình thường", mock_options)
    assert r5["purpose"] == "Đi làm, đi phố & Gia đình"


def test_fallback_budget_parsing(mock_options):
    """Kiểm tra trích xuất ngân sách bằng các từ: triệu, tr, củ."""
    assert gh._parse_fallback("giá dưới 45 triệu", mock_options)["budget_trieu"] == 45.0
    assert gh._parse_fallback("tầm 35tr đổ lại", mock_options)["budget_trieu"] == 35.0
    assert gh._parse_fallback("ngân sách 25.5 củ", mock_options)["budget_trieu"] == 25.5
    assert gh._parse_fallback("tầm 22,5 triệu", mock_options)["budget_trieu"] == 22.5

    # Kiểm tra chuyển đổi sang VND đúng 1 lần
    res = gh._parse_fallback("giá dưới 45 triệu", mock_options)
    assert res["budget_vnd"] == 45_000_000.0


def test_fallback_trunk_and_inferred(mock_options):
    """Kiểm tra trích xuất dung tích cốp (lít) và suy luận cốp rộng."""
    # Trực tiếp có số
    r1 = gh._parse_fallback("cốp khoảng 28 lít", mock_options)
    assert r1["trunk_liters"] == 28.0
    assert r1["inferred_trunk"] is False

    # Suy luận từ cụm từ 'cốp rộng'
    r2 = gh._parse_fallback("cần xe ga cốp rộng đựng vừa balo", mock_options)
    assert r2["trunk_liters"] == 25.0
    assert r2["inferred_trunk"] is True


def test_fallback_body_and_fuel(mock_options):
    """Kiểm tra trích xuất kiểu xe và nhiên liệu."""
    r = gh._parse_fallback("xe điện vinfast pin trâu", mock_options)
    assert r["brand"] == "VinFast"
    assert r["fuel"] == "Điện"
    assert r["body"] == "Xe điện"

    r50 = gh._parse_fallback("cần xe ga 50cc cho học sinh", mock_options)
    assert r50["body"] == "Tay ga (50cc)"


def test_parse_need_mock_gemini_success(mock_options):
    """Kiểm tra parse_need khi Gemini trả JSON hợp lệ (Mocked)."""
    valid_gemini_json = """
    {
      "purpose": "Học sinh, Sinh viên",
      "budget_trieu": 40.0,
      "power_kw": null,
      "trunk_liters": 20.0,
      "year": 2024,
      "brand": "Honda",
      "fuel": "Xăng",
      "body": "Tay ga",
      "condition": "Mới"
    }
    """
    with patch("gemini_helper.get_gemini_client") as mock_get_client, \
         patch("gemini_helper._call_gemini_with_fallback", return_value=(valid_gemini_json, "gemini-mock")):
        mock_get_client.return_value = MagicMock()

        res = gh.parse_need("Em là sinh viên tìm xe ga Honda 40 triệu", mock_options, api_key="dummy_key")

        assert res["purpose"] == "Học sinh, Sinh viên"
        assert res["budget_trieu"] == 40.0
        assert res["budget_vnd"] == 40_000_000.0
        assert res["brand"] == "Honda"
        assert res["body"] == "Tay ga"
        assert res["year"] == 2024
        assert "gemini" in res["source"]


def test_parse_need_mock_gemini_corrupt_json(mock_options):
    """Kiểm tra parse_need tự động chuyển fallback khi Gemini trả JSON hỏng."""
    corrupt_output = "Xin chào! Đây là xe tôi nghĩ bạn thích: {bad_json..."
    with patch("gemini_helper.get_gemini_client") as mock_get_client, \
         patch("gemini_helper._call_gemini_with_fallback", return_value=(corrupt_output, "gemini-mock")):
        mock_get_client.return_value = MagicMock()

        res = gh.parse_need("Em là sinh viên cần xe 30 triệu", mock_options, api_key="dummy_key")

        assert res["source"] == "fallback"
        assert res["purpose"] == "Học sinh, Sinh viên"
        assert res["budget_trieu"] == 30.0


def test_parse_need_mock_gemini_exception(mock_options):
    """Kiểm tra parse_need tự động chuyển fallback khi phát sinh exception."""
    with patch("gemini_helper.get_gemini_client") as mock_get_client, \
         patch("gemini_helper._call_gemini_with_fallback", side_effect=RuntimeError("Mạng lỗi")):
        mock_get_client.return_value = MagicMock()

        res = gh.parse_need("Em cần tìm xe đi làm 50 triệu", mock_options, api_key="dummy_key")

        assert res["source"] == "fallback"
        assert res["purpose"] == "Đi làm, đi phố & Gia đình"
        assert res["budget_trieu"] == 50.0


def test_parse_need_no_api_key(mock_options):
    """Kiểm tra khi không có API key thì dùng fallback trực tiếp."""
    with patch("gemini_helper.get_gemini_api_key", return_value=None):
        res = gh.parse_need("Em cần tìm xe 45 triệu", mock_options)
        assert res["source"] == "fallback"
        assert res["budget_trieu"] == 45.0


def test_explain_fallback_normal_and_empty():
    """Kiểm tra hàm giải thích fallback với kết quả rỗng và kết quả bình thường."""
    weights = PURPOSE_WEIGHTS["Đi làm, đi phố & Gia đình"]

    # 1. Kết quả rỗng
    empty_df = pd.DataFrame()
    exp_empty = gh._explain_fallback("Đi làm, đi phố & Gia đình", weights, {}, empty_df)
    assert "Không có mẫu xe nào thỏa mãn" in exp_empty

    # 2. Kết quả có dữ liệu
    sample_df = pd.DataFrame([{
        "Tên xe": "Honda Vision",
        "Hãng": "Honda",
        "Giá(VND)": 31000000.0,
        "Kiểu xe": "Tay ga",
        "Nhiên liệu": "Xăng",
        "Dung tích cốp (L)": 16.0,
        "Công suất tối đa (Kw)": 6.59,
        "Năm sản xuất": 2024,
        "Độ phù hợp (%)": 88.5,
    }])
    exp_normal = gh._explain_fallback("Đi làm, đi phố & Gia đình", weights, {}, sample_df)
    assert "Honda Vision" in exp_normal
    assert "88.5%" in exp_normal
    assert "Lời khuyên thực tế" in exp_normal
