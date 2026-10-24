"""
Unit tests for search history management (test_history.py).
"""

import json
import os
import tempfile
import pytest

import history as hist


@pytest.fixture
def temp_history_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_save_and_load_history(temp_history_file):
    """Kiểm tra lưu và đọc lịch sử tìm kiếm."""
    # Khi file chưa tồn tại -> trả về rỗng
    assert hist.load_history(temp_history_file) == []

    purpose = "Học sinh, Sinh viên"
    query = {"Giá(VND)": 45000000.0, "Kiểu xe": "Tay ga"}
    weights = {"Giá(VND)": 0.35, "Kiểu xe": 0.20}
    top_bikes = ["Honda Vision", "Yamaha Janus"]

    hist.save_search(purpose, query, weights, top_bikes, file_path=temp_history_file)

    loaded = hist.load_history(temp_history_file)
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec["muc_dich"] == purpose
    assert rec["tieu_chi"] == query
    assert rec["danh_sach_xe"] == top_bikes
    assert "timestamp" in rec


def test_no_sensitive_keys_saved(temp_history_file):
    """Bảo đảm tuyệt đối không lưu API key hoặc token vào lịch sử."""
    query = {
        "Giá(VND)": 50000000.0,
        "api_key": "AIzaSySecretDoNotSave",
        "secret": "my_secret_token",
    }
    hist.save_search("Đi làm", query, {}, ["Xe 1"], file_path=temp_history_file)

    loaded = hist.load_history(temp_history_file)
    saved_query = loaded[0]["tieu_chi"]
    assert "api_key" not in saved_query
    assert "secret" not in saved_query
    assert "Giá(VND)" in saved_query


def test_prevent_duplicate_rerun_saves(temp_history_file):
    """Kiểm tra không lưu trùng lặp khi Streamlit rerun với cùng thông số."""
    purpose = "Học sinh, Sinh viên"
    query = {"Giá(VND)": 45000000.0}
    weights = {"Giá(VND)": 1.0}
    top_bikes = ["Honda Vision"]

    # Lưu 3 lần liên tiếp giống nhau
    hist.save_search(purpose, query, weights, top_bikes, file_path=temp_history_file)
    hist.save_search(purpose, query, weights, top_bikes, file_path=temp_history_file)
    hist.save_search(purpose, query, weights, top_bikes, file_path=temp_history_file)

    loaded = hist.load_history(temp_history_file)
    # Chỉ lưu duy nhất 1 bản ghi
    assert len(loaded) == 1


def test_clear_history(temp_history_file):
    """Kiểm tra xóa toàn bộ lịch sử."""
    hist.save_search("Học sinh", {"Giá(VND)": 30000000.0}, {}, ["Xe A"], file_path=temp_history_file)
    assert len(hist.load_history(temp_history_file)) == 1

    hist.clear_history(temp_history_file)
    assert hist.load_history(temp_history_file) == []


def test_corrupt_history_file_handling(temp_history_file):
    """Kiểm tra xử lý an toàn khi file JSON bị hỏng."""
    with open(temp_history_file, "w", encoding="utf-8") as f:
        f.write("{invalid_json_format...")

    # Phải trả về danh sách rỗng, không làm sập chương trình
    loaded = hist.load_history(temp_history_file)
    assert loaded == []

