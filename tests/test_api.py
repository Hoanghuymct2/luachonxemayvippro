"""
Unit and integration tests for FastAPI layer (test_api.py).
"""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_api_health_check():
    """Kiểm tra endpoint GET /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["recommender_ready"] is True
    assert data["total_motorcycles"] == 225


def test_api_model_info():
    """Kiểm tra endpoint GET /model-info."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "Weighted_KNN" in data["model_name"]
    assert data["dataset_records"] == 225
    assert len(data["purpose_profiles"]) == 5
    assert "Honda" in data["valid_categories"]["Hãng"]
    assert "Điện" in data["valid_categories"]["Nhiên liệu"]
    assert data["price_stats"]["min_vnd"] > 0


def test_api_recommend_success():
    """Kiểm tra POST /recommend với yêu cầu thông thường."""
    payload = {
        "purpose": "Học sinh, Sinh viên",
        "max_budget": 45000000.0,
        "body": "Tay ga",
        "k": 3,
        "with_explanation": False
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 3
    assert len(data["results"]) == 3
    for item in data["results"]:
        assert item["price"] <= 45000000.0
        assert item["rank"] in [1, 2, 3]
        assert "fit_score" in item
        assert "knn_distance" in item


def test_api_recommend_hard_budget_boundary():
    """Kiểm tra lọc ngân sách cứng: xe bằng mức trần được chấp nhận, vượt trần bị loại."""
    payload = {
        "purpose": "Học sinh, Sinh viên",
        "max_budget": 33700000.0,  # Giá đúng bằng SYM New Attila 125
        "k": 5
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    for item in data["results"]:
        assert item["price"] <= 33700000.0


def test_api_recommend_empty_when_budget_too_low():
    """Kiểm tra ngân sách quá thấp trả về kết quả rỗng không lỗi."""
    payload = {
        "purpose": "Học sinh, Sinh viên",
        "max_budget": 5000000.0,  # 5 triệu
        "k": 5
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 0
    assert len(data["results"]) == 0


def test_api_recommend_validation_error_k_invalid():
    """Kiểm tra Pydantic bắt lỗi khi k < 1."""
    payload = {
        "purpose": "Học sinh, Sinh viên",
        "k": 0
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_api_interpret_natural_language():
    """Kiểm tra POST /interpret diễn giải nhu cầu tự nhiên."""
    payload = {
        "text": "Em là sinh viên cần tìm xe ga dưới 45 triệu cốp rộng"
    }
    response = client.post("/interpret", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["purpose"] == "Học sinh, Sinh viên"
    assert data["budget_trieu"] == 45.0
    assert data["budget_vnd"] == 45000000.0
    assert data["body"] == "Tay ga"
    assert data["source"] in ["gemini (gemini-3.1-flash-lite)", "fallback"] or "gemini" in data["source"]


def test_api_explain():
    """Kiểm tra POST /explain."""
    payload = {
        "purpose": "Đi làm, đi phố & Gia đình",
        "query": {"max_budget": 50000000.0},
        "results": [
            {
                "name": "SYM Tuscany 150",
                "brand": "SYM",
                "price": 45000000.0,
                "trunk_liters": 27.0,
                "power_kw": 11.0,
                "fit_score": 82.5,
                "body": "Tay ga",
                "fuel": "Xăng"
            }
        ]
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert len(data["explanation"]) > 20
