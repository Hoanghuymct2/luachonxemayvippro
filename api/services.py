"""
Service layer for FastAPI application.
Handles business logic, recommender singleton, and LLM integrations.
"""

import os
from typing import Any, Dict, List, Optional
import pandas as pd

from core.recommender import (
    DEFAULT_PURPOSE,
    PURPOSE_DESCRIPTIONS,
    PURPOSE_WEIGHTS,
    MotoRecommender,
)
from llm.gemini_helper import (
    explain_results,
    get_model_status,
    parse_need,
)
from api.schemas import (
    CriteriaMatchItem,
    InterpretResponse,
    ModelInfoResponse,
    MotoRecommendationItem,
    RecommendRequest,
    RecommendResponse,
)

_RECOMMENDER_INSTANCE: Optional[MotoRecommender] = None


def get_recommender() -> MotoRecommender:
    """Khởi tạo và lưu giữ singleton instance của MotoRecommender."""
    global _RECOMMENDER_INSTANCE
    if _RECOMMENDER_INSTANCE is None:
        _RECOMMENDER_INSTANCE = MotoRecommender()
    return _RECOMMENDER_INSTANCE


def get_model_info_service() -> ModelInfoResponse:
    """Trả về metadata đầy đủ về mô hình và dữ liệu xe máy."""
    rec = get_recommender()
    stats = rec.get_stats()
    return ModelInfoResponse(
        model_name="Motorcycle_Weighted_KNN_DSS",
        version="2.0.0",
        algorithm="Weighted K-Nearest Neighbors (MCDM Proximity Ranking)",
        distance_metric="Weighted Euclidean Distance",
        dataset_records=int(stats["total_rows"]),
        feature_dimensions=int(stats["feature_dims"]),
        purpose_profiles=list(PURPOSE_WEIGHTS.keys()),
        default_purpose=DEFAULT_PURPOSE,
        valid_categories=rec.valid_categories,
        numeric_features=rec.df[stats.get("numeric_features", ["Giá(VND)", "Công suất tối đa (Kw)", "Dung tích cốp (L)", "Năm sản xuất"])].columns.tolist(),
        categorical_features=["Hãng", "Nhiên liệu", "Kiểu xe", "Tình trạng"],
        price_stats={
            "min_vnd": float(stats["price_min"]),
            "max_vnd": float(stats["price_max"]),
            "median_vnd": float(stats["price_median"]),
        },
        power_stats={
            "min_kw": float(stats["power_min"]),
            "max_kw": float(stats["power_max"]),
            "median_kw": float(stats["power_median"]),
        },
        trunk_stats={
            "min_liters": float(stats["trunk_min"]),
            "max_liters": float(stats["trunk_max"]),
            "median_liters": float(stats["trunk_median"]),
        },
    )


def recommend_service(req: RecommendRequest) -> RecommendResponse:
    """Xử lý yêu cầu gợi ý xe từ client."""
    rec = get_recommender()

    # Xây dựng query dictionary cho engine
    query: Dict[str, Any] = {
        "max_budget": req.max_budget,
        "min_budget": req.min_budget,
        "target_price": req.target_price,
        "prefer_economy": req.prefer_economy,
        "Dung tích cốp (L)": req.trunk_liters,
        "Công suất tối đa (Kw)": req.power_kw,
        "Năm sản xuất": float(req.year) if req.year else None,
        "Hãng": req.brand,
        "Kiểu xe": req.body,
        "Nhiên liệu": req.fuel,
        "Tình trạng": req.condition,
        "mandatory_fuel": req.mandatory_fuel,
        "is_budget_only": (req.max_budget is not None and req.target_price is None),
    }

    # Lấy bộ trọng số tương ứng mục đích
    purpose_to_use = req.purpose if req.purpose in PURPOSE_WEIGHTS else DEFAULT_PURPOSE
    weights = PURPOSE_WEIGHTS[purpose_to_use]

    # Thực thi Weighted KNN
    results_df = rec.recommend(query=query, weights=weights, k=req.k)

    # Chuyển đổi sang Pydantic schema
    items: List[MotoRecommendationItem] = []
    for _, row in results_df.iterrows():
        raw_bd = row.get("Chi tiết tiêu chí", [])
        bd_items = [
            CriteriaMatchItem(
                column=c["column"],
                type=c["type"],
                target=c["target"],
                car_value=c["car_value"],
                match_pct=float(c["match_pct"]),
                is_good=bool(c["is_good"]),
                message=c["message"],
            )
            for c in raw_bd
        ]

        price_val = float(row["Giá(VND)"])
        items.append(
            MotoRecommendationItem(
                rank=int(row["Thứ hạng"]),
                name=str(row["Tên xe"]),
                brand=str(row["Hãng"]),
                price=price_val,
                price_formatted=f"{int(price_val):,} VNĐ".replace(",", "."),
                body=str(row["Kiểu xe"]),
                fuel=str(row["Nhiên liệu"]),
                trunk_liters=float(row["Dung tích cốp (L)"]),
                power_kw=float(row["Công suất tối đa (Kw)"]),
                year=int(row["Năm sản xuất"]),
                fit_score=float(row["Độ phù hợp (%)"]),
                knn_distance=float(row["Khoảng cách KNN"]),
                criteria_breakdown=bd_items,
            )
        )

    # Giải thích Gemini nếu client yêu cầu
    explanation_text = None
    if req.with_explanation and len(items) > 0:
        explanation_text = explain_results(
            purpose=purpose_to_use,
            weights=weights,
            query=query,
            result=results_df,
        )

    # Ghi nhận các bộ lọc đã áp dụng
    applied = {
        "purpose": purpose_to_use,
        "max_budget": req.max_budget,
        "min_budget": req.min_budget,
        "target_price": req.target_price,
        "trunk_liters": req.trunk_liters,
        "power_kw": req.power_kw,
        "brand": req.brand,
        "body": req.body,
        "fuel": req.fuel,
        "year": req.year,
        "k": req.k,
    }

    return RecommendResponse(
        purpose=purpose_to_use,
        total_results=len(items),
        applied_filters=applied,
        results=items,
        explanation=explanation_text,
    )


def interpret_service(text: str) -> InterpretResponse:
    """Phân tích câu tự nhiên bằng Gemini (hoặc fallback) thành bộ tiêu chí JSON."""
    rec = get_recommender()
    options = {
        "Hãng": rec.unique_values("Hãng"),
        "Nhiên liệu": rec.unique_values("Nhiên liệu"),
        "Kiểu xe": rec.unique_values("Kiểu xe"),
        "Tình trạng": rec.unique_values("Tình trạng"),
    }
    parsed = parse_need(text=text, options=options)
    return InterpretResponse(
        purpose=parsed.get("purpose", DEFAULT_PURPOSE),
        budget_trieu=parsed.get("budget_trieu"),
        budget_vnd=parsed.get("budget_vnd"),
        trunk_liters=parsed.get("trunk_liters"),
        power_kw=parsed.get("power_kw"),
        year=parsed.get("year"),
        brand=parsed.get("brand"),
        body=parsed.get("body"),
        fuel=parsed.get("fuel"),
        condition=parsed.get("condition", "Mới"),
        source=parsed.get("source", "fallback"),
        inferred_trunk=parsed.get("inferred_trunk", False),
    )


def explain_service(purpose: str, results: List[Dict[str, Any]], query: Optional[Dict[str, Any]] = None) -> str:
    """Tạo đoạn văn giải thích kết quả gợi ý từ chuyên gia DSS / Gemini."""
    purpose_to_use = purpose if purpose in PURPOSE_WEIGHTS else DEFAULT_PURPOSE
    weights = PURPOSE_WEIGHTS[purpose_to_use]
    return explain_results(
        purpose=purpose_to_use,
        weights=weights,
        query=query or {},
        result=results,
    )
