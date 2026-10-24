"""
REST API for Motorcycle Decision Support System (FastAPI).
Consolidated into a single, clean module.
Run standalone with: uvicorn api:app --host 127.0.0.1 --port 8000 --reload
"""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from recommender import (
    ALL_GROUPS,
    CATEGORICAL_COLS,
    DEFAULT_PURPOSE,
    NUMERIC_COLS,
    PURPOSE_DESCRIPTIONS,
    PURPOSE_WEIGHTS,
    MotoRecommender,
)
import gemini_helper as gh

# ---------------------------------------------------------------------------
# 1. Pydantic Schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    purpose: str = Field(default=DEFAULT_PURPOSE)
    max_budget: Optional[float] = Field(default=None, ge=0)
    target_price: Optional[float] = Field(default=None, ge=0)
    brand: Optional[str] = None
    fuel: Optional[str] = None
    body_style: Optional[str] = None
    min_power: Optional[float] = Field(default=None, ge=0)
    min_trunk: Optional[float] = Field(default=None, ge=0)
    year: Optional[int] = Field(default=None, ge=1990, le=2030)
    k: int = Field(default=5, ge=1, le=50)


class MotorcycleItem(BaseModel):
    rank: int
    model: str
    brand: str
    price: float
    fuel: str
    body_style: str
    power_kw: float
    trunk_liters: float
    year: int
    distance: float
    score: float
    criteria_breakdown: List[str]


class RecommendResponse(BaseModel):
    total_matched: int
    k: int
    purpose: str
    results: List[MotorcycleItem]


class InterpretRequest(BaseModel):
    text: str = Field(..., min_length=1)


class InterpretResponse(BaseModel):
    purpose: str
    max_budget: Optional[float] = None
    target_price: Optional[float] = None
    brand: Optional[str] = None
    fuel: Optional[str] = None
    body_style: Optional[str] = None
    min_trunk: Optional[float] = None
    min_power: Optional[float] = None
    k: int = 5
    source: str = "gemini"


class ExplainRequest(BaseModel):
    recommendations: List[Dict[str, Any]]
    user_need: str
    purpose: Optional[str] = None


class ExplainResponse(BaseModel):
    explanation: str
    source: str = "gemini"


# ---------------------------------------------------------------------------
# 2. FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Motorcycle DSS REST API",
    description="Hệ Thống Hỗ Trợ Ra Quyết Định Gợi Ý Xe Máy (KNN + Gemini)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_recommender: Optional[MotoRecommender] = None


def get_recommender() -> MotoRecommender:
    global _recommender
    if _recommender is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
        _recommender = MotoRecommender(csv_path)
    return _recommender


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    rec = get_recommender()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "records_count": len(rec.df),
        "feature_dim": len(rec.all_feature_names),
    }


@app.get("/model-info", tags=["System"])
def model_info() -> Dict[str, Any]:
    rec = get_recommender()
    stats = rec.get_stats()
    return {
        "version": "2.0.0",
        "dataset_records": stats["total_rows"],
        "feature_dimensions": stats["feature_dims"],
        "purposes": list(PURPOSE_WEIGHTS.keys()),
        "brands": rec.unique_values("Hãng"),
        "fuels": rec.unique_values("Nhiên liệu"),
        "body_styles": rec.unique_values("Kiểu xe"),
        "numeric_ranges": {
            "Giá(VND)": {"min": stats["price_min"], "max": stats["price_max"]},
            "Công suất tối đa (Kw)": {"min": stats["power_min"], "max": stats["power_max"]},
            "Dung tích cốp (L)": {"min": stats["trunk_min"], "max": stats["trunk_max"]},
            "Năm sản xuất": {"min": 2022.0, "max": 2024.0},
        },
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
def recommend_motorcycles(req: RecommendRequest) -> RecommendResponse:
    rec = get_recommender()
    weights = PURPOSE_WEIGHTS.get(req.purpose, PURPOSE_WEIGHTS[DEFAULT_PURPOSE])

    query_dict: Dict[str, Any] = {
        "max_budget": req.max_budget,
        "target_price": req.target_price,
        "is_budget_only": (req.target_price is None and req.max_budget is not None),
        "Hãng": req.brand if req.brand and req.brand != "Không yêu cầu" else None,
        "Nhiên liệu": req.fuel if req.fuel and req.fuel != "Không yêu cầu" else None,
        "Kiểu xe": req.body_style if req.body_style and req.body_style != "Không yêu cầu" else None,
        "Công suất tối đa (Kw)": req.min_power,
        "Dung tích cốp (L)": req.min_trunk,
        "Năm sản xuất": float(req.year) if req.year else None,
    }

    try:
        results_df = rec.recommend(query=query_dict, weights=weights, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    items: List[MotorcycleItem] = []
    if not results_df.empty:
        for _, row in results_df.iterrows():
            breakdown_list = [
                item.get("message", "") if isinstance(item, dict) else str(item)
                for item in row.get("Chi tiết tiêu chí", [])
            ]
            items.append(MotorcycleItem(
                rank=int(row["Thứ hạng"]),
                model=str(row["Tên xe"]),
                brand=str(row["Hãng"]),
                price=float(row["Giá(VND)"]),
                fuel=str(row["Nhiên liệu"]),
                body_style=str(row["Kiểu xe"]),
                power_kw=float(row["Công suất tối đa (Kw)"]),
                trunk_liters=float(row["Dung tích cốp (L)"]),
                year=int(row["Năm sản xuất"]),
                distance=float(row["Khoảng cách KNN"]),
                score=float(row["Độ phù hợp (%)"]),
                criteria_breakdown=breakdown_list,
            ))

    return RecommendResponse(
        total_matched=len(results_df),
        k=req.k,
        purpose=req.purpose,
        results=items,
    )


@app.post("/interpret", response_model=InterpretResponse, tags=["AI Assistance"])
def interpret_natural_language(req: InterpretRequest) -> InterpretResponse:
    parsed = gh.parse_need(req.text)
    return InterpretResponse(
        purpose=parsed.get("purpose", DEFAULT_PURPOSE),
        max_budget=parsed.get("max_budget") or parsed.get("Giá(VND)"),
        target_price=parsed.get("target_price"),
        brand=parsed.get("brand") or parsed.get("Hãng"),
        fuel=parsed.get("fuel") or parsed.get("Nhiên liệu"),
        body_style=parsed.get("body_style") or parsed.get("Kiểu xe"),
        min_trunk=parsed.get("min_trunk") or parsed.get("Dung tích cốp (L)"),
        min_power=parsed.get("min_power") or parsed.get("Công suất tối đa (Kw)"),
        k=int(parsed.get("k", 5)),
        source=parsed.get("source", "gemini"),
    )


@app.post("/explain", response_model=ExplainResponse, tags=["AI Assistance"])
def explain_recommendations(req: ExplainRequest) -> ExplainResponse:
    candidates_df = pd.DataFrame(req.recommendations)
    explanation, source = gh.explain_results(
        top_candidates=candidates_df,
        user_need=req.user_need,
        purpose=req.purpose or DEFAULT_PURPOSE,
    )
    return ExplainResponse(explanation=explanation, source=source)
