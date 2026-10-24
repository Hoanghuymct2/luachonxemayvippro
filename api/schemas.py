"""
Pydantic schemas for the FastAPI layer.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    recommender_ready: bool
    total_motorcycles: int


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    algorithm: str
    distance_metric: str
    dataset_records: int
    feature_dimensions: int
    purpose_profiles: List[str]
    default_purpose: str
    valid_categories: Dict[str, List[str]]
    numeric_features: List[str]
    categorical_features: List[str]
    price_stats: Dict[str, float]
    power_stats: Dict[str, float]
    trunk_stats: Dict[str, float]


class RecommendRequest(BaseModel):
    purpose: str = Field(default="Đi làm, đi phố & Gia đình", description="Hồ sơ mục đích sử dụng")
    max_budget: Optional[float] = Field(default=None, description="Ngân sách tối đa (VND) - Lọc trần")
    min_budget: Optional[float] = Field(default=None, description="Ngân sách tối thiểu (VND)")
    target_price: Optional[float] = Field(default=None, description="Mức giá mong muốn cụ thể (nếu có)")
    prefer_economy: bool = Field(default=True, description="Ưu tiên xe tiết kiệm chi phí hơn trong trần ngân sách")
    trunk_liters: Optional[float] = Field(default=None, description="Dung tích cốp mong muốn (Lít)")
    power_kw: Optional[float] = Field(default=None, description="Công suất mong muốn (Kw)")
    year: Optional[int] = Field(default=None, description="Năm sản xuất")
    brand: Optional[str] = Field(default=None, description="Thương hiệu yêu cầu")
    body: Optional[str] = Field(default=None, description="Kiểu xe yêu cầu")
    fuel: Optional[str] = Field(default=None, description="Loại nhiên liệu")
    condition: Optional[str] = Field(default=None, description="Tình trạng xe")
    mandatory_fuel: Optional[str] = Field(default=None, description="Ràng buộc cứng loại nhiên liệu")
    k: int = Field(default=5, ge=1, le=50, description="Số lượng xe gợi ý cần trả về")
    with_explanation: bool = Field(default=False, description="Tự động gọi Gemini để giải thích kết quả")

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("k phải lớn hơn hoặc bằng 1")
        return v


class CriteriaMatchItem(BaseModel):
    column: str
    type: str
    target: str
    car_value: str
    match_pct: float
    is_good: bool
    message: str


class MotoRecommendationItem(BaseModel):
    rank: int
    name: str
    brand: str
    price: float
    price_formatted: str
    body: str
    fuel: str
    trunk_liters: float
    power_kw: float
    year: int
    fit_score: float
    knn_distance: float
    criteria_breakdown: List[CriteriaMatchItem]


class RecommendResponse(BaseModel):
    purpose: str
    total_results: int
    applied_filters: Dict[str, Any]
    results: List[MotoRecommendationItem]
    explanation: Optional[str] = None


class InterpretRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Câu mô tả nhu cầu tự nhiên của người dùng")


class InterpretResponse(BaseModel):
    purpose: str
    budget_trieu: Optional[float] = None
    budget_vnd: Optional[float] = None
    trunk_liters: Optional[float] = None
    power_kw: Optional[float] = None
    year: Optional[int] = None
    brand: Optional[str] = None
    body: Optional[str] = None
    fuel: Optional[str] = None
    condition: Optional[str] = "Mới"
    source: str
    inferred_trunk: bool = False


class ExplainRequest(BaseModel):
    purpose: str
    query: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]]
