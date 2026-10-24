"""
FastAPI application entrypoint for Motorcycle DSS.
Provides REST API endpoints for model inspection, recommendations, and LLM interpretation.
"""

from contextlib import asynccontextmanager
import os
import sys

# Đảm bảo đường dẫn import thư viện từ thư mục gốc
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ExplainRequest,
    HealthResponse,
    InterpretRequest,
    InterpretResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
)
from api.services import (
    explain_service,
    get_model_info_service,
    get_recommender,
    interpret_service,
    recommend_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo nạp dữ liệu và fit scaler 1 lần duy nhất khi API khởi động
    try:
        rec = get_recommender()
        print(f"[FastAPI Lifespan] Nạp thành công {len(rec.df)} mẫu xe máy.")
    except Exception as e:
        print(f"[FastAPI Lifespan Lỗi] Không thể nạp recommender: {e}")
    yield
    print("[FastAPI Lifespan] Đang tắt dịch vụ API.")


app = FastAPI(
    title="Motorcycle Recommendation Decision Support System (DSS) API",
    description="REST API cho hệ thống hỗ trợ ra quyết định gợi ý xe máy kết hợp Weighted KNN và Google Gemini AI.",
    version="2.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS để cho phép Streamlit hoặc Client khác gọi vào
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Kiểm tra trạng thái hoạt động của API và tính sẵn sàng của Recommender."""
    try:
        rec = get_recommender()
        return HealthResponse(
            status="ok",
            version="2.0.0",
            recommender_ready=True,
            total_motorcycles=len(rec.df),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Dịch vụ gợi ý chưa sẵn sàng: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Lấy thông tin metadata về thuật toán, các cột đặc trưng, phân phối giá và hồ sơ trọng số."""
    try:
        return get_model_info_service()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi đọc metadata mô hình: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
def get_recommendation(request: RecommendRequest):
    """
    Tiếp nhận yêu cầu tìm xe và tính toán xếp hạng Weighted KNN.
    Bao gồm lọc cứng ngân sách, tách biệt trần ngân sách với target price, và giải thích tiêu chí.
    """
    try:
        return recommend_service(request)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý thuật toán gợi ý: {str(e)}"
        )


@app.post("/interpret", response_model=InterpretResponse, tags=["LLM"])
def interpret_natural_language(request: InterpretRequest):
    """
    Diễn giải câu mô tả nhu cầu tự nhiên tiếng Việt thành bộ tiêu chí JSON có cấu trúc.
    Tự động fallback sang phân tích luật/từ khóa nếu Gemini gặp sự cố.
    """
    try:
        return interpret_service(request.text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi phân tích nhu cầu: {str(e)}"
        )


@app.post("/explain", tags=["LLM"])
def explain_results_endpoint(request: ExplainRequest):
    """Tạo đoạn văn giải thích kết quả gợi ý chuyên sâu (150 - 220 từ) dựa trên kết quả thật."""
    try:
        explanation = explain_service(
            purpose=request.purpose,
            results=request.results,
            query=request.query
        )
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tạo giải thích: {str(e)}"
        )
