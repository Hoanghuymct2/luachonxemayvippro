"""
Script to fit preprocessing artifacts, validate data quality, and export model metadata.
"""

from datetime import datetime
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Đảm bảo import được từ thư mục gốc
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np

from core.recommender import (
    ALL_GROUPS,
    CATEGORICAL_COLS,
    DEFAULT_PURPOSE,
    NUMERIC_COLS,
    PURPOSE_WEIGHTS,
    load_and_clean_data,
    MotoRecommender,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(MODEL_DIR, "knn_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def train_and_export():
    print("=== EXPORTING MOTORCYCLE DSS MODEL ARTIFACTS ===")
    
    # 1. Khởi tạo recommender từ dataset chuẩn
    rec = MotoRecommender()
    stats = rec.get_stats()
    
    # 2. Đóng gói artifact (scaler, danh sách feature, ma trận đặc trưng P, metadata map)
    artifact = {
        "scaler": rec.scaler,
        "numeric_mins": rec.numeric_mins,
        "numeric_maxs": rec.numeric_maxs,
        "all_feature_names": rec.all_feature_names,
        "one_hot_feature_names": rec.one_hot_feature_names,
        "cat_columns_map": rec.cat_columns_map,
        "valid_categories": rec.valid_categories,
        "num_records": len(rec.df),
    }
    
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"-> Đã xuất artifact mô hình: {ARTIFACT_PATH}")
    
    # 3. Tạo metadata JSON ghi nhận thông tin thực tế
    metadata = {
        "model_name": "Motorcycle_Weighted_KNN_DSS",
        "version": "2.0.0",
        "algorithm": "Weighted K-Nearest Neighbors (MCDM Proximity Ranking)",
        "distance_metric": "Weighted Euclidean Distance",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_records": stats["total_rows"],
        "feature_dimensions": stats["feature_dims"],
        "numeric_features": NUMERIC_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "features_list": stats["features_list"],
        "preprocessing": {
            "numeric_scaling": "MinMaxScaler [0, 1]",
            "categorical_encoding": "One-Hot Encoding with fixed schema (18 dims)",
            "dimension_weighting": "Equal per-dimension allocation (w_j = W_g / m)",
            "budget_handling": "Hard ceiling filter with economy preference",
        },
        "purpose_profiles": list(PURPOSE_WEIGHTS.keys()),
        "default_purpose": DEFAULT_PURPOSE,
        "price_stats": {
            "min_vnd": stats["price_min"],
            "max_vnd": stats["price_max"],
            "median_vnd": stats["price_median"],
        },
        "power_stats": {
            "min_kw": stats["power_min"],
            "max_kw": stats["power_max"],
            "median_kw": stats["power_median"],
        },
        "trunk_stats": {
            "min_liters": stats["trunk_min"],
            "max_liters": stats["trunk_max"],
            "median_liters": stats["trunk_median"],
        },
    }
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"-> Đã xuất metadata mô hình: {METADATA_PATH}")
    print("=== XUẤT ARTIFACT THÀNH CÔNG ===")


if __name__ == "__main__":
    train_and_export()
