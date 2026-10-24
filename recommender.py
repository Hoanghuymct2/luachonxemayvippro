"""
Root proxy module for recommender.
Re-exports core.recommender for backward compatibility.
"""
from core.recommender import (
    ALL_GROUPS,
    CATEGORICAL_COLS,
    DEFAULT_PURPOSE,
    NUMERIC_COLS,
    PURPOSE_DESCRIPTIONS,
    PURPOSE_WEIGHTS,
    MotoRecommender,
    CarRecommender,
    load_and_clean_data,
    _per_dimension_weights,
    _criteria_breakdown,
    _parse_power_value,
    _parse_price_value,
    _parse_trunk_value,
)

__all__ = [
    "ALL_GROUPS",
    "CATEGORICAL_COLS",
    "DEFAULT_PURPOSE",
    "NUMERIC_COLS",
    "PURPOSE_DESCRIPTIONS",
    "PURPOSE_WEIGHTS",
    "MotoRecommender",
    "CarRecommender",
    "load_and_clean_data",
    "_per_dimension_weights",
    "_criteria_breakdown",
    "_parse_power_value",
    "_parse_price_value",
    "_parse_trunk_value",
]
