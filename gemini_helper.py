"""
Root proxy module for gemini_helper.
Re-exports llm.gemini_helper for backward compatibility.
"""
from llm.gemini_helper import (
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    get_gemini_api_key,
    get_gemini_client,
    get_model_status,
    parse_need,
    explain_results,
    _parse_fallback,
    _explain_fallback,
    _call_gemini_with_fallback,
)

__all__ = [
    "DEFAULT_MODEL",
    "FALLBACK_MODELS",
    "get_gemini_api_key",
    "get_gemini_client",
    "get_model_status",
    "parse_need",
    "explain_results",
    "_parse_fallback",
    "_explain_fallback",
    "_call_gemini_with_fallback",
]
