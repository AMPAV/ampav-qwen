"""Qwen3-Omni native runtime wrappers."""

from ._version import __version__
from .transformers import DEFAULT_MODEL_ID, Qwen3OmniTransformers
from .vllm import DEFAULT_BASE_URL, DEFAULT_VLLM_MODEL_ID, Qwen3OmniVllm


__all__ = [
    "__version__",
    "DEFAULT_MODEL_ID",
    "DEFAULT_BASE_URL",
    "DEFAULT_VLLM_MODEL_ID",
    "Qwen3OmniTransformers",
    "Qwen3OmniVllm",
]
