"""Qwen3-Omni native Transformers wrapper."""

from ._version import __version__
from .transformers import DEFAULT_MODEL_ID, Qwen3OmniTransformers


__all__ = ["__version__", "DEFAULT_MODEL_ID", "Qwen3OmniTransformers"]
