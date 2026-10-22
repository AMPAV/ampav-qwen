"""Thin OpenVINO GenAI wrapper for Qwen3-VL video generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any


class Qwen3VLVideo:
    """Run Qwen3-VL video-to-text generation with OpenVINO GenAI.

    OpenVINO is an explicitly selected optional runtime. The wrapper accepts
    already decoded native video tensors; callers own decoding, sampling,
    prompting, persistence, and interpretation.
    """

    def __init__(
        self,
        model_path: str | PathLike[str],
        device: str = "CPU",
        *,
        _openvino_genai: Any | None = None,
        **device_properties: Any,
    ) -> None:
        if _openvino_genai is None:
            try:
                import openvino_genai as _openvino_genai
            except ImportError as error:
                raise ImportError(
                    "Qwen3-VL OpenVINO support requires ampav-qwen[openvino]"
                ) from error

        self.model_path = str(model_path)
        self.device = device
        self._openvino_genai = _openvino_genai
        self._pipeline = _openvino_genai.VLMPipeline(
            self.model_path,
            device,
            **device_properties,
        )

    def process(
        self,
        prompt: str,
        videos: Sequence[Any],
        *,
        videos_metadata: Sequence[Any] | None = None,
        generation_config: Any | Mapping[str, Any] | None = None,
        **generation_options: Any,
    ) -> Any:
        """Generate text and return the native OpenVINO decoded result."""

        config = (
            generation_config
            if generation_config is not None
            else self._openvino_genai.GenerationConfig()
        )
        options = {
            "videos": videos,
            "generation_config": config,
            **generation_options,
        }
        if videos_metadata is not None:
            options["videos_metadata"] = videos_metadata
        return self._pipeline.generate(prompt, **options)
