"""Thin OpenVINO GenAI wrapper for Qwen3-VL video generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from openvino import Tensor
import openvino_genai


class Qwen3VLVideo:
    """Run native Qwen3-VL video-to-text generation with OpenVINO GenAI.

    Args:
        model_path: Local directory containing an OpenVINO-exported Qwen3-VL
            model and tokenizer files.
        device: OpenVINO device name such as ``"CPU"`` or ``"GPU"``.
        device_properties: Native OpenVINO properties forwarded to
            :class:`openvino_genai.VLMPipeline`.

    The wrapper accepts already decoded native video tensors. File decoding,
    frame sampling, prompting, and persistence remain client responsibilities.
    """

    def __init__(
        self,
        model_path: str | PathLike[str],
        device: str = "CPU",
        **device_properties: Any,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self._pipeline = openvino_genai.VLMPipeline(
            self.model_path,
            device,
            **device_properties,
        )

    def process(
        self,
        prompt: str,
        videos: Sequence[Tensor],
        *,
        videos_metadata: Sequence[openvino_genai.VideoMetadata] | None = None,
        generation_config: openvino_genai.GenerationConfig
        | Mapping[str, Any]
        | None = None,
        **generation_options: Any,
    ) -> openvino_genai.VLMDecodedResults:
        """Generate text and return the native OpenVINO decoded result.

        Args:
            prompt: Text prompt supplied to the native VLM pipeline.
            videos: One or more native OpenVINO video tensors.
            videos_metadata: Optional native sampling metadata corresponding to
                ``videos``.
            generation_config: Native generation configuration or mapping. A
                default ``GenerationConfig`` is created when omitted.
            generation_options: Additional native ``generate`` keyword
                arguments, including streamers or supported model options.
        """

        config = (
            generation_config
            if generation_config is not None
            else openvino_genai.GenerationConfig()
        )
        if videos_metadata is None:
            return self._pipeline.generate(
                prompt,
                videos=videos,
                generation_config=config,
                **generation_options,
            )
        return self._pipeline.generate(
            prompt,
            videos=videos,
            videos_metadata=videos_metadata,
            generation_config=config,
            **generation_options,
        )
