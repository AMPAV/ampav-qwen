"""Tests for the native Qwen3-VL video wrapper."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from ampav.qwen3vl import Qwen3VLVideo


class Qwen3VLVideoTest(TestCase):
    """Verify that native inputs and outputs pass through unchanged."""

    def test_constructs_native_pipeline(self) -> None:
        native = Mock()
        tool = Qwen3VLVideo(
            Path("model"),
            "GPU",
            _openvino_genai=native,
            CACHE_DIR="compiled-models",
        )

        native.VLMPipeline.assert_called_once_with(
            "model",
            "GPU",
            CACHE_DIR="compiled-models",
        )
        self.assertEqual(tool.model_path, "model")
        self.assertEqual(tool.device, "GPU")

    def test_returns_native_result_without_conversion(self) -> None:
        native = Mock()
        result = object()
        native.VLMPipeline.return_value.generate.return_value = result
        tool = Qwen3VLVideo("model", _openvino_genai=native)
        videos = [object()]
        metadata = [object()]
        config = object()

        actual = tool.process(
            "Describe the video.",
            videos,
            videos_metadata=metadata,
            generation_config=config,
            streamer="native-streamer",
        )

        self.assertIs(actual, result)
        native.VLMPipeline.return_value.generate.assert_called_once_with(
            "Describe the video.",
            videos=videos,
            videos_metadata=metadata,
            generation_config=config,
            streamer="native-streamer",
        )

    def test_creates_default_config_and_omits_absent_metadata(self) -> None:
        native = Mock()
        config = native.GenerationConfig.return_value
        tool = Qwen3VLVideo("model", _openvino_genai=native)
        videos = [object()]

        tool.process("Describe the video.", videos)

        native.VLMPipeline.return_value.generate.assert_called_once_with(
            "Describe the video.",
            videos=videos,
            generation_config=config,
        )

    def test_preserves_empty_native_config_mapping(self) -> None:
        native = Mock()
        tool = Qwen3VLVideo("model", _openvino_genai=native)
        videos = [object()]
        config: dict[str, object] = {}

        tool.process("Describe the video.", videos, generation_config=config)

        native.VLMPipeline.return_value.generate.assert_called_once_with(
            "Describe the video.",
            videos=videos,
            generation_config=config,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
