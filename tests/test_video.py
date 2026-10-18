"""Tests for the native Qwen3-VL video wrapper."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from ampav.qwen3vl import Qwen3VLVideo


class Qwen3VLVideoTest(TestCase):
    """Verify that native inputs and outputs pass through unchanged."""

    @patch("ampav.qwen3vl.video.openvino_genai.VLMPipeline")
    def test_constructs_native_pipeline(self, pipeline_type: Mock) -> None:
        tool = Qwen3VLVideo(
            Path("model"),
            "GPU",
            CACHE_DIR="compiled-models",
        )

        pipeline_type.assert_called_once_with(
            "model",
            "GPU",
            CACHE_DIR="compiled-models",
        )
        self.assertEqual(tool.model_path, "model")
        self.assertEqual(tool.device, "GPU")

    @patch("ampav.qwen3vl.video.openvino_genai.VLMPipeline")
    def test_returns_native_result_without_conversion(
        self,
        pipeline_type: Mock,
    ) -> None:
        result = object()
        pipeline_type.return_value.generate.return_value = result
        tool = Qwen3VLVideo("model")
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
        pipeline_type.return_value.generate.assert_called_once_with(
            "Describe the video.",
            videos=videos,
            videos_metadata=metadata,
            generation_config=config,
            streamer="native-streamer",
        )

    @patch("ampav.qwen3vl.video.openvino_genai.GenerationConfig")
    @patch("ampav.qwen3vl.video.openvino_genai.VLMPipeline")
    def test_creates_default_config_and_omits_absent_metadata(
        self,
        pipeline_type: Mock,
        config_type: Mock,
    ) -> None:
        config = config_type.return_value
        tool = Qwen3VLVideo("model")
        videos = [object()]

        tool.process("Describe the video.", videos)

        pipeline_type.return_value.generate.assert_called_once_with(
            "Describe the video.",
            videos=videos,
            generation_config=config,
        )

    @patch("ampav.qwen3vl.video.openvino_genai.VLMPipeline")
    def test_preserves_empty_native_config_mapping(
        self,
        pipeline_type: Mock,
    ) -> None:
        tool = Qwen3VLVideo("model")
        videos = [object()]
        config: dict[str, object] = {}

        tool.process("Describe the video.", videos, generation_config=config)

        pipeline_type.return_value.generate.assert_called_once_with(
            "Describe the video.",
            videos=videos,
            generation_config=config,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
