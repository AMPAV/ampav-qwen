"""Model-free tests for Qwen native Transformers wrappers."""

from pathlib import Path
import unittest

from ampav.qwen3omni import Qwen3OmniTransformers
from ampav.qwen3vl import Qwen3VLTransformersVideo


class FakeBatch(dict):
    """Minimal native batch supporting device and dtype moves."""

    def __init__(self) -> None:
        super().__init__(input_ids=[[1, 2]])
        self.moves: list[object] = []

    def to(self, target: object) -> "FakeBatch":
        self.moves.append(target)
        return self


class FakeProcessor:
    """Record native processor calls and decode generated IDs."""

    def __init__(self) -> None:
        self.template_calls: list[tuple[object, dict[str, object]]] = []
        self.processor_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **options):
        self.template_calls.append((messages, options))
        return "templated" if options.get("tokenize") is False else FakeBatch()

    def __call__(self, **options):
        self.processor_calls.append(options)
        return FakeBatch()

    def batch_decode(self, token_ids, **options):
        self.decoded = (token_ids, options)
        return ["decoded text"]


class FakeModel:
    """Record generation options without importing Torch or Transformers."""

    device = "cuda:0"
    dtype = None

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **options):
        self.calls.append(options)
        return [[1, 2, 9, 10]]


class TransformersToolsTest(unittest.TestCase):
    """Verify explicit modality controls and native option forwarding."""

    def test_vl_process_uses_native_processor_and_generation(self) -> None:
        model = FakeModel()
        processor = FakeProcessor()
        tool = Qwen3VLTransformersVideo(model=model, processor=processor)

        text = tool.process(
            Path("fixture.mp4"),
            "Describe it",
            generation_options={"max_new_tokens": 20},
        )

        self.assertEqual(text, "decoded text")
        self.assertEqual(model.calls[0]["do_sample"], False)
        self.assertEqual(model.calls[0]["max_new_tokens"], 20)
        self.assertTrue(
            processor.template_calls[0][0][0]["content"][0]["video"].endswith(
                "/fixture.mp4"
            )
        )

    def test_omni_requires_all_native_dependencies_together(self) -> None:
        with self.assertRaises(ValueError):
            Qwen3OmniTransformers(model=FakeModel())

    def test_omni_process_forwards_explicit_audio_control(self) -> None:
        model = FakeModel()
        processor = FakeProcessor()
        mm_calls: list[tuple[object, bool]] = []

        def process_mm_info(messages, *, use_audio_in_video):
            mm_calls.append((messages, use_audio_in_video))
            return "audio", None, "video"

        tool = Qwen3OmniTransformers(
            model=model,
            processor=processor,
            process_mm_info=process_mm_info,
        )

        text = tool.process(
            "fixture.mp4",
            "Describe it",
            use_audio_in_video=True,
            video_options={"fps": 1.0},
            generation_options={"max_new_tokens": 20},
        )

        self.assertEqual(text, "decoded text")
        self.assertTrue(mm_calls[0][1])
        self.assertTrue(processor.processor_calls[0]["use_audio_in_video"])
        self.assertTrue(model.calls[0]["use_audio_in_video"])
        self.assertEqual(model.calls[0]["do_sample"], False)
        self.assertEqual(model.calls[0]["max_new_tokens"], 20)
        messages = processor.template_calls[0][0]
        self.assertEqual(messages[0]["content"][0]["fps"], 1.0)


if __name__ == "__main__":
    unittest.main()
