"""Model-free tests for the Qwen3-Omni vLLM client."""

from pathlib import Path
import tempfile
import unittest

from ampav.qwen3omni import Qwen3OmniVllm


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **options):
        self.calls.append(options)
        return "native response"


class FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


class VllmToolsTest(unittest.TestCase):
    """Verify the shared regular-vLLM and vLLM-Omni request boundary."""

    def test_text_request_uses_text_only_output(self) -> None:
        client = FakeClient()
        tool = Qwen3OmniVllm(client=client)

        result = tool.process("Describe it", max_tokens=20)

        self.assertEqual(result, "native response")
        call = client.chat.completions.calls[0]
        self.assertEqual(call["modalities"], ["text"])
        self.assertEqual(call["max_tokens"], 20)
        self.assertEqual(call["extra_body"], {})
        self.assertEqual(
            call["messages"][1]["content"],
            [{"type": "text", "text": "Describe it"}],
        )

    def test_local_video_uses_documented_data_url_shape(self) -> None:
        client = FakeClient()
        tool = Qwen3OmniVllm(client=client)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "fixture.mp4"
            video.write_bytes(b"video")

            tool.process("Describe it", video=video, use_audio_in_video=True)

        call = client.chat.completions.calls[0]
        content = call["messages"][1]["content"]
        self.assertTrue(
            content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        )
        self.assertEqual(content[1], {"type": "text", "text": "Describe it"})
        self.assertEqual(
            call["extra_body"],
            {"mm_processor_kwargs": {"use_audio_in_video": True}},
        )

    def test_audio_in_video_requires_video(self) -> None:
        tool = Qwen3OmniVllm(client=FakeClient())

        with self.assertRaises(ValueError):
            tool.process("Describe it", use_audio_in_video=True)


if __name__ == "__main__":
    unittest.main()
