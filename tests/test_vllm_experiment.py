"""Model-free tests for streamed vLLM experiment evidence."""

from types import SimpleNamespace
import unittest

from experiments.qwen3_omni_vllm_metadata import (
    collect_stream,
    parse_gpu_rows,
    parse_json_response,
)


class DumpableChunk(SimpleNamespace):
    def model_dump(self, *, mode):
        return {"mode": mode, "choices": len(self.choices)}


class VllmExperimentTest(unittest.TestCase):
    """Verify validity, TTFT, token, and GPU parsing helpers."""

    def test_collects_stream_and_first_token_time(self) -> None:
        chunks = [
            DumpableChunk(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
                usage=None,
            ),
            DumpableChunk(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content='{"status":'))
                ],
                usage=None,
            ),
            DumpableChunk(
                choices=[SimpleNamespace(delta=SimpleNamespace(content='"ok"}'))],
                usage=SimpleNamespace(
                    model_dump=lambda mode: {"completion_tokens": 4}
                ),
            ),
        ]
        times = iter((12.5,))

        result = collect_stream(chunks, started_at=10.0, clock=lambda: next(times))

        self.assertEqual(result["response_text"], '{"status":"ok"}')
        self.assertEqual(result["time_to_first_token_seconds"], 2.5)
        self.assertEqual(result["usage"]["completion_tokens"], 4)
        self.assertEqual(len(result["native_chunks"]), 3)

    def test_parses_structured_output_and_gpu_rows(self) -> None:
        self.assertEqual(
            parse_json_response('```json\n{"status":"ok"}\n```')[0],
            {"status": "ok"},
        )
        rows = parse_gpu_rows("0, NVIDIA L4, 1200, 23034, 97\n")
        self.assertEqual(rows[0]["memory_used_mib"], 1200)
        self.assertEqual(rows[0]["utilization_percent"], 97)


if __name__ == "__main__":
    unittest.main()
