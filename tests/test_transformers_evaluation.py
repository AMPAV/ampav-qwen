"""Tests for Qwen3-VL run diagnostics and pair comparison."""

import unittest

from experiments.evaluate_qwen3vl_metadata_runs import analyze_run, compare_runs


def manifest(*, transcript: bool = False) -> dict:
    """Build a minimal retained manifest for diagnostic tests."""

    return {
        "prompt_task": "activities",
        "transcript_supplied": transcript,
        "elapsed_seconds": 12.0,
        "model_load_seconds": 1.0,
        "decode_seconds": 0.5,
        "preprocess_seconds": 0.5,
        "generation_seconds": 10.0,
        "generation": {
            "input_tokens": 100,
            "output_tokens": 20,
            "visual_tokens": 50,
        },
        "video": {
            "sampling_request": {
                "start_seconds": 560.0,
                "end_seconds": 650.0,
                "sample_fps": 1.0,
                "num_frames": None,
            },
            "processor_metadata": {
                "duration": 1000.0,
                "frames_indices": [560, 600, 650],
            },
            "model_timestamp_tokens_seconds": [560.0, 600.0, 650.0],
        },
        "resource": {
            "cuda_peak_allocated_bytes": 1000,
            "cuda_peak_reserved_bytes": 1200,
        },
    }


class TransformersEvaluationTest(unittest.TestCase):
    """Verify diagnostic semantics without a live model or retained runs."""

    def test_flags_missing_source_and_segment_relative_timestamp(self) -> None:
        native = {
            "response_text": "{}",
            "parse_error": None,
            "structured": {
                "activities_and_events": [
                    {
                        "description": "Race starts",
                        "start_seconds": 0,
                        "end_seconds": 600,
                    }
                ]
            },
        }

        result = analyze_run("relative", manifest(), native)

        self.assertEqual(result["missing_source_count"], 1)
        self.assertEqual(result["timestamps_outside_requested_segment_count"], 1)
        self.assertEqual(result["array_item_counts"]["activities_and_events"], 1)

    def test_flags_transcript_denial_and_absent_attribution(self) -> None:
        native = {
            "response_text": "No audio transcript was provided.",
            "parse_error": None,
            "structured": {
                "evidence": [
                    {
                        "description": "Cyclists race",
                        "timestamp_seconds": 600,
                        "source": "visual",
                    }
                ]
            },
        }

        result = analyze_run("transcript", manifest(transcript=True), native)

        self.assertTrue(result["transcript_denial_detected"])
        self.assertEqual(result["transcript_attribution_count"], 0)

    def test_pair_comparison_reports_transparent_deltas(self) -> None:
        native = {
            "response_text": "{}",
            "parse_error": None,
            "structured": {"activities_and_events": []},
        }
        baseline = analyze_run("video", manifest(), native)
        changed_manifest = manifest(transcript=True)
        changed_manifest["generation"]["input_tokens"] = 125
        changed_manifest["generation_seconds"] = 14.0
        comparison = analyze_run("transcript", changed_manifest, native)

        pair = compare_runs(baseline, comparison)

        self.assertEqual(pair["delta"]["input_tokens"], 25)
        self.assertEqual(pair["delta"]["generation_seconds"], 4.0)


if __name__ == "__main__":
    unittest.main()
