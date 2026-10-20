"""Model-free tests for the Transformers Qwen3-VL experiment helpers."""

import unittest
from unittest.mock import patch
from pathlib import Path

from experiments.qwen3vl_transformers_video_metadata import (
    build_prompt,
    build_messages,
    bind_video_sampler,
    cached_model_revision,
    collect_reported_timestamps,
    model_timestamp_tokens,
    parse_args,
    parse_json_response,
    plan_keyframes,
    pop_video_metadata,
    json_safe,
    validate_sampling,
)


class TransformersExperimentTest(unittest.TestCase):
    """Verify prompt boundaries, parsing, and temporal evidence planning."""

    def test_video_only_prompt_restricts_sources(self) -> None:
        prompt = build_prompt("aboutness")

        self.assertIn('use source "visual" only', prompt)
        self.assertNotIn("<transcript>", prompt)

    def test_transcript_prompt_separates_audio_and_visual_timing(self) -> None:
        prompt = build_prompt("combined", "00:02 A starter's pistol fires.")

        self.assertIn("external ASR transcript", prompt)
        self.assertIn("did not hear the video's audio", prompt)
        self.assertIn("must not be used to validate visual timing", prompt)
        self.assertIn("<transcript>", prompt)

    def test_messages_use_plain_local_video_path(self) -> None:
        messages = build_messages(Path("fixture.mp4"), "Analyze it")

        video = messages[0]["content"][0]["video"]
        self.assertTrue(video.endswith("/fixture.mp4"))
        self.assertFalse(video.startswith("file:"))

    def test_processor_metadata_is_removed_before_generation(self) -> None:
        inputs = {"input_ids": [1, 2], "video_metadata": ["metadata"]}

        metadata = pop_video_metadata(inputs)

        self.assertEqual(metadata, "metadata")
        self.assertNotIn("video_metadata", inputs)

    def test_json_safe_normalizes_scalar_subclasses(self) -> None:
        class VersionString(str):
            pass

        normalized = json_safe(VersionString("2.14.1+cu130"))

        self.assertEqual(normalized, "2.14.1+cu130")
        self.assertIs(type(normalized), str)

    def test_parses_plain_and_fenced_json_objects(self) -> None:
        self.assertEqual(parse_json_response('{"summary": "race"}')[0], {"summary": "race"})
        self.assertEqual(
            parse_json_response('```json\n{"summary": "race"}\n```')[0],
            {"summary": "race"},
        )

    def test_rejects_non_object_json(self) -> None:
        parsed, error = parse_json_response("[]")

        self.assertIsNone(parsed)
        self.assertEqual(error, "response JSON is not an object")

    def test_collects_only_valid_timestamp_fields(self) -> None:
        value = {
            "events": [
                {"start_seconds": 4, "end_seconds": 8.5},
                {"timestamp_seconds": None},
                {"timestamp_seconds": -1},
            ],
            "unrelated": {"duration_seconds": 90},
        }

        self.assertEqual(collect_reported_timestamps(value), [4.0, 8.5])

    def test_groups_sampled_frames_by_temporal_patch(self) -> None:
        timestamps = model_timestamp_tokens([0, 10, 20, 30], 10.0, 2)

        self.assertEqual(timestamps, [0.5, 2.5])

    def test_pads_last_frame_for_temporal_patch(self) -> None:
        timestamps = model_timestamp_tokens([0, 10, 20], 10.0, 2)

        self.assertEqual(timestamps, [0.5, 2.0])

    def test_keyframe_plan_uses_unique_nearest_visible_frames(self) -> None:
        plan = plan_keyframes([0.4, 0.6, 9.0], [0.5, 10.0], max_keyframes=12)

        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["visible_timestamp_seconds"], 0.5)
        self.assertEqual(plan[1]["visible_timestamp_seconds"], 10.0)

    def test_sampling_requires_exactly_one_control(self) -> None:
        validate_sampling(0.2, None)
        validate_sampling(None, 64)
        with self.assertRaises(ValueError):
            validate_sampling(None, None)
        with self.assertRaises(ValueError):
            validate_sampling(0.2, 64)

    def test_video_sampler_binds_controls_for_generic_loader(self) -> None:
        def sample_frames(*args, **kwargs):
            return args, kwargs

        sampler = bind_video_sampler(sample_frames, None, 4)
        _, kwargs = sampler("metadata")

        self.assertEqual(kwargs, {"fps": None, "num_frames": 4})

    def test_segment_sampler_offsets_indices_to_absolute_video_time(self) -> None:
        metadata = type(
            "Metadata",
            (),
            {"fps": 10.0, "total_num_frames": 1_000, "duration": 100.0},
        )()

        def sample_frames(metadata, **kwargs):
            self.assertEqual(metadata.total_num_frames, 101)
            self.assertEqual(kwargs, {"fps": None, "num_frames": 3})
            return [0, 50, 100]

        sampler = bind_video_sampler(sample_frames, None, 3, 20.0, 30.0)

        self.assertEqual(sampler(metadata), [200, 250, 300])

    def test_cached_revision_is_read_from_snapshot_path(self) -> None:
        path = "/cache/models--Qwen/snapshots/abc123/config.json"

        self.assertEqual(cached_model_revision(path), "abc123")
        self.assertIsNone(cached_model_revision("/models/config.json"))

    def test_cli_uses_default_fps_unless_num_frames_is_requested(self) -> None:
        with patch("sys.argv", ["probe", "input.mp4", "run"]):
            default_args = parse_args()
        with patch(
            "sys.argv",
            ["probe", "input.mp4", "run", "--num-frames", "64"],
        ):
            frame_args = parse_args()

        self.assertEqual(default_args.sample_fps, 0.2)
        self.assertEqual(default_args.max_total_video_pixels, 8_388_608)
        self.assertEqual(default_args.min_total_video_pixels, 4096)
        self.assertIsNone(default_args.num_frames)
        self.assertIsNone(frame_args.sample_fps)
        self.assertEqual(frame_args.num_frames, 64)


if __name__ == "__main__":
    unittest.main()
