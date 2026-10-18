"""Model-free tests for the Qwen3-VL experiment helpers."""

import unittest

from experiments.qwen3vl_video_metadata import (
    metric_value,
    scaled_dimensions,
    select_frame_indices,
)


class ExperimentHelperTest(unittest.TestCase):
    """Verify deterministic sampling and native metric serialization."""

    def test_selects_requested_frame_rate(self) -> None:
        self.assertEqual(
            select_frame_indices(250, 25.0, 1.0, None),
            list(range(0, 250, 25)),
        )

    def test_caps_frames_while_covering_full_candidate_range(self) -> None:
        indices = select_frame_indices(250, 25.0, 1.0, 3)

        self.assertEqual(len(indices), 3)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 225)

    def test_serializes_mean_std_pair_without_renaming(self) -> None:
        pair = type("Pair", (), {"mean": 3.5, "std": 0.25})()

        self.assertEqual(metric_value(pair), {"mean": 3.5, "std": 0.25})

    def test_scales_dimensions_to_pixel_budget_without_upscaling(self) -> None:
        width, height = scaled_dimensions(1548, 972, 262144)

        self.assertLessEqual(width * height, 262144)
        self.assertEqual(width % 2, 0)
        self.assertEqual(height % 2, 0)
        self.assertEqual(scaled_dimensions(320, 180, 262144), (320, 180))


if __name__ == "__main__":
    unittest.main()
