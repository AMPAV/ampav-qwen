"""Model-free tests for the Qwen3-Omni experiment helpers."""

import unittest
from unittest.mock import patch

from experiments.qwen3_omni_audiovisual_metadata import (
    DEFAULT_PROMPT,
    MODES,
    parse_args,
    parse_json_response,
)


class OmniExperimentTest(unittest.TestCase):
    """Verify the controlled modality and bounded CLI contract."""

    def test_prompt_requires_modality_specific_evidence(self) -> None:
        normalized = " ".join(DEFAULT_PROMPT.split())

        self.assertIn('source "visual", "audio", or', normalized)
        self.assertIn("no audio was supplied", normalized)
        self.assertEqual(MODES, {"video-only": False, "audiovisual": True})

    def test_parses_plain_and_fenced_objects(self) -> None:
        expected = {"summary": "race"}
        self.assertEqual(parse_json_response('{"summary": "race"}')[0], expected)
        self.assertEqual(
            parse_json_response('```json\n{"summary": "race"}\n```')[0],
            expected,
        )

    def test_cli_defaults_to_bounded_controlled_pair(self) -> None:
        with patch("sys.argv", ["probe", "input.mp4", "run"]):
            args = parse_args()

        self.assertEqual(args.modes, ["video-only", "audiovisual"])
        self.assertEqual(args.fps, 1.0)
        self.assertEqual(args.total_pixels, 4_194_304)
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.attention_implementation, "auto")


if __name__ == "__main__":
    unittest.main()
