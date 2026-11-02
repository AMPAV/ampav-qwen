"""Tests for the package version export."""

from importlib.metadata import version
import unittest

from ampav.qwen3omni import __version__ as omni_version
from ampav.qwen3vl import __version__ as vl_version


class PackageVersionTest(unittest.TestCase):
    """Verify the public version matches installed package metadata."""

    def test_public_version_matches_distribution(self) -> None:
        distribution_version = version("ampav-qwen")

        self.assertEqual(vl_version, distribution_version)
        self.assertEqual(omni_version, distribution_version)


if __name__ == "__main__":
    unittest.main()
