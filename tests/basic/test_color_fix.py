import unittest
from unittest.mock import MagicMock
from aider.io import ensure_hash_prefix, convert_rgb_to_hex, InputOutput


class TestColorFix(unittest.TestCase):

    # --- Unit Tests for Helpers ---

    def test_convert_rgb_to_hex(self):
        # Test standard RGB
        self.assertEqual(convert_rgb_to_hex("rgb(255, 0, 0)"), "#ff0000")
        # Test RGB with different spacing
        self.assertEqual(convert_rgb_to_hex("rgb(0,255,255)"), "#00ffff")
        # Test RGBA (should still convert the RGB part)
        self.assertEqual(convert_rgb_to_hex("rgba(255, 255, 255, 1)"), "#ffffff")
        # Test non-rgb string (should remain unchanged)
        self.assertEqual(convert_rgb_to_hex("blue"), "blue")
        self.assertEqual(convert_rgb_to_hex("#bcbdbf"), "#bcbdbf")

    def test_ensure_hash_prefix_integration(self):
        # Test that it fixes bare hex
        self.assertEqual(ensure_hash_prefix("bcbdbf"), "#bcbdbf")
        # Test that it integrates with our new RGB converter
        self.assertEqual(ensure_hash_prefix("rgb(255, 0, 0)"), "#ff0000")

    # --- Integration Test for the Crash ---

    def test_get_style_no_crash_on_invalid_color(self):
        """
        Verify that _get_style catches ValueError when prompt_toolkit
        rejects a color that passed initial validation.
        """
        # 1. Setup InputOutput with a color that causes the mismatch
        io = InputOutput(pretty=True, user_input_color="invalid-on-purpose")

        # 2. We don't need to actually run the UI, we just check if the
        #    method handles the failure internally.
        try:
            style = io._get_style()
            # If we reach here, the try-except caught the error!
        except ValueError:
            self.fail("_get_style() raised ValueError unexpectedly!")

    def test_get_style_with_rgb_conversion(self):
        """
        Verify that an RGB color is converted and successfully
        parsed into a style object.
        """
        io = InputOutput(pretty=True, user_input_color="rgb(0, 255, 0)")
        style = io._get_style()

        # In prompt_toolkit, styles are converted to internal objects.
        # We just want to ensure it's not empty and didn't crash.
        self.assertIsNotNone(style)


if __name__ == '__main__':
    unittest.main()