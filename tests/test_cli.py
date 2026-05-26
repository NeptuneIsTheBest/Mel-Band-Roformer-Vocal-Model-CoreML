from __future__ import annotations

import unittest

from melband_roformer_coreml.cli import build_parser


class CliTest(unittest.TestCase):
    def test_help_parser_has_commands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        for command in ("prepare", "convert", "verify", "infer"):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
