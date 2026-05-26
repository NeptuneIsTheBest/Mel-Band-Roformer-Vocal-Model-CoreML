from __future__ import annotations

import contextlib
import io
import unittest
from argparse import ArgumentParser, _SubParsersAction

from melband_roformer_coreml.cli import build_parser


class CliTest(unittest.TestCase):
    def get_subparser(self, parser: ArgumentParser, command: str) -> ArgumentParser:
        for action in parser._actions:
            if isinstance(action, _SubParsersAction):
                return action.choices[command]
        raise AssertionError("parser has no subcommands")

    def test_help_parser_has_commands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        for command in ("prepare", "convert", "verify", "infer"):
            self.assertIn(command, help_text)

    def test_maskcore_cli_options_are_removed(self) -> None:
        parser = build_parser()
        convert_help = self.get_subparser(parser, "convert").format_help()
        verify_help = self.get_subparser(parser, "verify").format_help()
        self.assertNotIn("--force-maskcore", convert_help)
        self.assertNotIn("--skip-full", convert_help)
        self.assertNotIn("--mode", verify_help)
        self.assertNotIn("maskcore", verify_help)

    def test_convert_does_not_accept_maskcore_flags(self) -> None:
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["convert", "--force-maskcore"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["convert", "--skip-full"])

    def test_verify_does_not_accept_mode(self) -> None:
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["verify", "--mode", "maskcore"])


if __name__ == "__main__":
    unittest.main()
