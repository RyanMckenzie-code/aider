import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aider.args import get_parser
from aider.coders import Coder
from aider.coders.wholefile_coder import WholeFileCoder
from aider.commands import Commands
from aider.io import InputOutput
from aider.main import main
from aider.models import Model


class TestSupervisedMode(unittest.TestCase):
    def setUp(self):
        self.original_cwd = os.getcwd()
        self.original_env = os.environ.copy()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)
        os.environ["OPENAI_API_KEY"] = "deadbeef"
        os.environ["AIDER_CHECK_UPDATE"] = "false"
        os.environ["AIDER_ANALYTICS"] = "false"
        os.environ["HOME"] = self.tempdir
        self.gpt35 = Model("gpt-3.5-turbo")

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.tempdir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_supervised_mode_parser_flags_and_aliases(self):
        parser = get_parser([], None)

        self.assertTrue(parser.parse_args(["--supervised-mode"]).supervised_mode)
        self.assertFalse(parser.parse_args(["--no-supervised-mode"]).supervised_mode)
        self.assertTrue(parser.parse_args(["--supervise"]).supervised_mode)
        self.assertFalse(parser.parse_args(["--no-supervise"]).supervised_mode)

    def test_supervised_mode_is_passed_to_coder_from_cli(self):
        with patch("aider.coders.Coder.create") as mock_coder:
            main(
                ["--no-git", "--yes", "--supervised-mode"],
                input=DummyInput(),
                output=DummyOutput(),
            )

        mock_coder.assert_called_once()
        _, kwargs = mock_coder.call_args
        self.assertIs(kwargs["supervised_mode"], True)

    def test_supervised_mode_is_false_by_default(self):
        with patch("aider.coders.Coder.create") as mock_coder:
            main(["--no-git", "--yes"], input=DummyInput(), output=DummyOutput())

        mock_coder.assert_called_once()
        _, kwargs = mock_coder.call_args
        self.assertIs(kwargs["supervised_mode"], False)

    def test_supervised_mode_can_be_loaded_from_env_file(self):
        Path(".env").write_text("AIDER_SUPERVISED_MODE=on\n")

        with patch("aider.coders.Coder.create") as mock_coder:
            main(["--no-git", "--yes"], input=DummyInput(), output=DummyOutput())

        mock_coder.assert_called_once()
        _, kwargs = mock_coder.call_args
        self.assertIs(kwargs["supervised_mode"], True)

    def test_supervise_command_toggles_and_accepts_explicit_values(self):
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        coder = Coder.create(self.gpt35, None, io)
        commands = Commands(io, coder)

        self.assertFalse(coder.supervised_mode)

        commands.cmd_supervise("")
        self.assertTrue(coder.supervised_mode)

        commands.cmd_supervise("off")
        self.assertFalse(coder.supervised_mode)

        commands.cmd_supervise("on")
        self.assertTrue(coder.supervised_mode)

        commands.cmd_supervise("0")
        self.assertFalse(coder.supervised_mode)

        commands.cmd_supervise("yes")
        self.assertTrue(coder.supervised_mode)

    def test_supervise_command_reports_bad_arguments(self):
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        coder = Coder.create(self.gpt35, None, io)
        commands = Commands(io, coder)

        with patch.object(io, "tool_error") as mock_tool_error:
            commands.cmd_supervise("maybe")

        mock_tool_error.assert_called_once_with("Usage: /supervise [on|off]")

    def test_supervised_mode_rejects_edit_without_changing_file(self):
        fname = Path("sample.txt")
        fname.write_text("old\n")
        io = InputOutput(pretty=False, fancy_input=False, yes=None)
        io.confirm_ask = MagicMock(return_value=False)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=[str(fname)],
            supervised_mode=True,
        )

        coder.partial_response_content = "sample.txt\n```\nnew\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(fname.read_text(), "old\n")
        io.confirm_ask.assert_called_once_with("Apply this change?", subject="sample.txt")

    def test_supervised_mode_applies_only_approved_edits(self):
        first = Path("first.txt")
        second = Path("second.txt")
        first.write_text("first old\n")
        second.write_text("second old\n")
        io = InputOutput(pretty=False, fancy_input=False, yes=None)
        io.confirm_ask = MagicMock(side_effect=[True, False])
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=[str(first), str(second)],
            supervised_mode=True,
        )

        coder.partial_response_content = (
            "first.txt\n```\nfirst new\n```\n\nsecond.txt\n```\nsecond new\n```"
        )
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, {"first.txt"})
        self.assertEqual(first.read_text(), "first new\n")
        self.assertEqual(second.read_text(), "second old\n")
        self.assertEqual(io.confirm_ask.call_count, 2)

    def test_supervised_mode_off_applies_without_prompting(self):
        fname = Path("sample.txt")
        fname.write_text("old\n")
        io = InputOutput(pretty=False, fancy_input=False, yes=None)
        io.confirm_ask = MagicMock(return_value=False)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=[str(fname)],
            supervised_mode=False,
        )

        coder.partial_response_content = "sample.txt\n```\nnew\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, {"sample.txt"})
        self.assertEqual(fname.read_text(), "new\n")
        io.confirm_ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
