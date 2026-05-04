import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aider.coders.wholefile_coder import WholeFileCoder
from aider.io import InputOutput
from aider.models import Model


class TestReadOnlyEnforcement(unittest.TestCase):
    def setUp(self):
        self.original_cwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)
        self.gpt35 = Model("gpt-3.5-turbo")

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_read_only_file_cannot_be_edited_when_user_answers_no(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("read only\n")

        io = InputOutput(yes=False)
        coder = WholeFileCoder(main_model=self.gpt35, io=io, fnames=[], read_only_fnames=["ro.txt"])

        coder.partial_response_content = "ro.txt\n```\nnew content\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "read only\n")
        self.assertIn(coder.abs_root_path("ro.txt"), coder.abs_read_only_fnames)
        self.assertNotIn(coder.abs_root_path("ro.txt"), coder.abs_fnames)

    def test_read_only_file_still_available_as_context(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("shared context\n")

        io = InputOutput(yes=False)
        coder = WholeFileCoder(main_model=self.gpt35, io=io, fnames=[], read_only_fnames=["ro.txt"])

        read_only_context = coder.get_read_only_files_content()
        self.assertIn("ro.txt", read_only_context)
        self.assertIn("shared context", read_only_context)

    def test_read_only_file_stays_blocked_during_apply_updates(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("read only\n")
        io = InputOutput(yes=None)
        coder = WholeFileCoder(main_model=self.gpt35, io=io, fnames=[], read_only_fnames=["ro.txt"])

        coder.partial_response_content = "ro.txt\n```\nnew content\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "read only\n")
        self.assertIn(coder.abs_root_path("ro.txt"), coder.abs_read_only_fnames)
        self.assertNotIn(coder.abs_root_path("ro.txt"), coder.abs_fnames)

    def test_read_only_path_normalization_blocks_dot_slash_bypass(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("read only\n")

        io = InputOutput(yes=False)
        coder = WholeFileCoder(main_model=self.gpt35, io=io, fnames=[], read_only_fnames=["./ro.txt"])

        coder.partial_response_content = "ro.txt\n```\nnew content\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "read only\n")

    def test_notifies_when_prompt_targets_read_only_but_block_has_no_filename(self):
        ro_file = Path("ro.txt")
        rw_file = Path("rw.txt")
        ro_file.write_text("read only\n")
        rw_file.write_text("editable\n")

        io = InputOutput(yes=False)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=["rw.txt"],
            read_only_fnames=["ro.txt"],
        )

        coder.partial_response_content = "Please update `ro.txt`\n```\nnew content\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "read only\n")
        self.assertEqual(rw_file.read_text(), "editable\n")
        self.assertIn("ro.txt was added with /read and cannot be edited.", coder.reflected_message)

    def test_read_only_edit_never_uses_generic_not_in_chat_prompt(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("read only\n")

        io = InputOutput(yes=None)
        io.confirm_ask = MagicMock(return_value=False)
        coder = WholeFileCoder(main_model=self.gpt35, io=io, fnames=[], read_only_fnames=["ro.txt"])

        coder.partial_response_content = "ro.txt\n```\nnew content\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "read only\n")
        self.assertEqual(io.confirm_ask.call_count, 0)

    def test_user_request_targeting_read_only_is_blocked_without_permission(self):
        ro_file = Path("readonly.py")
        rw_file = Path("editable.py")
        ro_file.write_text("def protected_message():\n    return 'keep'\n")
        rw_file.write_text("def editable_message():\n    return 'editable'\n")

        io = InputOutput(yes=False)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=["editable.py"],
            read_only_fnames=["readonly.py"],
        )

        msg = "Change readonly.py so protected_message() returns 'DO NOT EDIT ME'."
        processed = coder.preproc_user_input(msg)

        self.assertIsNone(processed)
        self.assertEqual(ro_file.read_text(), "def protected_message():\n    return 'keep'\n")
        self.assertEqual(rw_file.read_text(), "def editable_message():\n    return 'editable'\n")
        self.assertIn(coder.abs_root_path("readonly.py"), coder.abs_read_only_fnames)
        self.assertIn(coder.abs_root_path("editable.py"), coder.abs_fnames)

    def test_user_can_promote_read_only_then_request_proceeds(self):
        ro_file = Path("readonly.py")
        rw_file = Path("editable.py")
        ro_file.write_text("def protected_message():\n    return 'keep'\n")
        rw_file.write_text("def editable_message():\n    return 'editable'\n")

        io = InputOutput(yes=None)
        io.confirm_ask = MagicMock(return_value=True)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=["editable.py"],
            read_only_fnames=["readonly.py"],
        )

        msg = "Change readonly.py so protected_message() returns 'DO NOT EDIT ME'."
        processed = coder.preproc_user_input(msg)

        self.assertEqual(processed, msg)
        self.assertNotIn(coder.abs_root_path("readonly.py"), coder.abs_read_only_fnames)
        self.assertIn(coder.abs_root_path("readonly.py"), coder.abs_fnames)
        self.assertIn(coder.abs_root_path("editable.py"), coder.abs_fnames)

    def test_exact_repro_read_only_file_uses_read_only_prompt_not_generic(self):
        ro_file = Path("readonly.py")
        rw_file = Path("editable.py")
        ro_file.write_text("def protected_message():\n    return 'ORIGINAL'\n")
        rw_file.write_text("def editable_message():\n    return 'editable'\n")

        io = InputOutput(yes=None)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=["editable.py"],
            read_only_fnames=["readonly.py"],
        )

        coder.partial_response_content = "readonly.py\n```\ndef protected_message():\n    return 'edited despite readonly'\n```"
        with patch("builtins.input", return_value=""):
            edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "def protected_message():\n    return 'ORIGINAL'\n")
        self.assertEqual(rw_file.read_text(), "def editable_message():\n    return 'editable'\n")
        self.assertIn(coder.abs_root_path("readonly.py"), coder.abs_read_only_fnames)
        self.assertNotIn(coder.abs_root_path("readonly.py"), coder.abs_fnames)

    def test_blocked_read_only_edit_explains_why_output_is_blocked(self):
        ro_file = Path("readonly.py")
        ro_file.write_text("def protected_message():\n    return 'ORIGINAL'\n")

        io = InputOutput(yes=None)
        io.tool_error = MagicMock()
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=[],
            read_only_fnames=["readonly.py"],
        )

        coder.partial_response_content = "readonly.py\n```\ndef protected_message():\n    return 'blocked'\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertTrue(io.tool_error.called)
        self.assertIn("That is why this edit output is blocked.", io.tool_error.call_args_list[0][0][0])

    def test_typo_target_that_matches_read_only_does_not_show_generic_prompt(self):
        ro_file = Path("readonly.py")
        ro_file.write_text("def protected_message():\n    return 'ORIGINAL'\n")
        Path("editable.py").write_text("def editable_message():\n    return 'editable'\n")

        io = InputOutput(yes=None)
        io.confirm_ask = MagicMock(return_value=False)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=["editable.py"],
            read_only_fnames=["readonly.py"],
        )

        Path("readonyl.py").write_text("typo file\n")
        edited = coder.allowed_to_edit("readonyl.py")

        self.assertIsNone(edited)
        self.assertEqual(io.confirm_ask.call_count, 1)
        prompt_text = io.confirm_ask.call_args[0][0]
        self.assertIn("looks like readonly.py, which is read-only", prompt_text)
        self.assertNotEqual(
            prompt_text, "Allow edits to file that has not been added to the chat?"
        )


if __name__ == "__main__":
    unittest.main()
