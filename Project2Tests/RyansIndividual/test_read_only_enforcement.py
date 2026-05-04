import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_read_only_file_cannot_be_edited(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("read only\n")

        io = InputOutput(yes=True)
        coder = WholeFileCoder(main_model=self.gpt35, io=io, fnames=[], read_only_fnames=["ro.txt"])

        coder.partial_response_content = "ro.txt\n```\nnew content\n```"
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "read only\n")

    def test_read_only_file_still_available_as_context(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("shared context\n")

        io = InputOutput(yes=True)
        coder = WholeFileCoder(main_model=self.gpt35, io=io, fnames=[], read_only_fnames=["ro.txt"])

        read_only_context = coder.get_read_only_files_content()
        self.assertIn("ro.txt", read_only_context)
        self.assertIn("shared context", read_only_context)

    def test_mixed_edit_batch_with_read_only_file_is_rejected(self):
        ro_file = Path("ro.txt")
        rw_file = Path("rw.txt")
        ro_file.write_text("read only\n")
        rw_file.write_text("editable\n")

        io = InputOutput(yes=True)
        coder = WholeFileCoder(
            main_model=self.gpt35,
            io=io,
            fnames=["rw.txt"],
            read_only_fnames=["ro.txt"],
        )

        coder.partial_response_content = """ro.txt
```
new ro
```
rw.txt
```
new rw
```
"""
        edited_files = coder.apply_updates()

        self.assertEqual(edited_files, set())
        self.assertEqual(ro_file.read_text(), "read only\n")
        self.assertEqual(rw_file.read_text(), "editable\n")

    def test_read_only_path_normalization_blocks_dot_slash_bypass(self):
        ro_file = Path("ro.txt")
        ro_file.write_text("read only\n")

        io = InputOutput(yes=True)
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

        io = InputOutput(yes=True)
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


if __name__ == "__main__":
    unittest.main()
