from pathlib import Path
from unittest.mock import MagicMock

import git

from aider.coders import Coder
from aider.coders.wholefile_coder import WholeFileCoder
from aider.models import Model
from aider.utils import GitTemporaryDirectory


class TestReadOnlyEditGuard:
    GPT35 = Model("gpt-3.5-turbo")

    def test_allowed_to_edit_read_only_file(self):
        with GitTemporaryDirectory():
            repo = git.Repo()

            fname = Path("readonly.txt")
            fname.touch()
            repo.git.add(str(fname))
            repo.git.commit("-m", "init")

            io = MagicMock()
            io.confirm_ask = MagicMock(return_value=True)
            io.tool_warning = MagicMock()

            coder = Coder.create(self.GPT35, None, io, read_only_fnames=["readonly.txt"])

            assert not coder.allowed_to_edit("readonly.txt")
            io.tool_warning.assert_called_with("Skipping edits to read-only file readonly.txt.")
            assert "readonly.txt" not in str(coder.abs_fnames)

    def test_allowed_to_edit_read_only_path_alias(self):
        with GitTemporaryDirectory():
            repo = git.Repo()

            fname = Path("readonly.py")
            fname.touch()
            repo.git.add(str(fname))
            repo.git.commit("-m", "init")

            io = MagicMock()
            io.confirm_ask = MagicMock(return_value=True)
            io.tool_warning = MagicMock()

            coder = Coder.create(self.GPT35, None, io, read_only_fnames=["readonly.py"])

            assert not coder.allowed_to_edit("path/to/readonly.py")
            io.tool_warning.assert_called_with("Skipping edits to read-only file path/to/readonly.py.")
            io.confirm_ask.assert_not_called()

    def test_prepare_to_edit_drops_read_only_alias_edit(self):
        with GitTemporaryDirectory():
            repo = git.Repo()

            ro = Path("readonly.py")
            ro.write_text("def protected_message():\n    return \"DO NOT EDIT ME\"\n")
            repo.git.add(str(ro))

            editable = Path("editable.py")
            editable.write_text("def ok():\n    return 1\n")
            repo.git.add(str(editable))

            repo.git.commit("-m", "init")

            io = MagicMock()
            io.confirm_ask = MagicMock(return_value=True)
            io.tool_warning = MagicMock()

            coder = Coder.create(
                self.GPT35,
                None,
                io,
                fnames=["editable.py"],
                read_only_fnames=["readonly.py"],
            )

            edits = [("path/to/readonly.py", "block", ["def protected_message():\n", "    return \"edited despite readonly\"\n"]) ]
            prepared = coder.prepare_to_edit(edits)

            assert prepared == []
            io.confirm_ask.assert_not_called()
            io.tool_warning.assert_called_with("Skipping edits to read-only file path/to/readonly.py.")


class TestWholeFileReadOnlyGuard:
    GPT35 = Model("gpt-3.5-turbo")

    def test_apply_edits_skips_read_only_file(self):
        with GitTemporaryDirectory():
            repo = git.Repo()

            ro = Path("readonly.py")
            ro.write_text("def protected_message():\n    return \"DO NOT EDIT ME\"\n")
            repo.git.add(str(ro))
            repo.git.commit("-m", "init")

            io = MagicMock()
            io.tool_warning = MagicMock()

            coder = WholeFileCoder.create(self.GPT35, None, io, read_only_fnames=["readonly.py"])
            edits = [("readonly.py", "block", ["def protected_message():\n", "    return \"edited despite readonly\"\n"])]

            coder.apply_edits(edits)

            assert ro.read_text() == "def protected_message():\n    return \"DO NOT EDIT ME\"\n"
            io.tool_warning.assert_called_with("Skipping edits to read-only file readonly.py.")

    def test_wholefile_get_edits_excludes_read_only_files(self):
        with GitTemporaryDirectory():
            repo = git.Repo()

            ro = Path("readonly.py")
            ro.write_text("def protected_message():\n    return \"DO NOT EDIT ME\"\n")
            editable = Path("editable.py")
            editable.write_text("def message():\n    return \"editable original\"\n")
            repo.git.add(str(ro))
            repo.git.add(str(editable))
            repo.git.commit("-m", "init")

            io = MagicMock()
            io.tool_warning = MagicMock()

            coder = WholeFileCoder.create(
                self.GPT35,
                None,
                io,
                fnames=["editable.py"],
                read_only_fnames=["readonly.py"],
            )

            coder.partial_response_content = (
                "readonly.py\n```python\ndef protected_message():\n    return \"edited despite readonly\"\n```\n\n"
                "editable.py\n```python\ndef message():\n    return \"edited editable\"\n```\n"
            )

            edits = coder.get_edits(mode="update")

            assert len(edits) == 1
            assert edits[0][0] == "editable.py"
