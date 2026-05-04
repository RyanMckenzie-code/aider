from pathlib import Path
from unittest.mock import MagicMock

import git

from aider.coders import Coder
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
