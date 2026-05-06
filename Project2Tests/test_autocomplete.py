import os
from pathlib import Path

from aider.repo import GitRepo
from aider.io import InputOutput
from aider.coders import Coder
from aider import models


def test_autocomplete_includes_untracked_excludes_ignored(tmp_path):
    # --- Setup repo ---
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    os.chdir(repo_dir)

    # init git repo
    os.system("git init")

    # tracked file
    (repo_dir / "README.md").write_text("# test", encoding="utf-8")
    os.system("git add README.md")
    os.system("git commit -m 'init'")

    # untracked files
    (repo_dir / "newfile.py").write_text("print('hi')", encoding="utf-8")
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "helper.py").write_text("x=1", encoding="utf-8")

    # ignored file
    (repo_dir / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo_dir / "hidden.log").write_text("secret", encoding="utf-8")

    # --- Create coder instance ---
    io = InputOutput(pretty=False)
    model = models.Model(models.DEFAULT_MODEL_NAME)

    coder = Coder.create(main_model=model, io=io, fnames=[])

    # attach repo
    coder.repo = GitRepo(io, [], None)

    # --- Get autocomplete candidates ---
    addable_files = coder.get_addable_relative_files()

    # --- Assertions ---
    assert "newfile.py" in addable_files
    assert "src/helper.py" in addable_files

    # ignored file should NOT appear
    assert "hidden.log" not in addable_files