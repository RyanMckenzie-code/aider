def create_bad_file(path):
    with open(path, "wb") as f:
        f.write(b"print('hello')\n")
        f.write(b"\xed\xb2\xb0\n")  # invalid UTF-8


def test_io_handles_bad_utf8(tmp_path):
    from aider.io import InputOutput

    bad_file = tmp_path / "bad.py"
    create_bad_file(bad_file)

    io = InputOutput()
    io.encoding = "utf-8"

    # This is the real regression target
    content = io.read_text(str(bad_file))

    assert content is not None
    assert isinstance(content, str)