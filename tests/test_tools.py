"""Tools: sandboxed code execution (with timeout) and path-traversal-safe file I/O."""
import pytest

from app.tools import ToolError, read_file, run_code, write_file


def test_run_code_returns_output():
    assert run_code({"code": "print(2 + 3)"}).strip() == "5"


def test_run_code_times_out():
    with pytest.raises(ToolError, match="timed out"):
        run_code({"code": "import time; time.sleep(30)"})


def test_file_roundtrip_in_workspace():
    write_file({"path": "t_roundtrip.txt", "content": "hello agent"})
    assert read_file({"path": "t_roundtrip.txt"}) == "hello agent"


def test_read_missing_file_raises():
    with pytest.raises(ToolError, match="not found"):
        read_file({"path": "definitely-missing-xyz.txt"})


def test_path_traversal_is_blocked():
    # Trying to escape the workspace must be refused, not silently allowed.
    with pytest.raises(ToolError, match="escapes the workspace"):
        write_file({"path": "../../etc/evil.txt", "content": "nope"})
    with pytest.raises(ToolError, match="escapes the workspace"):
        read_file({"path": "../../../etc/passwd"})
