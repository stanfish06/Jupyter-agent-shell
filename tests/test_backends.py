from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mash.backends import (
    BACKENDS,
    JuliaBackend,
    MatlabBackend,
    PythonBackend,
    RBackend,
    default_language,
    get_backend,
)


def fake_venv(root: Path, ipykernel: bool) -> Path:
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (venv / "pyvenv.cfg").write_text("home = /usr\n")
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    if ipykernel:
        (site / "ipykernel").mkdir()
    return venv


def test_aliases() -> None:
    assert get_backend("py") is BACKENDS["python"]
    assert get_backend(" Julia ") is get_backend("jl") is BACKENDS["julia"]
    assert get_backend("R").name == "r"
    with pytest.raises(ValueError, match="choose from matlab, python, r, julia"):
        get_backend("cobol")


def test_default_language_is_python(monkeypatch: pytest.MonkeyPatch) -> None:
    assert default_language() == "python"
    monkeypatch.setenv("MASH_DEFAULT_KERNEL", "MATLAB")
    assert default_language() == "matlab"


def test_python_prefers_workspace_venv_with_ipykernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MASH_PYTHON")
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/mash/.venv")
    venv = fake_venv(tmp_path, ipykernel=True)
    py = PythonBackend()
    assert py.status(tmp_path) == (True, "workspace .venv")
    argv, env = py.launch(tmp_path, "x")
    assert argv == [str(venv / "bin" / "python"), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    assert env["VIRTUAL_ENV"] == str(venv) and env["PATH"].startswith(str(venv / "bin"))


def test_python_layers_ipykernel_with_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASH_PYTHON")
    monkeypatch.setenv("UV", "/opt/uv")
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/mash/.venv")
    fake_venv(tmp_path, ipykernel=False)
    py = PythonBackend()
    assert py.status(tmp_path) == (True, "workspace .venv (ipykernel layered in by uv)")
    argv, env = py.launch(tmp_path, "x")
    assert argv[:5] == ["/opt/uv", "run", "--no-sync", "--with", "ipykernel"]
    assert "VIRTUAL_ENV" not in env


def test_python_missing_without_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASH_PYTHON")
    ok, detail = PythonBackend().status(tmp_path)
    assert not ok and "no .venv" in detail


def test_matlab_startup_script_quotes_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mash.backends.matlab_root", lambda: Path("/Applications/MATLAB_R2099a.app"))
    _, env = MatlabBackend().launch(Path("/tmp/it's here"), "mash_1_s1_matlab")
    assert env["MWI_MATLAB_STARTUP_SCRIPT"] == (
        "cd('/tmp/it''s here'); matlab.engine.shareEngine('mash_1_s1_matlab');"
    )
    assert MatlabBackend().snapshot_code(Path("/tmp/a'b/matlab")) == "save('/tmp/a''b/matlab.mat')"


def test_r_and_julia_report_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mash.backends.shutil.which", lambda name: None)
    for env in ("MASH_R", "MASH_JULIA"):
        monkeypatch.delenv(env, raising=False)
    assert RBackend().status(tmp_path) == (False, "R not found; mash can install it with rig")
    assert JuliaBackend().status(tmp_path) == (False, "julia not found; mash can install it with juliaup")
    assert "rig 0.11.0" in RBackend().install_summary()
