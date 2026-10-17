from __future__ import annotations

import asyncio
import faulthandler
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pytest
from jupyter_client.provisioning import LocalProvisioner

from mash.session import KernelDied
from mash.workspace import Workspace

from .conftest import started

pytestmark = pytest.mark.anyio

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


async def test_kernels_share_cell_numbers_not_variables(workspace: Workspace) -> None:
    s = await started(workspace, "python", "python")
    assert list(s.kernels) == ["py", "py2"]
    a = await s.execute("x = 41\nx + 1", "human", "py")
    assert a.status == "ok" and "42" in a.text()
    b = await s.execute("x", "agent", "py2")
    assert b.status == "error" and "NameError" in b.text()
    assert [(c.index, c.kernel, c.author) for c in s.cells] == [(1, "py", "human"), (2, "py2", "agent")]


async def test_kernel_selection(workspace: Workspace) -> None:
    s = await started(workspace, "python", "python")
    with pytest.raises(ValueError, match="pass kernel="):
        await s.execute("1", "agent")
    with pytest.raises(KeyError):
        s.kernel("python")
    with pytest.raises(KeyError):
        s.kernel("nope")
    with pytest.raises(ValueError, match="already has"):
        s.add_kernel("python", "py")
    one = await started(workspace, "python", name="s2")
    assert one.kernel() is one.kernel("python") is one.kernels["py"]


async def test_kernels_run_concurrently_and_one_kernel_queues(workspace: Workspace) -> None:
    s = await started(workspace, "python", "python")
    t0 = time.monotonic()
    await asyncio.gather(
        s.execute("import time; time.sleep(1.5)", "human", "py"),
        s.execute("import time; time.sleep(1.5)", "agent", "py2"),
    )
    assert time.monotonic() - t0 < 2.7
    t0 = time.monotonic()
    cells = await asyncio.gather(
        s.execute("time.sleep(1)", "human", "py"), s.execute("time.sleep(1)", "agent", "py")
    )
    assert time.monotonic() - t0 >= 1.9
    assert all(c.status == "ok" for c in cells)


async def test_dead_kernel_leaves_others_running_and_restarts(workspace: Workspace) -> None:
    s = await started(workspace, "python", "python")
    py = s.kernels["py"]
    assert py._km is not None
    assert isinstance(py._km.provisioner, LocalProvisioner) and py._km.provisioner.process
    pid = py._km.provisioner.process.pid
    running = asyncio.create_task(s.execute("import time; time.sleep(60)", "human", "py"))
    await asyncio.sleep(1)
    os.kill(pid, signal.SIGKILL)
    cell = await asyncio.wait_for(running, 15)
    assert cell.status == "error" and "KernelDied" in cell.text()
    assert py.state == "dead" and py.error
    assert (await s.execute("1 + 1", "agent", "py2")).status == "ok"
    with pytest.raises(KernelDied):
        await s.execute("1", "human", "py")
    new = await s.restart_kernel("py")
    await asyncio.wait_for(new.ready.wait(), 60)
    assert new.state == "idle" and s.kernels["py"] is new
    again = await s.execute("2 + 2", "human", "py")
    assert again.status == "ok" and again.index == 3


async def test_one_notebook_and_script_per_kernel(workspace: Workspace) -> None:
    s = await started(workspace, "python", "python")
    await s.execute("a = 1", "human", "py")
    await s.execute("b = 2", "agent", "py2")
    await s.execute("print(a)", "agent", "py")
    nb = nbformat.read(s.record_dir / "py.ipynb", as_version=4)
    assert nb.metadata.kernelspec.name == "python3"
    assert nb.metadata.language_info.name == "python"
    assert [c.execution_count for c in nb.cells] == [1, 2]
    assert [c.metadata.mash.index for c in nb.cells] == [1, 3]
    assert nb.cells[1].outputs[0].text == "1\n"
    script = (s.record_dir / "py.py").read_text()
    assert script.startswith("# %% In[1] you") and "# %% In[2] agent" in script
    nb2 = nbformat.read(s.record_dir / "py2.ipynb", as_version=4)
    assert [c.execution_count for c in nb2.cells] == [1]


async def test_stderr_is_not_a_failure_and_errors_keep_traceback(workspace: Workspace) -> None:
    s = await started(workspace, "python")
    warn = await s.execute(
        "import sys, warnings\nprint('note', file=sys.stderr)\nwarnings.warn('careful')", "human"
    )
    assert warn.status == "ok"
    err = await s.execute("def f():\n    return 1 / 0\nf()", "agent")
    text = err.text()
    assert err.status == "error"
    assert "ZeroDivisionError" in text and "return 1 / 0" in text and "\x1b[" not in text


async def test_variables_and_interrupt(workspace: Workspace) -> None:
    s = await started(workspace, "python")
    await s.execute("values = [1, 2, 3]", "human")
    listing = await s.kernel().variables()
    assert "values" in listing and "list" in listing
    assert [c.code for c in s.cells] == ["values = [1, 2, 3]"]
    running = asyncio.create_task(s.execute("import time; time.sleep(30)", "agent"))
    await asyncio.sleep(1)
    await s.kernel().interrupt()
    cell = await asyncio.wait_for(running, 10)
    assert cell.status == "error" and "KeyboardInterrupt" in cell.text()
    assert s.kernel().state == "idle"


async def test_images_arrive_as_data(workspace: Workspace) -> None:
    s = await started(workspace, "python")
    code = (
        "import base64\nfrom IPython.display import Image, display\n"
        f"display(Image(data=base64.b64decode('{PNG}')))"
    )
    cell = await s.execute(code, "agent")
    assert cell.status == "ok"
    assert any("image/png" in o.data for o in cell.outputs if o.kind == "data")


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not on PATH")
async def test_python_kernel_in_uv_project_without_ipykernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MASH_PYTHON")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text(
        '[project]\nname = "proj"\nversion = "0"\nrequires-python = ">=3.12"\ndependencies = []\n'
    )
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(proj / ".venv")], check=True, capture_output=True
    )
    before = sorted(p.name for p in (proj / ".venv").rglob("*"))
    ws = Workspace("proj", proj)
    try:
        s = await started(ws, "python", timeout=120)
        cell = await s.execute("import sys, ipykernel; print(sys.prefix)", "agent")
        assert cell.status == "ok", cell.text()
    finally:
        await ws.close()
    assert not (proj / "uv.lock").exists()
    assert sorted(p.name for p in (proj / ".venv").rglob("*")) == before


async def test_html_only_output_is_readable(workspace: Workspace) -> None:
    s = await started(workspace, "python")
    cell = await s.execute(
        "from IPython.display import display\ndisplay({'text/html': '<b>hi</b>'}, raw=True)", "agent"
    )
    assert cell.text() == "<b>hi</b>"


async def test_shutdown_while_a_detached_process_holds_kernel_stdout(
    workspace: Workspace, tmp_path: Path
) -> None:
    # MathWorksServiceHost does this: it inherits the kernel's stdout pipe and outlives the kernel
    s = await started(workspace, "python")
    pidfile = tmp_path / "pid"
    code = f"import subprocess; subprocess.run(['sh', '-c', 'sleep 120 & echo $! > {pidfile}'])"
    assert (await s.execute(code, "agent")).status == "ok"
    faulthandler.dump_traceback_later(60, exit=True)
    try:
        await asyncio.wait_for(s.shutdown(), 45)
    finally:
        faulthandler.cancel_dump_traceback_later()
        os.kill(int(pidfile.read_text()), signal.SIGTERM)
    assert s.kernel().state == "dead"
