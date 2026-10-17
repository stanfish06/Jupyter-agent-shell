from __future__ import annotations

import asyncio
import os

import nbformat
import pytest

from mash.backends import JuliaBackend, RBackend
from mash.session import Cell, Session
from mash.workspace import Workspace

from .conftest import started

pytestmark = pytest.mark.anyio


@pytest.fixture
def runtime_home(monkeypatch: pytest.MonkeyPatch) -> None:
    home = os.environ.get("MASH_TEST_HOME")
    if not home:
        pytest.skip("set MASH_TEST_HOME to a mash data dir with R and Julia installed (mash-setup r julia)")
    monkeypatch.setenv("MASH_HOME", home)


def has_image(cell: Cell) -> bool:
    return any(m in o.data for o in cell.outputs if o.kind == "data" for m in ("image/png", "image/jpeg"))


async def interrupt_sleep(s: Session, kernel: str, code: str) -> None:
    running = asyncio.create_task(s.execute(code, "agent", kernel))
    await asyncio.sleep(2)
    await s.kernel(kernel).interrupt()
    cell = await asyncio.wait_for(running, 20)
    assert cell.finished is not None
    assert s.kernel(kernel).state == "idle"


async def test_r_kernel(runtime_home: None, workspace: Workspace) -> None:
    ok, detail = RBackend().status(workspace.path)
    if not ok:
        pytest.skip(detail)
    s = await started(workspace, "r", "python", timeout=120)
    mean = await s.execute("x <- c(1, 2, 3)\nmean(x)", "human", "r")
    assert mean.status == "ok" and "2" in mean.text()
    df = await s.execute("data.frame(a = 1:2, b = c('u', 'v'))", "agent", "r")
    assert df.status == "ok" and "u" in df.text()
    plot = await s.execute("plot(1:10)", "agent", "r")
    assert plot.status == "ok" and has_image(plot)
    warn = await s.execute("warning('careful'); 1", "agent", "r")
    assert warn.status == "ok"
    err = await s.execute("stop('boom')", "agent", "r")
    assert err.status == "error" and "boom" in err.text()
    listing = await s.kernel("r").variables()
    assert "x" in listing and "numeric" in listing
    await interrupt_sleep(s, "r", "Sys.sleep(30)")
    assert (await s.execute("print(sum(x))", "agent", "r")).status == "ok"
    assert (await s.execute("1 + 1", "agent", "py")).status == "ok"
    nb = nbformat.read(s.record_dir / "r.ipynb", as_version=4)
    assert nb.metadata.kernelspec.name == "ir" and nb.cells[0].execution_count == 1
    assert (s.record_dir / "r.R").read_text().startswith("# %% In[1] you")


async def test_julia_kernel(runtime_home: None, workspace: Workspace) -> None:
    ok, detail = JuliaBackend().status(workspace.path)
    if not ok:
        pytest.skip(detail)
    s = await started(workspace, "julia", timeout=300)
    cell = await s.execute("x = [1, 2, 3]\nsum(x)", "human")
    assert cell.status == "ok" and "6" in cell.text()
    err = await s.execute('error("boom")', "agent")
    assert err.status == "error" and "boom" in err.text()
    listing = await s.kernel().variables()
    assert "x" in listing
    await interrupt_sleep(s, "jl", "sleep(30)")
    assert (await s.execute("x .* 2", "agent")).status == "ok"
    nb = nbformat.read(s.record_dir / "jl.ipynb", as_version=4)
    assert nb.metadata.kernelspec.name.startswith("julia-1.")
    assert (s.record_dir / "jl.jl").read_text().startswith("# %% In[1] you")
