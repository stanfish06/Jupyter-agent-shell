from __future__ import annotations

import asyncio
import os
from pathlib import Path

import nbformat
import pytest

from mash.engine import EngineRepl
from mash.workspace import Workspace

from .conftest import started

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not os.environ.get("MASH_TEST_MATLAB"), reason="set MASH_TEST_MATLAB=1 (starts MATLAB)"
    ),
]


async def test_matlab_and_python_in_one_session(tmp_path: Path) -> None:
    path = tmp_path / "it's ws"
    path.mkdir()
    ws = Workspace("quoted", path)
    try:
        s = await started(ws, "matlab", "python", timeout=420)
        m = s.kernels["matlab"]
        a, b = await asyncio.gather(
            s.execute("a = 40 + 2", "human", "matlab"),
            s.execute("import time; time.sleep(1); print('py')", "agent", "py"),
        )
        assert a.status == "ok" and "42" in a.text(), a.text()
        assert b.status == "ok"
        pwd = await s.execute("disp(pwd)", "agent", "matlab")
        assert "it's ws" in pwd.text()
        fig = await s.execute("figure; plot(1:10)", "agent", "matlab")
        assert any(k.startswith("image/") for o in fig.outputs if o.kind == "data" for k in o.data)
        assert "a" in await m.variables()

        repl = EngineRepl(m.share_name)
        assert await repl.connect(), repl.error
        seen: list[str] = []
        await repl.eval("disp(a)", seen.append)
        repl.close()
        assert "42" in "".join(seen)

        snap = await m.snapshot()
        assert snap is not None and snap.status == "ok" and (s.record_dir / "matlab.mat").exists()
        nb = nbformat.read(s.record_dir / "matlab.ipynb", as_version=4)
        assert nb.metadata.kernelspec.name == "jupyter_matlab_kernel"
        assert (s.record_dir / "matlab.m").read_text().startswith("%% In[1] you")
    finally:
        await ws.close()
