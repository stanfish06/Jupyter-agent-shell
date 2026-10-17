from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import CallToolResult, ImageContent, TextContent

from mash.agent import Agent
from mash.config import free_port
from mash.tools import WORKSPACE_HEADER, ToolServer
from mash.workspace import Manager, Workspace

from .conftest import started
from .test_session import PNG

pytestmark = pytest.mark.anyio


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


@asynccontextmanager
async def mcp_client(server: ToolServer, workspace: str) -> AsyncGenerator[ClientSession]:
    async with (
        create_mcp_http_client(headers={WORKSPACE_HEADER: workspace}) as http,
        streamable_http_client(server.url, http_client=http) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def no_matlab() -> Path:
    raise RuntimeError("MATLAB not found")


@pytest.fixture
async def served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[ToolServer, Workspace]]:
    # hide any julia/R on PATH so their status is deterministic
    monkeypatch.setattr("mash.backends.shutil.which", lambda name: None)
    monkeypatch.setattr("mash.backends.matlab_root", no_matlab)
    path = tmp_path / "ws"
    path.mkdir()
    manager = Manager()
    ws = manager.new_workspace(path)
    server = ToolServer(manager, port=free_port(0))
    await server.start()
    for _ in range(100):
        if server._uvicorn and server._uvicorn.started:
            break
        await asyncio.sleep(0.05)
    yield server, ws
    await manager.close()
    await server.stop()


async def test_tools_over_http(served: tuple[ToolServer, Workspace]) -> None:
    server, ws = served
    async with mcp_client(server, ws.name) as c:
        listing = text(await c.call_tool("list_sessions", {}))
        assert "no sessions" in listing and "python: ready (MASH_PYTHON)" in listing
        assert "julia: not installed" in listing

        r = await c.call_tool("new_session", {"language": "python"})
        assert (
            text(r)
            == 'kernel py (Python) ready in session s1; pass kernel="py" to execute in it. Kernels in s1: py'
        )
        r = await c.call_tool("add_kernel", {"session": "s1", "language": "py"})
        assert text(r).startswith('kernel py2 (Python) ready in session s1; pass kernel="py2"')
        assert text(r).endswith("Kernels in s1: py, py2")

        r = await c.call_tool("execute", {"session": "s1", "code": "1"})
        assert r.is_error and "kernel" in text(r), text(r)
        r = await c.call_tool("execute", {"session": "s1", "code": "x = 5\nx * 2", "kernel": "py"})
        assert text(r).startswith("[py] In[1] by agent (ok):") and "10" in text(r)

        code = f"import base64\nfrom IPython.display import Image\nImage(data=base64.b64decode('{PNG}'))"
        r = await c.call_tool("execute", {"session": "s1", "code": code, "kernel": "py2"})
        assert any(isinstance(p, ImageContent) and p.mime_type == "image/png" for p in r.content)

        assert "x" in text(await c.call_tool("variables", {"session": "s1", "kernel": "py"}))
        r = await c.call_tool("history", {"session": "s1", "kernel": "py"})
        assert text(r).count("In[") == 1
        assert text(await c.call_tool("interrupt", {"session": "s1"})) == "nothing was running"

        r = await c.call_tool("add_kernel", {"session": "s1", "language": "julia"})
        assert "not installed" in text(r) and "mash-setup julia" in text(r)

        asked: list[tuple[str, str]] = []

        async def decline(language: str, who: str) -> str | None:
            asked.append((language, who))
            return "the human declined"

        server.installer = decline
        r = await c.call_tool("add_kernel", {"session": "s1", "language": "R"})
        assert asked == [("r", "agent")] and "declined" in text(r)
        r = await c.call_tool("add_kernel", {"session": "s1", "language": "matlab"})
        assert text(r) == "MATLAB is not available: MATLAB not found" and asked == [("r", "agent")]

        r = await c.call_tool("new_session", {})
        assert text(r).startswith("kernel py (Python) ready in session s2")

        listing = text(await c.call_tool("list_sessions", {}))
        assert "  py: Python, idle" in listing and "  py2: Python, idle" in listing


async def test_agent_context_tags_kernels(workspace: Workspace) -> None:
    s = await started(workspace, "python")

    async def ask(title: str, options: list) -> str | None:
        return None

    agent = Agent(workspace, "http://127.0.0.1:1/mcp", lambda u: None, ask)
    await s.execute("print('hi')", "human")
    prefix = agent.context_prefix()
    assert "- s1: kernels py (Python, idle); created by human" in prefix
    assert "[s1/py] In[1]: print('hi')" in prefix and "hi" in prefix
    assert "In[1]" not in agent.context_prefix()


def test_interpreter_is_this_venv() -> None:
    assert Path(sys.executable).exists()
