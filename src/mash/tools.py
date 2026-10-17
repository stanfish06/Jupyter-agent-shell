from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Generator
from typing import TYPE_CHECKING

import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from mash.backends import default_language, get_backend, language_status
from mash.config import PREFERRED_MCP_PORT, free_port
from mash.session import IMAGE_MIMES

if TYPE_CHECKING:
    from mash.session import Cell, Session
    from mash.workspace import Manager, Workspace

WORKSPACE_HEADER = "x-mash-workspace"

# (language, requested by) -> error text, or None once the language is set up
Installer = Callable[[str, str], Awaitable[str | None]]

INSTRUCTIONS = """\
mash gives you Jupyter kernels (MATLAB, Python, R, Julia) in sessions shared live with a human.
A session holds one or more kernels. Each kernel is a separate process with its own variables.
Kernels never share variables: to move data between languages, write a file in the working
directory (CSV, Parquet, .mat, ...) from one kernel and read it in another.
The human sees every cell you run and may run cells between your turns.
Use list_sessions to see sessions, their kernels, and which languages are ready.
Use history to see recent cells and execute to run code. execute and variables always take kernel=,
the kernel's name from list_sessions (py, r, matlab, jl, ...). Variables persist between execute
calls on the same kernel.
Use add_kernel to start another language in a session, and new_session for a fresh set of kernels.
"""


@contextlib.contextmanager
def tool_errors() -> Generator[None]:
    # MCPServer hides the text of any other exception from the agent
    try:
        yield
    except TimeoutError as e:
        raise ToolError("timed out waiting for the kernel") from e
    except (KeyError, ValueError, RuntimeError) as e:
        raise ToolError(str(e.args[0]) if e.args else type(e).__name__) from e


def render_cell(cell: Cell) -> list[TextContent | ImageContent]:
    parts: list[TextContent | ImageContent] = []
    text = f"[{cell.kernel}] In[{cell.index}] by {cell.author} ({cell.status}):\n{cell.code}\n"
    body = cell.text().rstrip()
    if body:
        text += f"--- output ---\n{body}\n"
    parts.append(TextContent(type="text", text=text))
    for o in cell.outputs:
        if o.kind != "data":
            continue
        for mime, data in o.data.items():
            if mime in IMAGE_MIMES and isinstance(data, str):
                parts.append(ImageContent(type="image", data=data, mime_type=mime))
    return parts


class ToolServer:
    def __init__(self, manager: Manager, port: int | None = None) -> None:
        self.manager = manager
        self.port = port or free_port(PREFERRED_MCP_PORT)
        self.server = MCPServer("mash", instructions=INSTRUCTIONS)
        self.installer: Installer | None = None
        self._uvicorn: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._register()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def _workspace(self, ctx: Context) -> Workspace:
        name = (ctx.headers or {}).get(WORKSPACE_HEADER)
        if not name:
            raise ValueError(f"missing {WORKSPACE_HEADER} header")
        return self.manager.get(name)

    async def start_kernel(self, sess: Session, language: str, name: str | None = None) -> str:
        backend = get_backend(language)
        ok, detail = await asyncio.to_thread(backend.status, sess.cwd)
        if not ok:
            if not backend.installable:
                return f"{backend.title} is not available: {detail}"
            if self.installer is None:
                hint = f"ask the human to run: mash-setup {backend.name}"
                return f"{backend.title}: {backend.missing} ({detail}); {hint}"
            err = await self.installer(backend.name, "agent")
            if err:
                return f"{backend.title} was not set up: {err}"
        handle = sess.add_kernel(backend.name, name)
        await asyncio.wait_for(handle.ready.wait(), backend.start_timeout + 30)
        if handle.state == "dead":
            return f"kernel {handle.id} failed to start: {handle.error}"
        kernels = ", ".join(sess.kernels)
        return (
            f"kernel {handle.id} ({backend.title}) ready in session {sess.name}; "
            f'pass kernel="{handle.id}" to execute in it. Kernels in {sess.name}: {kernels}'
        )

    def _register(self) -> None:
        s = self.server

        @s.tool(description="List sessions, their kernels and states, and which languages are ready.")
        async def list_sessions(ctx: Context) -> str:
            with tool_errors():
                ws = self._workspace(ctx)
                lines = []
                for sess in ws.sessions.values():
                    lines.append(f"{sess.name} (created by {sess.created_by})")
                    for k in sess.kernels.values():
                        owner = f" by {k.owner}" if k.owner else ""
                        error = f": {k.error}" if k.state == "dead" and k.error else ""
                        lines.append(f"  {k.id}: {k.backend.title}, {k.state}{owner}{error}")
                    if not sess.kernels:
                        lines.append("  no kernels")
                if not lines:
                    lines.append("no sessions")
                langs = await asyncio.to_thread(language_status, ws.path)
                lines.append("languages:")
                for b, ok, detail in langs:
                    state = "ready" if ok else b.missing
                    lines.append(f"  {b.name}: {state} ({detail})")
                return "\n".join(lines)

        @s.tool(
            description="Run code in one kernel of a session and return its output. kernel is the "
            "kernel's name from list_sessions (py, r, matlab, jl, ...). Figures are returned as images."
        )
        async def execute(
            session: str, kernel: str, code: str, ctx: Context
        ) -> list[TextContent | ImageContent]:
            with tool_errors():
                sess = self._workspace(ctx).get(session)
                cell = await sess.execute(code, "agent", kernel)
                return render_cell(cell)

        @s.tool(
            description="List variables in a kernel "
            "(whos in MATLAB, %whos in Python, ls() in R, varinfo() in Julia)."
        )
        async def variables(session: str, kernel: str, ctx: Context) -> str:
            with tool_errors():
                k = self._workspace(ctx).get(session).kernel(kernel)
                return await k.variables() or "no variables"

        @s.tool(description="Show recent cells run in a session, optionally only those of one kernel.")
        async def history(
            session: str, ctx: Context, n: int = 10, kernel: str | None = None
        ) -> list[TextContent | ImageContent]:
            with tool_errors():
                cells = self._workspace(ctx).get(session).history(n, kernel)
                if not cells:
                    return [TextContent(type="text", text="no cells yet")]
                return [p for c in cells for p in render_cell(c)]

        @s.tool(
            description="Start a new session with one kernel. language is matlab, python, r or julia; "
            "the default is python. Returns once the kernel is ready."
        )
        async def new_session(ctx: Context, name: str | None = None, language: str | None = None) -> str:
            with tool_errors():
                sess = self._workspace(ctx).new_session(name, created_by="agent")
                return await self.start_kernel(sess, language or default_language())

        @s.tool(
            description="Start another kernel (matlab, python, r or julia) in an existing session. "
            "If it is not set up (no runtime, or no Python venv), mash asks the human before setting it up. "
            "name defaults to the language's short name (matlab, py, r, jl), numbered if taken."
        )
        async def add_kernel(session: str, language: str, ctx: Context, name: str | None = None) -> str:
            with tool_errors():
                sess = self._workspace(ctx).get(session)
                return await self.start_kernel(sess, language, name)

        @s.tool(
            description="Restart a dead or stuck kernel. Its variables are lost; its cells stay in history."
        )
        async def restart_kernel(session: str, kernel: str, ctx: Context) -> str:
            with tool_errors():
                sess = self._workspace(ctx).get(session)
                handle = await sess.restart_kernel(kernel)
                await asyncio.wait_for(handle.ready.wait(), handle.backend.start_timeout + 30)
                if handle.state == "dead":
                    return f"kernel {handle.id} failed to start: {handle.error}"
                return f"kernel {handle.id} restarted"

        @s.tool(description="Interrupt what is running in a kernel, or in every busy kernel of the session.")
        async def interrupt(session: str, ctx: Context, kernel: str | None = None) -> str:
            with tool_errors():
                sess = self._workspace(ctx).get(session)
                targets = (
                    [sess.kernel(kernel)]
                    if kernel
                    else [k for k in sess.kernels.values() if k.state == "busy"]
                )
                for k in targets:
                    await k.interrupt()
                return (
                    f"interrupt sent to {', '.join(k.id for k in targets)}"
                    if targets
                    else "nothing was running"
                )

    async def start(self) -> None:
        app = self.server.streamable_http_app(stateless_http=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self._uvicorn = uvicorn.Server(config)
        self._task = asyncio.create_task(self._uvicorn.serve())

    async def stop(self) -> None:
        if self._uvicorn:
            self._uvicorn.should_exit = True
        if self._task:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, 5)
