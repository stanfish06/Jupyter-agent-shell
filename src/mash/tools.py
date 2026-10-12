from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ImageContent, TextContent

from mash.config import MATLAB_START_TIMEOUT, PREFERRED_MCP_PORT, free_port

if TYPE_CHECKING:
    from mash.session import Cell
    from mash.workspace import Manager, Workspace

WORKSPACE_HEADER = "x-mash-workspace"

INSTRUCTIONS = """\
mash (matlab-agent-shell) gives you MATLAB sessions shared live with a human.
Each session is a separate MATLAB process with its own variables.
The human sees every cell you run and may run cells between your turns.
Use list_sessions to see what exists, history to see recent cells, and execute to run code.
Variables persist between execute calls in the same session.
Use new_session when you want a MATLAB whose variables do not interfere with an existing session.
"""


def render_cell(cell: Cell) -> list[TextContent | ImageContent]:
    parts: list[TextContent | ImageContent] = []
    text = f"In[{cell.index}] by {cell.author} ({cell.status}):\n{cell.code}\n"
    body = cell.text().rstrip()
    if body:
        text += f"--- output ---\n{body}\n"
    parts.append(TextContent(type="text", text=text))
    for o in cell.outputs:
        if o.kind != "data":
            continue
        for mime, data in o.data.items():
            if mime.startswith("image/") and isinstance(data, str):
                parts.append(ImageContent(type="image", data=data, mime_type=mime))
    return parts


class ToolServer:
    def __init__(self, manager: Manager, port: int | None = None) -> None:
        self.manager = manager
        self.port = port or free_port(PREFERRED_MCP_PORT)
        self.server = MCPServer("mash", instructions=INSTRUCTIONS)
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

    def _register(self) -> None:
        s = self.server

        @s.tool(description="List MATLAB sessions in this workspace with their state and who created them.")
        async def list_sessions(ctx: Context) -> str:
            ws = self._workspace(ctx)
            if not ws.sessions:
                return "no sessions"
            lines = []
            for sess in ws.sessions.values():
                owner = f" by {sess.owner}" if sess.owner else ""
                lines.append(f"{sess.name}: {sess.state}{owner}, created by {sess.created_by}")
            return "\n".join(lines)

        @s.tool(
            description="Run MATLAB code in a session and return its output. Figures are returned as images."
        )
        async def execute(session: str, code: str, ctx: Context) -> list[TextContent | ImageContent]:
            sess = self._workspace(ctx).get(session)
            cell = await sess.execute(code, "agent")
            return render_cell(cell)

        @s.tool(description="List variables in a session's workspace (MATLAB whos).")
        async def whos(session: str, ctx: Context) -> str:
            return await self._workspace(ctx).get(session).whos() or "workspace is empty"

        @s.tool(description="Show recent cells run in a session.")
        async def history(session: str, ctx: Context, n: int = 10) -> list[TextContent | ImageContent]:
            cells = self._workspace(ctx).get(session).history(n)
            if not cells:
                return [TextContent(type="text", text="no cells yet")]
            return [p for c in cells for p in render_cell(c)]

        @s.tool(
            description="Start a new MATLAB session with its own process and variables. Returns once ready."
        )
        async def new_session(ctx: Context, name: str | None = None) -> str:
            ws = self._workspace(ctx)
            sess = ws.new_session(name, created_by="agent")
            await asyncio.wait_for(sess.ready.wait(), MATLAB_START_TIMEOUT)
            if sess.state == "dead":
                return f"session {sess.name} failed to start: {sess.error}"
            return f"session {sess.name} ready"

        @s.tool(description="Interrupt whatever is running in a session.")
        async def interrupt(session: str, ctx: Context) -> str:
            sess = self._workspace(ctx).get(session)
            await sess.interrupt()
            return f"interrupt sent to {session}"

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
