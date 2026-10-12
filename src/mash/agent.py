from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acp import PROTOCOL_VERSION, Client, RequestError, connect_to_agent
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    ConfigOptionUpdate,
    CreateTerminalResponse,
    DeclineElicitationResponse,
    DeniedOutcome,
    EnvVariable,
    FileSystemCapabilities,
    HttpHeader,
    HttpMcpServer,
    Implementation,
    PermissionOption,
    ReadTextFileResponse,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallUpdate,
    WriteTextFileResponse,
)

from mash.config import ADAPTER
from mash.tools import WORKSPACE_HEADER

if TYPE_CHECKING:
    from mash.workspace import Workspace

UpdateHandler = Callable[[Any], None]
PermissionHandler = Callable[[str, list[PermissionOption]], Awaitable[str | None]]


class Agent(Client):
    def __init__(
        self,
        workspace: Workspace,
        mcp_url: str,
        on_update: UpdateHandler,
        ask_permission: PermissionHandler,
    ) -> None:
        self.workspace = workspace
        self.mcp_url = mcp_url
        self.on_update = on_update
        self.ask_permission = ask_permission
        self.session_id: str | None = None
        self.mode: str | None = None
        self.config_options: list[Any] = []
        self.busy = False
        self.last_seen_cell_idx: dict[str, int] = {}
        self._proc: aio_subprocess.Process | None = None
        self._conn: Any = None

    async def start(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("CLAUDE_CODE_", "CLAUDECODE", "CLAUDE_PID"))
        }
        self._proc = await asyncio.create_subprocess_exec(
            "node",
            str(ADAPTER),
            cwd=str(self.workspace.path),
            env=env,
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.DEVNULL,
        )
        assert self._proc.stdin and self._proc.stdout
        self._conn = connect_to_agent(self, self._proc.stdin, self._proc.stdout)
        await self._conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(readTextFile=True, writeTextFile=True)
            ),
            client_info=Implementation(name="mash", title="mash", version="0.1.0"),
        )
        session = await self._conn.new_session(
            cwd=str(self.workspace.path),
            mcp_servers=[
                HttpMcpServer(
                    type="http",
                    name="mash",
                    url=self.mcp_url,
                    headers=[HttpHeader(name=WORKSPACE_HEADER, value=self.workspace.name)],
                )
            ],
        )
        self.session_id = session.session_id
        self.config_options = list(session.config_options or [])
        self.mode = session.modes.current_mode_id if session.modes else None
        if session.modes and self.mode != "default":
            ids = [m.id for m in session.modes.available_modes]
            if "default" in ids:
                await self._conn.set_session_mode(session_id=self.session_id, mode_id="default")
                self.mode = "default"
        for sess in self.workspace.sessions.values():
            self.last_seen_cell_idx[sess.name] = len(sess.cells)

    def context_prefix(self) -> str:
        lines = ["Workspace sessions:"]
        for sess in self.workspace.sessions.values():
            lines.append(f"- {sess.name}: {sess.state}, created by {sess.created_by}")
        fresh = []
        for sess in self.workspace.sessions.values():
            start = self.last_seen_cell_idx.get(sess.name, 0)
            for cell in sess.cells[start:]:
                if cell.author == "human":
                    out = cell.text().strip()
                    fresh.append(f"[{sess.name}] In[{cell.index}]: {cell.code}")
                    if out:
                        fresh.append("    " + out.replace("\n", "\n    ")[:2000])
            self.last_seen_cell_idx[sess.name] = len(sess.cells)
        if fresh:
            lines.append("Cells the human ran since your last turn:")
            lines.extend(fresh)
        return "\n".join(lines)

    async def prompt(self, text: str) -> str:
        if self._conn is None or self.session_id is None:
            raise RuntimeError("agent not started")
        self.busy = True
        try:
            body = f"{self.context_prefix()}\n\n{text}"
            resp = await self._conn.prompt(
                session_id=self.session_id, prompt=[TextContentBlock(type="text", text=body)]
            )
            return resp.stop_reason
        finally:
            self.busy = False

    def option(self, config_id: str) -> Any | None:
        return next((o for o in self.config_options if o.id == config_id), None)

    def option_label(self, config_id: str) -> str:
        opt = self.option(config_id)
        if opt is None:
            return "?"
        for choice in getattr(opt, "options", []) or []:
            if choice.value == opt.current_value:
                return choice.name
        return str(opt.current_value)

    async def set_option(self, config_id: str, value: str) -> None:
        if self._conn is None or self.session_id is None:
            raise RuntimeError("agent not started")
        resp = await self._conn.set_config_option(
            config_id=config_id, session_id=self.session_id, value=value
        )
        if resp.config_options is not None:
            self.config_options = list(resp.config_options)

    async def cancel(self) -> None:
        if self._conn and self.session_id and self.busy:
            await self._conn.cancel(session_id=self.session_id)

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            with contextlib.suppress(ProcessLookupError):
                await self._proc.wait()

    def _auto_allow(self, tool_call: ToolCallUpdate) -> bool:
        raw = tool_call.raw_input if isinstance(tool_call.raw_input, dict) else {}
        name = raw.get("session")
        if not name or name not in self.workspace.sessions:
            return False
        return self.workspace.sessions[name].created_by == "agent"

    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse:
        allow = next((o for o in options if o.kind == "allow_once"), None)
        if allow and self._auto_allow(tool_call):
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", optionId=allow.option_id)
            )
        chosen = await self.ask_permission(tool_call.title or "tool call", options)
        if chosen is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", optionId=chosen))

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if isinstance(update, ConfigOptionUpdate) and update.config_options is not None:
            self.config_options = list(update.config_options)
        self.on_update(update)

    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        lines = Path(path).read_text().splitlines(keepends=True)
        start = (line or 1) - 1
        end = start + limit if limit else None
        return ReadTextFileResponse(content="".join(lines[start:end]))

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return WriteTextFileResponse()

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/kill")

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        return DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        return None

    def on_connect(self, conn: Any) -> None:
        return None

    async def ext_method(self, method: str, params: dict) -> dict:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        return None
