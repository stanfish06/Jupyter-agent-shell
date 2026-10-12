from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import os
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, Client, RequestError, connect_to_agent
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    ClientCapabilities,
    Implementation,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallStart,
    ToolCallUpdate,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "node_modules" / ".bin" / "claude-agent-acp"


class SmokeClient(Client):
    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse:
        allow = next((o for o in options if o.kind in ("allow_once", "allow_always")), options[0])
        print(f"| permission: {tool_call.title} -> {allow.kind}")
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", optionId=allow.option_id))

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if isinstance(update, AgentMessageChunk) and isinstance(update.content, TextContentBlock):
            print(update.content.text, end="", flush=True)
        elif isinstance(update, AgentThoughtChunk):
            pass
        elif isinstance(update, ToolCallStart):
            print(f"\n| tool: {update.title}")
        else:
            print(f"\n| {type(update).__name__}")

    async def ext_method(self, method: str, params: dict) -> dict:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        pass


async def main(prompt: str) -> int:
    proc = await asyncio.create_subprocess_exec(
        "node", str(ADAPTER), stdin=aio_subprocess.PIPE, stdout=aio_subprocess.PIPE
    )
    assert proc.stdin and proc.stdout
    conn = connect_to_agent(SmokeClient(), proc.stdin, proc.stdout)
    init = await conn.initialize(
        protocol_version=PROTOCOL_VERSION,
        client_capabilities=ClientCapabilities(),
        client_info=Implementation(name="mash", title="mash", version="0.0.1"),
    )
    print("agent:", init.agent_info, "\nauth methods:", [m.id for m in init.auth_methods or []])
    session = await conn.new_session(mcp_servers=[], cwd=os.getcwd())
    print("session:", session.session_id)
    resp = await conn.prompt(
        session_id=session.session_id, prompt=[TextContentBlock(type="text", text=prompt)]
    )
    print("\nstop reason:", resp.stop_reason)
    proc.terminate()
    await proc.wait()
    return 0


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Reply with the single word: pong"
    sys.exit(asyncio.run(main(text)))
