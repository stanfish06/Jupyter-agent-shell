from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from acp.schema import AgentMessageChunk, PermissionOption, TextContentBlock, ToolCallStart

from mash.agent import Agent
from mash.engine import EngineRepl
from mash.session import SessionEvent
from mash.tools import ToolServer
from mash.workspace import Manager


def on_event(ev: SessionEvent) -> None:
    if ev.kind == "state":
        print(f"[{ev.session.name}] state={ev.session.state} owner={ev.session.owner}")
    elif ev.kind == "cell_started" and ev.cell:
        print(f"[{ev.session.name}] In[{ev.cell.index}] {ev.cell.author}: {ev.cell.code}")
    elif ev.kind == "output" and ev.output:
        print(f"[{ev.session.name}]   out: {ev.output.text().strip()[:200]}")
    elif ev.kind == "cell_finished" and ev.cell:
        print(f"[{ev.session.name}] -> {ev.cell.status}")


def on_update(update: Any) -> None:
    if isinstance(update, AgentMessageChunk) and isinstance(update.content, TextContentBlock):
        print(update.content.text, end="", flush=True)
    elif isinstance(update, ToolCallStart):
        print(f"\n| tool: {update.title}")


async def ask(title: str, options: list[PermissionOption]) -> str | None:
    allow = next(o for o in options if o.kind.startswith("allow"))
    print(f"\n| permission asked for '{title}', answering {allow.kind}")
    return allow.option_id


async def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="mash-"))
    print("workspace:", root)
    manager = Manager()
    tools = ToolServer(manager, port=8766)
    await tools.start()
    ws = manager.new_workspace(root)
    ws.subscribe(on_event)
    main_sess = ws.new_session("main", languages=["matlab"])
    kernel = main_sess.kernel()
    try:
        await asyncio.wait_for(kernel.ready.wait(), 240)
    except TimeoutError:
        print("start timeout; kernel log tail:")
        print("\n".join(kernel.logs[-40:]))
        return 1
    if kernel.state == "dead":
        print("kernel failed:", kernel.error)
        print("\n".join(kernel.logs[-30:]))
        return 1
    print("desktop:", kernel.desktop_url)

    cell = await main_sess.execute("a = 40; b = a + 2", "human")
    print("human cell text:", cell.text().strip())
    try:
        cell = await asyncio.wait_for(
            main_sess.execute("figure; plot(sin(0:0.1:6)); title('smoke')", "human"), 90
        )
    except TimeoutError:
        print("figure cell timed out; pending:", kernel._pending, "kernel log tail:")
        print("\n".join(kernel.logs[-40:]))
        await manager.close()
        return 1
    print("figure outputs:", [(o.kind, list(o.data)) for o in cell.outputs])
    nb = main_sess.record_dir / "matlab.ipynb"
    print("notebook:", nb, nb.stat().st_size, "bytes")

    repl = EngineRepl(kernel.share_name)
    ok = await repl.connect()
    print("engine connected:", ok, repl.error)
    if ok:
        code = "disp(b); for k=1:3, fprintf('tick %d\\n', k); pause(0.3); end"
        await repl.eval(code, lambda t: print("  repl:", t, end=""))

    agent = Agent(ws, tools.url, on_update, ask)
    await agent.start()
    print("\nagent session:", agent.session_id, "mode:", agent.mode)
    ask_text = (
        "Use the mash tools. In session 'main', check what variable b holds, "
        "then compute c = b * 2 and report c."
    )
    reason = await agent.prompt(ask_text)
    print("\nstop:", reason)
    print("cells now:", [(c.index, c.author, c.code) for c in main_sess.cells])

    await agent.stop()
    await manager.close()
    await tools.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
