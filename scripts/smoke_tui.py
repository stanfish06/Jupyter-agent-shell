from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from mash.tui.app import MashApp
from mash.tui.widgets import CodeInput, SessionView


async def main() -> int:
    os.environ.setdefault("MASH_DEFAULT_KERNEL", "matlab")
    root = Path(tempfile.mkdtemp(prefix="mash-tui-"))
    app = MashApp(root)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.pause()
        wview = app.current_workspace_view()
        assert wview is not None
        session = next(iter(wview.workspace.sessions.values()))
        while not session.kernels:
            await pilot.pause(0.1)
        kernel = session.kernel()
        print("session:", session.name, kernel.id, kernel.state)
        await asyncio.wait_for(kernel.ready.wait(), 300)
        await pilot.pause()
        print("ready:", kernel.state, "desktop:", kernel.desktop_url)
        view = wview.current_view()
        assert isinstance(view, SessionView)
        inp = view.query_one("#code-input", CodeInput)
        inp.focus()
        await pilot.pause()
        for code in ["z = magic(4)", "figure; plot(z); title('tui')", "undefined_thing(3)"]:
            inp.value = code
            await pilot.press("enter")
            for _ in range(600):
                await pilot.pause(0.1)
                last = session.cells[-1] if session.cells else None
                if session.state == "idle" and last and last.code == code and last.finished:
                    break
        await pilot.pause(0.5)
        if any(c.finished is None for c in session.cells):
            print("HUNG cell; pending:", kernel._pending, "state:", kernel.state)
            print("kernel log tail:\n" + "\n".join(kernel.logs[-25:]))
        print(
            "cells:",
            [
                (
                    c.index,
                    c.status,
                    len(c.outputs),
                    round((c.finished - c.started).total_seconds(), 1) if c.finished else None,
                )
                for c in session.cells
            ],
        )
        print("widgets:", len(view.transcript.cells), "figures:", len(view.query(".figure")))
        inp.focus()
        inp.value = "ma"
        inp.cursor_position = 2
        print("before tab: state", session.state, "focused", type(app.focused).__name__)
        await pilot.press("tab")
        await pilot.pause(4.0)
        print("completion:", repr(inp.value), "hint:", str(view.query_one("#hint").render())[:120])
        await pilot.press("ctrl+r")
        await pilot.pause(0.5)
        print("repl visible:", view.repl.has_class("visible"))
        await pilot.press("ctrl+a")
        await pilot.pause(0.5)
        print("agent pane visible:", wview.agent_pane.has_class("visible"))
        for keys in (("ctrl+b", "l"), ("ctrl+b", "h"), ("ctrl+b", "j"), ("ctrl+b", "k"), ("ctrl+w",)):
            for k in keys:
                await pilot.press(k)
                await pilot.pause(0.2)
            f = app.focused
            print(
                "after",
                "+".join(keys),
                "->",
                type(f).__name__,
                getattr(f, "id", None),
                "screen:",
                type(app.screen).__name__,
            )
        await pilot.press("ctrl+b")
        await pilot.pause(0.2)
        print(
            "armed ->",
            type(app.screen).__name__,
            "footer keys:",
            [b.binding.key for b in app.screen.active_bindings.values() if b.binding.show]
            if hasattr(app.screen, "active_bindings")
            else "?",
        )
        await pilot.press("x")
        await pilot.pause(0.2)
        print("after ctrl+b x ->", type(app.focused).__name__, "screen:", type(app.screen).__name__)
        app.save_screenshot(str(root / "mash.svg"))
        print("screenshot:", root / "mash.svg")
        await app.action_quit()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
