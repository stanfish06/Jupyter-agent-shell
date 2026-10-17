from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import OptionList, Static

from mash.tui.app import ChoiceScreen, MashApp
from mash.tui.widgets import CodeInput, SessionView

pytestmark = pytest.mark.anyio


async def until(pilot: Pilot, cond: Callable[[], object], timeout: float = 60) -> None:
    for _ in range(int(timeout / 0.1)):
        if cond():
            return
        await pilot.pause(0.1)
    raise AssertionError("timed out waiting for condition")


def label(view: SessionView) -> str:
    return str(view.query_one("#prompt-label", Static).render())


async def run(pilot: Pilot, view: SessionView, code: str) -> None:
    inp = view.query_one("#code-input", CodeInput)
    inp.focus()
    inp.value = code
    n = len(view.session.cells)
    await pilot.press("enter")
    await until(pilot, lambda: len(view.session.cells) > n and view.session.cells[-1].finished)


async def test_tui_runs_two_kernels_in_one_session(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    app = MashApp(root)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.pause()
        wview = app.current_workspace_view()
        assert wview is not None
        session = next(iter(wview.workspace.sessions.values()))
        await until(pilot, lambda: "py" in session.kernels and session.kernels["py"].state == "idle")
        view = wview.current_view()
        assert isinstance(view, SessionView)
        await pilot.pause(0.2)
        assert label(view) == "py >>"

        await run(pilot, view, "print(6 * 7)")
        first = view.transcript.get(session.cells[0])
        assert first is not None
        assert str(first.query_one(".cell-header", Static).render()) == "[py] In[1] you ok"

        await pilot.press("ctrl+k")
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        options = app.screen.query_one(OptionList)
        ids = [options.get_option_at_index(i).id for i in range(options.option_count)]
        assert ids[0] == "use:py" and "add:python" in ids and "add:julia" in ids
        options.highlighted = ids.index("add:python")
        await pilot.press("enter")
        await until(pilot, lambda: "py2" in session.kernels and session.kernels["py2"].state == "idle")
        await pilot.pause(0.2)
        assert view.active == "py2" and label(view) == "py2 >>"

        await run(pilot, view, "print('second')")
        assert [(c.index, c.kernel) for c in session.cells] == [(1, "py"), (2, "py2")]
        assert wview.tab_label(session) == "s1"

        inp = view.query_one("#code-input", CodeInput)
        inp.value = "import time; time.sleep(30)"
        await pilot.press("enter")
        await until(pilot, lambda: session.kernels["py2"].state == "busy")
        await pilot.press("ctrl+c")
        await until(pilot, lambda: session.cells[-1].finished)
        assert session.cells[-1].status == "error" and "KeyboardInterrupt" in session.cells[-1].text()
        await app.action_quit()


async def test_agent_install_request_asks_the_human(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[str] = []
    monkeypatch.setattr("mash.backends.JuliaBackend._install", lambda self, log: installed.append("julia"))
    root = tmp_path / "ws"
    root.mkdir()
    app = MashApp(root)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.pause()
        installer = app.tools.installer
        assert installer is not None
        results: list[str | None] = []

        async def ask() -> None:
            results.append(await installer("julia", "agent"))

        for answer in ("no", "yes"):
            app.run_worker(ask())
            await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
            screen = app.screen
            assert isinstance(screen, ChoiceScreen) and "agent wants to install Julia" in screen.title_text
            options = app.screen.query_one(OptionList)
            assert options.highlighted == 1  # Cancel is the default
            options.highlighted = 0 if answer == "yes" else 1
            n = len(results)
            await pilot.press("enter")
            await until(pilot, lambda n=n: len(results) > n)
        assert results == ["the human declined to install Julia", None]
        assert installed == ["julia"]
        await app.action_quit()


async def test_first_launch_without_python_env_asks_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MASH_PYTHON")

    def install(self: object, log: object) -> None:
        monkeypatch.setenv("MASH_PYTHON", sys.executable)

    monkeypatch.setattr("mash.backends.PythonBackend._install", install)
    root = tmp_path / "ws"
    root.mkdir()
    app = MashApp(root)
    async with app.run_test(size=(140, 45)) as pilot:
        await until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        screen = app.screen
        assert isinstance(screen, ChoiceScreen) and screen.title_text == "Create a Python venv?"
        options = screen.query_one(OptionList)
        await until(pilot, lambda: options.highlighted == 1)  # on_mount has put the default on Cancel
        options.highlighted = 0
        await pilot.press("enter")
        wview = app.current_workspace_view()
        assert wview is not None
        session = next(iter(wview.workspace.sessions.values()))
        await until(pilot, lambda: "py" in session.kernels and session.kernels["py"].state == "idle")
        await app.action_quit()
