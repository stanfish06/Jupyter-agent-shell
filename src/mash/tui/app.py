from __future__ import annotations

import asyncio
import contextlib
import webbrowser
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    ConfigOptionUpdate,
    PermissionOption,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)
from textual import events, on
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Static,
    Tab,
    Tabs,
)
from textual.widgets.option_list import Option

from mash.agent import Agent
from mash.backends import default_language, get_backend, language_status
from mash.session import KernelHandle, Session, SessionEvent
from mash.tools import ToolServer
from mash.tui.widgets import (
    STATE_MARK,
    ChatLog,
    CodeInput,
    PromptArea,
    SessionView,
    matlab_kernel,
    page_scroll,
    wid,
)
from mash.workspace import Manager, Workspace


class PermissionScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Deny")]

    def __init__(self, title: str, options: list[PermissionOption]) -> None:
        super().__init__()
        self.title_text = title
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Agent wants to: {self.title_text}")
            with Horizontal():
                for o in self.options:
                    variant = "success" if o.kind.startswith("allow") else "error"
                    yield Button(o.name, id=f"opt-{wid(o.option_id)}", variant=variant)

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        for o in self.options:
            if event.button.id == f"opt-{wid(o.option_id)}":
                self.dismiss(o.option_id)
                return
        self.dismiss(None)


KEYS = [
    ("enter", "run the input line", "shell"),
    ("tab", "complete against the live workspace", "shell"),
    ("up / down", "input history", "shell"),
    ("pageup / pagedown", "scroll the pane above the focused input", "shell"),
    ("ctrl+k", "switch kernel, add a language, or restart a dead kernel", "global"),
    ("ctrl+n", "new session with the default kernel", "global"),
    ("ctrl+t", "new workspace", "global"),
    ("ctrl+a", "toggle agent pane", "global"),
    ("ctrl+r", "toggle raw MATLAB engine REPL", "global"),
    ("ctrl+o", "open MATLAB desktop in browser", "global"),
    ("ctrl+c", "interrupt the active kernel (or the only busy one)", "global"),
    ("ctrl+x", "cancel the agent's turn", "global"),
    ("ctrl+w", "focus workspace list", "global"),
    ("escape", "leave workspace list, back to previous focus", "workspaces"),
    ("enter", "switch to the selected workspace", "workspaces"),
    ("ctrl+b", "arm prefix", "global"),
    ("ctrl+p", "command palette", "global"),
    ("f1", "this key list", "global"),
    ("ctrl+q", "snapshot MATLAB workspaces and quit", "global"),
    ("h j k l / arrows", "move focus between panes", "prefix"),
    ("n / p", "next / previous session tab", "prefix"),
    ("1-9", "nth session tab", "prefix"),
    ("escape", "cancel", "prefix"),
    ("enter", "send to agent", "agent"),
    ("/model, /effort, /mode", "pick model, effort level, or permission mode", "agent"),
    ("shift+enter / ctrl+j", "newline in the agent prompt", "agent"),
    ("/", "focus the filter", "keys"),
    ("escape", "close", "keys"),
]


class KeysScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "close"), Binding("slash", "search", "search", key_display="/")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Input(placeholder="filter keys", id="keys-filter")
            yield DataTable(id="keys-table", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("key", "action", "where")
        self.fill("")
        self.query_one("#keys-filter", Input).focus()

    def fill(self, needle: str) -> None:
        table = self.query_one(DataTable)
        table.clear()
        needle = needle.lower()
        for key, action, where in KEYS:
            if needle in f"{key} {action} {where}".lower():
                table.add_row(key, action, where)

    @on(Input.Changed, "#keys-filter")
    def filter_changed(self, event: Input.Changed) -> None:
        self.fill(event.value)

    def action_search(self) -> None:
        self.query_one("#keys-filter", Input).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "slash" and self.focused is not self.query_one("#keys-filter", Input):
            event.stop()
            event.prevent_default()
            self.action_search()


class PrefixScreen(ModalScreen[None]):
    MOVE_KEYS = ("h", "j", "k", "l", "left", "down", "up", "right")
    TAB_KEYS = ("n", "p", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    BINDINGS = [
        Binding("h", "move('h')", "focus", key_display="h j k l ← ↓ ↑ →"),
        *[Binding(k, f"move('{k}')", show=False) for k in MOVE_KEYS[1:]],
        Binding("n", "tab('n')", "next tab"),
        Binding("p", "tab('p')", "prev tab"),
        Binding("1", "tab('1')", "nth tab", key_display="1-9"),
        *[Binding(k, f"tab('{k}')", show=False) for k in TAB_KEYS[3:]],
        Binding("escape", "dismiss", "cancel"),
    ]

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_key(self, event: events.Key) -> None:
        if event.key not in self.MOVE_KEYS + self.TAB_KEYS + ("escape",):
            event.stop()
            self.dismiss()

    def action_move(self, key: str) -> None:
        cast("MashApp", self.app).prefix_move(key)
        self.dismiss()

    def action_tab(self, key: str) -> None:
        cast("MashApp", self.app).prefix_tab(key)
        self.dismiss()


class ChoiceScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, title: str, choices: list[tuple[str, str, str]], current: str) -> None:
        super().__init__()
        self.title_text = title
        self.choices = choices
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.title_text)
            yield OptionList(id="choices")

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        for value, name, desc in self.choices:
            mark = "● " if value == self.current else "  "
            ol.add_option(Option(f"{mark}{name}  [dim]{desc}[/]", id=value))
        idx = next((i for i, c in enumerate(self.choices) if c[0] == self.current), 0)
        ol.highlighted = idx
        ol.focus()

    @on(OptionList.OptionSelected)
    def chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))


class WorkspaceList(ListView):
    BINDINGS = [Binding("escape", "leave", "back")]

    def action_leave(self) -> None:
        cast("MashApp", self.app).leave_workspaces()


class PathScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Workspace directory")
            yield Input(value=str(Path.cwd()), id="path")

    @on(Input.Submitted)
    def done(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)


class AgentPane(Vertical):
    def __init__(self, workspace: Workspace, app: MashApp) -> None:
        super().__init__()
        self.workspace = workspace
        self.mash = app
        self.agent = Agent(workspace, app.tools.url, self.on_update, app.ask_permission)
        self.started = False
        self.state_text = "not started"

    def compose(self) -> ComposeResult:
        yield ChatLog()
        yield Static("agent: not started", id="agent-status")
        yield PromptArea(id="agent-input")

    @property
    def chat(self) -> ChatLog:
        return self.query_one(ChatLog)

    def status(self, text: str) -> None:
        self.state_text = text
        setup = ""
        if self.agent.config_options:
            setup = f"  {self.agent.option_label('model')} · effort {self.agent.option_label('effort')}"
        self.query_one("#agent-status", Static).update(f"agent: {text}{setup}")
        self.mash.refresh_workspace_labels()

    async def choose_option(self, config_id: str) -> None:
        await self.ensure_started()
        opt = self.agent.option(config_id)
        if opt is None:
            self.mash.notify(f"agent has no {config_id} option", severity="warning")
            return
        choices = [(c.value, c.name, c.description or "") for c in opt.options]
        value = await self.mash.pick(ChoiceScreen(opt.name, choices, str(opt.current_value)))
        if value is None:
            return
        try:
            await self.agent.set_option(config_id, value)
        except Exception as e:
            await self.chat.note(f"could not set {config_id}: {e}", "red")
            return
        await self.chat.note(f"{opt.name} → {self.agent.option_label(config_id)}")
        self.status(self.state_text)

    async def ensure_started(self) -> None:
        if self.started:
            return
        self.started = True
        self.workspace.agent = self.agent
        self.status("starting")
        try:
            await self.agent.start()
        except Exception as e:
            self.status(f"failed: {e}")
            self.started = False
            return
        self.status("idle")

    def on_update(self, update: Any) -> None:
        self.mash.call_later(self._render_update, update)

    async def _render_update(self, update: Any) -> None:
        if isinstance(update, AgentMessageChunk) and isinstance(update.content, TextContentBlock):
            await self.chat.agent_chunk(update.content.text)
        elif isinstance(update, AgentThoughtChunk) and isinstance(update.content, TextContentBlock):
            self.status("thinking")
        elif isinstance(update, ToolCallStart):
            await self.chat.note(f"⚙ {update.title}")
            self.status(f"tool: {update.title}")
        elif isinstance(update, ToolCallProgress) and update.status in ("completed", "failed"):
            self.status("working")
        elif isinstance(update, AgentPlanUpdate):
            steps = "\n".join(f"  [{e.status}] {e.content}" for e in update.entries)
            await self.chat.note(f"plan:\n{steps}")
        elif isinstance(update, ConfigOptionUpdate):
            self.status(self.state_text)

    def on_key(self, event: events.Key) -> None:
        if self.app.focused is self.query_one("#agent-input", PromptArea) and page_scroll(
            self.chat, event.key
        ):
            event.stop()
            event.prevent_default()

    @on(PromptArea.Submitted)
    def submit(self, event: PromptArea.Submitted) -> None:
        text = event.text.strip()
        self.query_one("#agent-input", PromptArea).clear()
        if not text:
            return
        if text in ("/model", "/effort", "/mode"):
            self.run_worker(self.choose_option(text[1:]), group="agent-option")
            return
        self.run_worker(self._turn(text), exclusive=True, group="agent")

    async def _turn(self, text: str) -> None:
        await self.ensure_started()
        if not self.started:
            return
        await self.chat.user(text)
        self.status("working")
        try:
            reason = await self.agent.prompt(text)
            self.status(f"idle ({reason})")
        except Exception as e:
            await self.chat.note(f"error: {e}", "red")
            self.status("error")


class WorkspaceView(Vertical):
    def __init__(self, workspace: Workspace, app: MashApp) -> None:
        super().__init__(id=f"ws-{wid(workspace.name)}")
        self.workspace = workspace
        self.mash = app
        self._mount_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="session-area"):
                yield Tabs(id="tabs")
                yield ContentSwitcher(id="session-switcher")
            yield AgentPane(self.workspace, self.mash)

    @property
    def tabs(self) -> Tabs:
        return self.query_one("#tabs", Tabs)

    @property
    def switcher(self) -> ContentSwitcher:
        return self.query_one("#session-switcher", ContentSwitcher)

    @property
    def agent_pane(self) -> AgentPane:
        return self.query_one(AgentPane)

    def current_view(self) -> SessionView | None:
        cur = self.switcher.current
        return self.switcher.query_one(f"#{cur}", SessionView) if cur else None

    async def add_session(self, session: Session) -> SessionView:
        # session events and start_session can both try to add the view; mount it once
        async with self._mount_lock:
            existing = self.view_for(session)
            if existing is not None:
                return existing
            view = SessionView(session, lambda kid, code: self.mash.run_code(session, kid, code))
            await self.switcher.mount(view)
            self.tabs.add_tab(Tab(self.tab_label(session), id=f"tab-{wid(session.name)}"))
            self.tabs.active = f"tab-{wid(session.name)}"
            return view

    def view_for(self, session: Session) -> SessionView | None:
        try:
            return self.switcher.query_one(f"#session-{wid(session.name)}", SessionView)
        except Exception:
            return None

    @staticmethod
    def tab_label(session: Session) -> str:
        mark = STATE_MARK[session.state] if session.kernels else "∅"
        who = "you" if session.owner == "human" else session.owner or ""
        return f"{session.name} {mark}{who}".strip()

    def refresh_tab(self, session: Session) -> None:
        with contextlib.suppress(Exception):
            self.tabs.query_one(f"#tab-{wid(session.name)}", Tab).label = self.tab_label(session)

    @on(Tabs.TabActivated)
    def switch(self, event: Tabs.TabActivated) -> None:
        if event.tab is None:
            return
        self.switcher.current = event.tab.id.replace("tab-", "session-", 1)
        view = self.current_view()
        if view:
            view.query_one("#code-input", CodeInput).focus()


class MashApp(App[None]):
    TITLE = "mash"
    CSS_PATH = "mash.tcss"
    BINDINGS = [
        Binding("ctrl+k", "kernels", "Kernels", priority=True),
        Binding("ctrl+n", "new_session", "New session", priority=True),
        Binding("ctrl+t", "new_workspace", "New workspace", priority=True),
        Binding("ctrl+a", "toggle_agent", "Agent", priority=True),
        Binding("ctrl+r", "toggle_repl", "REPL", priority=True),
        Binding("ctrl+o", "open_desktop", "Desktop", priority=True),
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+x", "cancel_agent", "Stop agent", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+b", "prefix", "Prefix", priority=True),
        Binding("f1", "keys", "Keys", priority=True),
        Binding("ctrl+w", "focus_workspaces", "Workspaces", priority=True),
    ]

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.manager = Manager()
        self.tools = ToolServer(self.manager)
        self._prefix_focus: Widget | None = None
        self._workspace_return: Widget | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield WorkspaceList(id="workspaces")
            yield ContentSwitcher(id="ws-switcher")
        yield Footer()

    async def on_mount(self) -> None:
        self.tools.installer = self.install_language
        await self.tools.start()
        await self.add_workspace(self.root)

    @property
    def ws_switcher(self) -> ContentSwitcher:
        return self.query_one("#ws-switcher", ContentSwitcher)

    def current_workspace_view(self) -> WorkspaceView | None:
        cur = self.ws_switcher.current
        return self.ws_switcher.query_one(f"#{cur}", WorkspaceView) if cur else None

    def workspace_view(self, ws: Workspace) -> WorkspaceView:
        return self.ws_switcher.query_one(f"#ws-{wid(ws.name)}", WorkspaceView)

    async def add_workspace(self, path: Path) -> WorkspaceView:
        ws = self.manager.new_workspace(path)
        ws.subscribe(self.on_session_event)
        view = WorkspaceView(ws, self)
        await self.ws_switcher.mount(view)
        self.ws_switcher.current = view.id
        lv = self.query_one("#workspaces", ListView)
        await lv.append(ListItem(Label(ws.name), id=f"wsitem-{wid(ws.name)}"))
        lv.index = len(lv.children) - 1
        await self.start_session(ws)
        return view

    def refresh_workspace_labels(self) -> None:
        for ws in self.manager.workspaces.values():
            try:
                view = self.workspace_view(ws)
                flag = ""
                s = view.agent_pane.state_text
                if "working" in s or "tool" in s or "thinking" in s:
                    flag = " ●"
                elif "starting" in s:
                    flag = " …"
                self.query_one(f"#wsitem-{wid(ws.name)}", ListItem).query_one(Label).update(
                    f"{ws.name}{flag}"
                )
            except Exception:
                continue

    async def start_session(self, ws: Workspace, name: str | None = None) -> Session:
        session = ws.new_session(name)
        view = await self.workspace_view(ws).add_session(session)
        # worker: add_kernel may wait on the install dialog, which on_mount can't await
        self.run_worker(self.add_kernel(view, default_language()), group="kernels")
        return session

    def on_session_event(self, ev: SessionEvent) -> None:
        self.call_later(self._handle_session_event, ev)

    async def _handle_session_event(self, ev: SessionEvent) -> None:
        ws = next((w for w in self.manager.workspaces.values() if ev.session in w.sessions.values()), None)
        if ws is None:
            return
        wview = self.workspace_view(ws)
        view = wview.view_for(ev.session)
        if view is None:
            if ev.kind != "log":
                view = await wview.add_session(ev.session)
            else:
                return
        k = ev.kernel
        if ev.kind == "state":
            wview.refresh_tab(ev.session)
            view.refresh_prompt()
            if k and k.state == "dead" and k.error and ev.session.kernels.get(k.id) is k:
                view.hint(f"{k.id} died: {k.error[:200]} · ctrl+k to restart")
        elif ev.kind == "ready" and k:
            view.hint(f"{k.id} ready ({k.backend.title})")
        elif ev.kind == "cell_started" and ev.cell and k:
            await view.transcript.add_cell(ev.cell, k.backend.lexer)
        elif ev.kind == "output" and ev.cell and ev.output:
            w = view.transcript.get(ev.cell)
            if w:
                await w.show_output(ev.output)
                view.transcript.scroll_end(animate=False)
        elif ev.kind == "cell_finished" and ev.cell:
            w = view.transcript.get(ev.cell)
            if w:
                w.refresh_header()
        elif ev.kind == "log" and k and k.state == "starting":
            view.hint(f"{k.id}: {ev.text[-120:]}")

    async def run_code(self, session: Session, kid: str | None, code: str) -> None:
        kernel = session.kernels.get(kid) if kid else None
        if kernel is None:
            self.notify("no kernel in this session; ctrl+k to add one", severity="warning")
        elif kernel.state == "starting":
            self.notify(f"{kernel.id} is still starting", severity="warning")
        elif kernel.state == "dead":
            self.notify(f"{kernel.id} is dead; ctrl+k to restart it", severity="error")
        else:
            self.run_worker(self._execute(kernel, code), group=f"exec-{session.name}-{kernel.id}")

    async def _execute(self, kernel: KernelHandle, code: str) -> None:
        try:
            await kernel.execute(code, "human")
        except Exception as e:
            self.notify(f"{kernel.id}: {e}", severity="error")

    def action_kernels(self) -> None:
        view = self.current_session_view()
        if view is not None:
            self.run_worker(self.pick_kernel(view), group="kernels")

    async def pick_kernel(self, view: SessionView) -> None:
        session = view.session
        choices: list[tuple[str, str, str]] = []
        for k in session.kernels.values():
            if k.state == "dead":
                choices.append((f"restart:{k.id}", f"restart {k.id}", (k.error or "dead")[:60]))
            else:
                choices.append((f"use:{k.id}", k.id, f"{k.backend.title} · {k.state}"))
        for b, ok, detail in await asyncio.to_thread(language_status, session.cwd):
            choices.append((f"add:{b.name}", f"add {b.title}", detail if ok else f"{b.missing} · {detail}"))
        value = await self.pick(ChoiceScreen(f"Kernels in {session.name}", choices, f"use:{view.active}"))
        action, _, arg = (value or "").partition(":")
        try:
            if action == "use":
                view.set_active(arg)
            elif action == "restart":
                view.set_active((await session.restart_kernel(arg)).id)
            elif action == "add":
                await self.add_kernel(view, arg)
        except Exception as e:
            self.notify(f"{action} {arg}: {e}", severity="error")
        view.query_one("#code-input", CodeInput).focus()

    async def add_kernel(self, view: SessionView, language: str) -> None:
        backend = get_backend(language)
        ok, _ = await asyncio.to_thread(backend.status, view.session.cwd)
        if not ok:
            err = await self.install_language(backend.name, "human")
            if err:
                self.notify(err, severity="warning")
                view.hint(f"no {backend.title} kernel · ctrl+k to add one")
                return
        view.set_active(view.session.add_kernel(backend.name).id)

    async def install_language(self, language: str, requested_by: str) -> str | None:
        backend = get_backend(language)
        if not backend.installable:
            return f"mash can't install {backend.title}: {backend.install_summary()}"
        action = backend.setup_action()
        title = (
            f"The agent wants to {action}?"
            if requested_by == "agent"
            else f"{action[:1].upper()}{action[1:]}?"
        )
        choices = [("yes", action.split()[0].capitalize(), backend.install_summary()), ("no", "Cancel", "")]
        if await self.pick(ChoiceScreen(title, choices, "no")) != "yes":
            return f"the human declined to {action}"
        self.notify(f"setting up {backend.title} …", timeout=10)

        def log(line: str) -> None:
            self.call_from_thread(self.install_progress, backend.title, line)

        try:
            await asyncio.to_thread(backend.install, log)
        except Exception as e:
            self.notify(f"{backend.title} setup failed: {e}", severity="error", timeout=30)
            return f"setup failed: {e}"
        self.notify(f"{backend.title} is set up")
        return None

    def install_progress(self, title: str, line: str) -> None:
        view = self.current_session_view()
        if view is not None and line.strip():
            view.hint(f"[{title} setup] {line.strip()[-160:]}")

    async def ask_permission(self, title: str, options: list[PermissionOption]) -> str | None:
        fut: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self.push_screen(PermissionScreen(title, options), callback=fut.set_result)
        self.bell()
        return await fut

    def current_session_view(self) -> SessionView | None:
        wview = self.current_workspace_view()
        return wview.current_view() if wview else None

    async def action_new_session(self) -> None:
        wview = self.current_workspace_view()
        if wview:
            await self.start_session(wview.workspace)

    def action_new_workspace(self) -> None:
        def made(path: str | None) -> None:
            if path:
                self.run_worker(self.add_workspace(Path(path)))

        self.push_screen(PathScreen(), callback=made)

    async def action_toggle_agent(self) -> None:
        wview = self.current_workspace_view()
        if not wview:
            return
        pane = wview.agent_pane
        pane.toggle_class("visible")
        if pane.has_class("visible"):
            pane.query_one("#agent-input", PromptArea).focus()
            self.run_worker(pane.ensure_started(), group="agent-start")
        else:
            view = wview.current_view()
            if view:
                view.query_one("#code-input", CodeInput).focus()

    async def action_toggle_repl(self) -> None:
        view = self.current_session_view()
        if view:
            await view.toggle_repl()

    def action_open_desktop(self) -> None:
        view = self.current_session_view()
        kernel = matlab_kernel(view.session) if view else None
        if kernel and kernel.desktop_url:
            webbrowser.open(kernel.desktop_url)
            self.notify("opened MATLAB desktop in browser")
        else:
            self.notify("no MATLAB desktop URL in this session yet", severity="warning")

    async def action_interrupt(self) -> None:
        view = self.current_session_view()
        if view is None:
            return
        busy = [k for k in view.session.kernels.values() if k.state == "busy"]
        active = view.active_kernel()
        target = active if active in busy else (busy[0] if len(busy) == 1 else None)
        if target is not None:
            await target.interrupt()
            self.notify(f"interrupted {target.id} (was running: {target.owner})")
        elif busy:
            names = ", ".join(k.id for k in busy)
            self.notify(f"active kernel is idle; busy: {names} · ctrl+k to switch", severity="warning")

    async def action_cancel_agent(self) -> None:
        wview = self.current_workspace_view()
        if wview:
            await wview.agent_pane.agent.cancel()

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield SystemCommand("Keys", "List every key binding", self.action_keys)
        yield SystemCommand(
            "Model", "Choose the agent's model", lambda: self.run_worker(self.agent_option("model"))
        )
        yield SystemCommand(
            "Effort", "Choose the agent's effort level", lambda: self.run_worker(self.agent_option("effort"))
        )
        yield SystemCommand(
            "Permission mode",
            "Choose how tool calls are approved",
            lambda: self.run_worker(self.agent_option("mode")),
        )
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        yield SystemCommand("Quit", "Snapshot sessions and quit", self.action_quit)

    def action_keys(self) -> None:
        self.push_screen(KeysScreen())

    async def agent_option(self, config_id: str) -> None:
        wview = self.current_workspace_view()
        if wview is None:
            return
        pane = wview.agent_pane
        if not pane.has_class("visible"):
            await self.action_toggle_agent()
        await pane.choose_option(config_id)

    async def pick(self, screen: ModalScreen[str | None]) -> str | None:
        fut: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self.push_screen(screen, callback=fut.set_result)
        return await fut

    def action_prefix(self) -> None:
        self._prefix_focus = self.focused
        self.push_screen(PrefixScreen())

    def prefix_move(self, key: str) -> None:
        wview = self.current_workspace_view()
        if wview is None:
            return
        view = wview.current_view()
        if view is None:
            return
        cur = self._prefix_focus
        pane = wview.agent_pane
        center: list[Widget] = [view.transcript, view.query_one("#code-input", CodeInput)]
        if view.repl.has_class("visible"):
            center.append(view.repl.query_one("#repl-input", CodeInput))
        agent: list[Widget] = []
        if pane.has_class("visible"):
            agent = [pane.chat, pane.query_one("#agent-input", PromptArea)]
        column = agent if cur in agent else center
        target: Widget | None = None
        if key in ("j", "down", "k", "up"):
            idx = column.index(cur) if cur in column else len(column) - 1
            step = 1 if key in ("j", "down") else -1
            target = column[max(0, min(len(column) - 1, idx + step))]
        elif key in ("l", "right"):
            if cur is self.query_one("#workspaces", ListView):
                target = center[1]
            elif cur not in agent and agent:
                target = agent[-1]
        elif key in ("h", "left") and cur in agent:
            target = center[1]
        if target is not None:
            target.focus()

    def prefix_tab(self, key: str) -> None:
        wview = self.current_workspace_view()
        if wview is None:
            return
        if key == "n":
            wview.tabs.action_next_tab()
        elif key == "p":
            wview.tabs.action_previous_tab()
        else:
            tabs = list(wview.tabs.query(Tab))
            if int(key) <= len(tabs):
                wview.tabs.active = tabs[int(key) - 1].id or ""

    def action_focus_workspaces(self) -> None:
        lv = self.query_one("#workspaces", ListView)
        if self.focused is not lv:
            self._workspace_return = self.focused
        lv.focus()

    def leave_workspaces(self) -> None:
        target = self._workspace_return
        self._workspace_return = None
        if target is not None and target.is_attached:
            target.focus()
        else:
            self.focus_current_input()

    def focus_current_input(self) -> None:
        view = self.current_session_view()
        if view:
            view.query_one("#code-input", CodeInput).focus()

    @on(ListView.Selected, "#workspaces")
    def select_workspace(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.ws_switcher.current = event.item.id.replace("wsitem-", "ws-", 1)
            self._workspace_return = None
            self.focus_current_input()

    async def action_quit(self) -> None:
        self.notify("saving sessions and shutting down kernels …")
        for view in self.query(SessionView):
            view.repl.close()
        for ws in self.manager.workspaces.values():
            for sess in ws.sessions.values():
                for kernel in sess.kernels.values():
                    if kernel.state == "idle":
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(kernel.snapshot(), 20)
        await self.manager.close()
        await self.tools.stop()
        self.exit()
