from __future__ import annotations

import base64
import io
from collections.abc import Callable
from typing import Any

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import events, on
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, RichLog, Static, TextArea
from textual_image.widget import Image

from mash.engine import EngineRepl
from mash.htmltable import MAX_ROWS, parse_html_table
from mash.session import Cell, MatlabSession, Output


def output_widget(out: Output) -> Widget:
    if out.kind == "stream":
        w = Static(Text(out.data["text"].rstrip("\n")))
        if out.data["name"] == "stderr":
            w.add_class("stderr")
        return w
    if out.kind == "error":
        return Static(Text(f"{out.data['ename']}: {out.data['evalue']}"), classes="stderr")
    for mime, data in out.data.items():
        if mime.startswith("image/") and isinstance(data, str):
            return Image(io.BytesIO(base64.b64decode(data)), classes="figure")
    if "text/plain" in out.data:
        plain = out.data["text/plain"]
        table = parse_html_table(plain) if plain.lstrip().startswith("<html") else None
        if table:
            return Static(rich_table(table))
        return Static(Text(plain.rstrip("\n")))
    return Static("")


def rich_table(table: Any) -> Table:
    t = Table(
        title=table.caption or None, title_justify="left", show_lines=False, pad_edge=False, expand=False
    )
    for h in table.header or [""] * (len(table.rows[0]) if table.rows else 0):
        t.add_column(h, justify="right")
    for r in table.rows[:MAX_ROWS]:
        t.add_row(*r)
    if len(table.rows) > MAX_ROWS:
        t.caption = f"… {len(table.rows) - MAX_ROWS} more rows"
        t.caption_justify = "left"
    return t


class CellWidget(Vertical):
    def __init__(self, cell: Cell) -> None:
        super().__init__(classes=f"cell author-{cell.author}")
        self.cell = cell
        self._outputs: dict[int, Widget] = {}

    def compose(self):
        yield Static(self.header(), classes="cell-header")
        yield Static(Syntax(self.cell.code, "matlab", theme="ansi_dark", word_wrap=True), classes="code")
        yield Vertical(classes="outputs")

    def header(self) -> Text:
        tag = "you" if self.cell.author == "human" else self.cell.author
        style = {"running": "yellow", "ok": "green", "error": "red"}[self.cell.status]
        return Text.assemble(
            (f"In[{self.cell.index}] ", "bold"), (tag, style), (f" {self.cell.status}", style)
        )

    def refresh_header(self) -> None:
        self.query_one(".cell-header", Static).update(self.header())

    async def show_output(self, out: Output) -> None:
        box = self.query_one(".outputs", Vertical)
        w = output_widget(out)
        old = self._outputs.get(out.seq)
        if old is not None:
            await box.mount(w, after=old)
            await old.remove()
        else:
            await box.mount(w)
        self._outputs[out.seq] = w


class Transcript(VerticalScroll):
    def __init__(self) -> None:
        super().__init__()
        self.cells: dict[int, CellWidget] = {}

    async def add_cell(self, cell: Cell) -> CellWidget:
        w = CellWidget(cell)
        self.cells[cell.index] = w
        await self.mount(w)
        self.scroll_end(animate=False)
        return w

    def get(self, cell: Cell) -> CellWidget | None:
        return self.cells.get(cell.index)


class CodeInput(Input):
    class Complete(Message):
        def __init__(self, source: CodeInput, code: str, cursor: int) -> None:
            super().__init__()
            self.source = source
            self.code = code
            self.cursor = cursor

    BINDINGS = [
        ("up", "history_prev", "Prev"),
        ("down", "history_next", "Next"),
        ("tab", "complete", "Complete"),
    ]

    def __init__(self, placeholder: str = ">> MATLAB", **kw: Any) -> None:
        super().__init__(placeholder=placeholder, **kw)
        self.history: list[str] = []
        self.pos = 0

    def remember(self, code: str) -> None:
        if code and (not self.history or self.history[-1] != code):
            self.history.append(code)
        self.pos = len(self.history)

    def action_history_prev(self) -> None:
        if self.pos > 0:
            self.pos -= 1
            self.value = self.history[self.pos]
            self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        if self.pos < len(self.history) - 1:
            self.pos += 1
            self.value = self.history[self.pos]
        else:
            self.pos = len(self.history)
            self.value = ""
        self.cursor_position = len(self.value)

    def action_complete(self) -> None:
        self.post_message(self.Complete(self, self.value, self.cursor_position))

    def apply_completion(self, matches: list[str], start: int, end: int) -> str:
        if not matches:
            return ""
        prefix = matches[0]
        for m in matches[1:]:
            i = 0
            while i < min(len(prefix), len(m)) and prefix[i] == m[i]:
                i += 1
            prefix = prefix[:i]
        if prefix:
            self.value = self.value[:start] + prefix + self.value[end:]
            self.cursor_position = start + len(prefix)
        return "" if len(matches) == 1 else "  ".join(matches[:12]) + ("  …" if len(matches) > 12 else "")


class ReplPane(Vertical):
    def __init__(self, session: MatlabSession) -> None:
        super().__init__()
        self.session = session
        self.engine = EngineRepl(session.share_name)
        self.connecting = False
        self._partial = ""

    def compose(self):
        yield RichLog(wrap=True, markup=False, id="repl-log")
        yield CodeInput(placeholder="raw >> (MATLAB engine, streaming stdout)", id="repl-input")

    async def ensure_connected(self) -> None:
        if self.engine.eng is not None or self.connecting:
            return
        self.connecting = True
        log = self.query_one("#repl-log", RichLog)
        log.write(f"connecting to shared engine {self.session.share_name} …")
        ok = await self.engine.connect()
        log.write("connected" if ok else f"engine connect failed: {self.engine.error}")
        self.connecting = False

    def _write(self, text: str) -> None:
        log = self.query_one("#repl-log", RichLog)
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        for ln in lines:
            log.write(ln)

    @on(Input.Submitted, "#repl-input")
    async def run(self, event: Input.Submitted) -> None:
        code = event.value.strip()
        event.input.value = ""
        self.query_one("#repl-input", CodeInput).remember(code)
        if not code:
            return
        log = self.query_one("#repl-log", RichLog)
        log.write(Text(f">> {code}", style="bold"))
        await self.ensure_connected()
        await self.engine.eval(code, self._write)
        if self._partial:
            log.write(self._partial)
            self._partial = ""


def page_scroll(pane: Widget, key: str) -> bool:
    if key == "pageup":
        pane.scroll_page_up()
    elif key == "pagedown":
        pane.scroll_page_down()
    else:
        return False
    return True


class SessionView(Vertical):
    def __init__(self, session: MatlabSession, run: Callable[[str], Any]) -> None:
        super().__init__(id=f"session-{session.name}")
        self.session = session
        self.run_code = run

    def on_key(self, event: events.Key) -> None:
        focused = self.app.focused
        if focused is None or focused.id not in ("code-input", "repl-input"):
            return
        pane = self.repl.query_one("#repl-log", RichLog) if focused.id == "repl-input" else self.transcript
        if page_scroll(pane, event.key):
            event.stop()
            event.prevent_default()

    def compose(self):
        yield Transcript()
        yield Static("", id="hint")
        yield CodeInput(id="code-input")
        yield ReplPane(self.session)

    @property
    def transcript(self) -> Transcript:
        return self.query_one(Transcript)

    @property
    def repl(self) -> ReplPane:
        return self.query_one(ReplPane)

    def hint(self, text: str) -> None:
        self.query_one("#hint", Static).update(text)

    @on(Input.Submitted, "#code-input")
    async def submit(self, event: Input.Submitted) -> None:
        code = event.value.strip()
        inp = self.query_one("#code-input", CodeInput)
        inp.value = ""
        inp.remember(code)
        self.hint("")
        if code:
            await self.run_code(code)

    @on(CodeInput.Complete)
    async def complete(self, event: CodeInput.Complete) -> None:
        matches, start, end = await self.session.complete(event.code, event.cursor)
        self.hint(event.source.apply_completion(matches, start, end))

    async def toggle_repl(self) -> None:
        pane = self.repl
        pane.toggle_class("visible")
        if pane.has_class("visible"):
            pane.query_one("#repl-input", Input).focus()
            await pane.ensure_connected()
        else:
            self.query_one("#code-input", CodeInput).focus()


class PromptArea(TextArea):
    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class ChatLog(VerticalScroll):
    def __init__(self) -> None:
        super().__init__()
        self._current: Static | None = None
        self._buffer = ""

    async def user(self, text: str) -> None:
        self._current = None
        await self.mount(Static(Text(f"you: {text}"), classes="msg-user"))
        self.scroll_end(animate=False)

    async def agent_chunk(self, text: str) -> None:
        if self._current is None:
            self._buffer = ""
            self._current = Static("", classes="msg-agent")
            await self.mount(self._current)
        self._buffer += text
        self._current.update(Text(self._buffer))
        self.scroll_end(animate=False)

    async def note(self, text: str, style: str = "") -> None:
        self._current = None
        await self.mount(Static(Text(text, style=style), classes="msg-note"))
        self.scroll_end(animate=False)
