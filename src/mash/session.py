from __future__ import annotations

import asyncio
import contextlib
import os
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Literal

import psutil
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

from mash.backends import Backend, get_backend
from mash.htmltable import parse_html_table, table_text
from mash.notebook import NotebookRecorder

Author = Literal["human", "agent"]
State = Literal["starting", "idle", "busy", "dead"]
IMAGE_MIMES = ("image/png", "image/jpeg", "image/gif")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_seq = count(1)


def error_text(data: dict[str, Any]) -> str:
    head = f"{data['ename']}: {data['evalue']}"
    tb = ANSI.sub("", "\n".join(data.get("traceback") or [])).strip()
    if not tb:
        return head
    return tb if data["evalue"] in tb else f"{head}\n{tb}"


@dataclass
class Output:
    kind: Literal["stream", "data", "error"]
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    display_id: str | None = None
    seq: int = field(default_factory=lambda: next(_seq))

    def text(self) -> str:
        if self.kind == "stream":
            return self.data["text"]
        if self.kind == "error":
            return error_text(self.data)
        if "text/plain" in self.data:
            plain = self.data["text/plain"]
            table = parse_html_table(plain) if plain.lstrip().startswith("<html") else None
            return table_text(table) if table else plain
        for mime in ("text/markdown", "text/html", "text/latex"):
            if isinstance(self.data.get(mime), str):
                return self.data[mime]
        for mime, meta in self.metadata.items():
            if mime.startswith("image/"):
                return f"[figure {mime} {meta.get('width', '?')}x{meta.get('height', '?')}]"
        return ""


@dataclass
class Cell:
    index: int
    author: Author
    code: str
    kernel: str = ""
    outputs: list[Output] = field(default_factory=list)
    status: Literal["running", "ok", "error"] = "running"
    started: datetime = field(default_factory=datetime.now)
    finished: datetime | None = None

    def text(self) -> str:
        return "".join(o.text() for o in self.outputs if o.kind != "data" or o.data)


@dataclass
class SessionEvent:
    kind: Literal["state", "ready", "cell_started", "output", "cell_finished", "log"]
    session: Session
    kernel: KernelHandle | None = None
    cell: Cell | None = None
    output: Output | None = None
    text: str = ""


Listener = Callable[[SessionEvent], None]


class KernelDied(RuntimeError):
    pass


class OneSpec(KernelSpecManager):
    def __init__(self, spec: KernelSpec) -> None:
        super().__init__()
        self.spec = spec

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        return self.spec


class KernelHandle:
    def __init__(self, session: Session, kid: str, backend: Backend) -> None:
        self.session = session
        self.id = kid
        self.backend = backend
        self.share_name = re.sub(r"\W", "_", f"mash_{os.getpid()}_{session.name}_{kid}")
        self.state: State = "starting"
        self.owner: Author | None = None
        self.desktop_url: str | None = None
        self.language_info: dict[str, Any] = {}
        self.logs: list[str] = []
        self.ready = asyncio.Event()
        self.error: str | None = None
        self.recorder: NotebookRecorder | None = None
        self._km: AsyncKernelManager | None = None
        self._kc: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        self._pending: dict[str, Cell] = {}
        self._idle: dict[str, asyncio.Future[None]] = {}
        self._replies: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tasks: list[asyncio.Task[None]] = []

    def _emit(self, kind: Any, **kw: Any) -> None:
        self.session._emit(kind, self, **kw)

    def _emit_cell(self, kind: Any, cell: Cell, **kw: Any) -> None:
        if cell.index:
            self._emit(kind, cell=cell, **kw)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            argv, env = await asyncio.to_thread(self.backend.launch, self.session.cwd, self.share_name)
            spec = KernelSpec(
                argv=argv,
                display_name=self.backend.title,
                language=self.backend.name,
                interrupt_mode=self.backend.interrupt_mode,
            )
            self._km = AsyncKernelManager(kernel_name=self.backend.name, kernel_spec_manager=OneSpec(spec))
            await self._km.start_kernel(
                env=env,
                cwd=str(self.session.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            proc = getattr(self._km.provisioner, "process", None)
            if proc is not None and proc.stdout is not None:
                # read a dup: helpers such as MathWorksServiceHost inherit the pipe and outlive the kernel;
                # a readline blocked on proc.stdout would deadlock the provisioner's close() at shutdown
                pipe = os.dup(proc.stdout.fileno())
                threading.Thread(target=self._pump_logs, args=(pipe,), daemon=True).start()
            self._kc = self._km.client()
            self._kc.start_channels()
            await self._kc.wait_for_ready(timeout=self.backend.start_timeout)
            self._tasks = [
                asyncio.create_task(self._guard("iopub", self._read_iopub())),
                asyncio.create_task(self._guard("shell", self._read_shell())),
                asyncio.create_task(self._guard("watch", self._watch())),
            ]
            info = await self._request(self._kc.kernel_info(), 60)
            self.language_info = info["content"].get("language_info", {})
            path = self.session.record_dir / f"{self.id}.ipynb"
            self.recorder = NotebookRecorder(path, self.backend, self.language_info)
            code = self.backend.startup_code()
            if code:
                cell = await self._run(code, "human", record=False)
                if cell.status == "error":
                    raise RuntimeError(cell.text())
        except Exception as e:
            self.error = str(e) or type(e).__name__
            self._set_state("dead")
            self.ready.set()
            return
        self._set_state("idle")
        self._emit("ready")
        self.ready.set()

    def _pump_logs(self, pipe: int) -> None:
        prev = ""
        with os.fdopen(pipe, "rb") as out:
            for raw in iter(out.readline, b""):
                line = raw.decode(errors="replace").rstrip()
                self.desktop_url = self.backend.desktop_url(prev, line) or self.desktop_url
                prev = line
                self.logs.append(line)
                del self.logs[:-500]
                if self._loop:
                    with contextlib.suppress(RuntimeError):  # loop closed after shutdown
                        self._loop.call_soon_threadsafe(lambda ln=line: self._emit("log", text=ln))

    def _set_state(self, state: State, owner: Author | None = None) -> None:
        self.state = state
        self.owner = owner
        self._emit("state")

    def _die(self, reason: str) -> None:
        if self.state == "dead":
            return
        self.error = reason
        for futs in (self._idle, self._replies):
            for fut in futs.values():
                if not fut.done():
                    fut.set_exception(KernelDied(reason))
                    # mark retrieved: _run stops at the first failed future and never awaits the other
                    fut.exception()
        self._set_state("dead")

    async def execute(self, code: str, author: Author) -> Cell:
        await self.ready.wait()
        async with self._lock:
            if self.state == "dead":
                raise KernelDied(f"kernel {self.id} is dead: {self.error}")
            return await self._run(code, author)

    async def _run(self, code: str, author: Author, record: bool = True) -> Cell:
        loop = asyncio.get_running_loop()
        cells = self.session.cells
        cell = Cell(index=len(cells) + 1 if record else 0, author=author, code=code, kernel=self.id)
        if record:
            cells.append(cell)
        self._set_state("busy", author)
        self._emit_cell("cell_started", cell)
        msg_id = self._kc.execute(code, store_history=record)
        idle: asyncio.Future[None] = loop.create_future()
        reply_fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[msg_id] = cell
        self._idle[msg_id] = idle
        self._replies[msg_id] = reply_fut
        try:
            await idle
            reply = await reply_fut
            failed = reply["content"]["status"] != "ok" or self.backend.failed(cell.outputs)
        except KernelDied as e:
            out = Output("error", {"ename": "KernelDied", "evalue": str(e), "traceback": []})
            cell.outputs.append(out)
            self._emit_cell("output", cell, output=out)
            failed = True
        finally:
            self._pending.pop(msg_id, None)
            self._idle.pop(msg_id, None)
            self._replies.pop(msg_id, None)
        cell.status = "error" if failed else "ok"
        cell.finished = datetime.now()
        if self.state != "dead":
            self._set_state("idle")
        self._emit_cell("cell_finished", cell)
        if record and self.recorder:
            self.recorder.append(cell)
        return cell

    async def _guard(self, name: str, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logs.append(f"mash: {name} reader died: {e!r}")
            self._die(f"{name} reader died: {e!r}")

    async def _watch(self) -> None:
        assert self._km is not None
        while await self._km.is_alive():
            await asyncio.sleep(1)
        self._die("kernel process exited")

    async def _read_iopub(self) -> None:
        while True:
            msg = await self._kc.get_iopub_msg()
            pid = msg["parent_header"].get("msg_id")
            cell = self._pending.get(pid)
            if cell is None:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "status" and c["execution_state"] == "idle":
                self._pending.pop(pid)
                fut = self._idle.pop(pid)
                if not fut.done():
                    fut.set_result(None)
            elif t == "stream":
                out = Output("stream", {"name": c["name"], "text": c["text"]})
                cell.outputs.append(out)
                self._emit_cell("output", cell, output=out)
            elif t in ("display_data", "execute_result"):
                did = c.get("transient", {}).get("display_id")
                out = Output("data", c["data"], c.get("metadata", {}), did)
                cell.outputs.append(out)
                self._emit_cell("output", cell, output=out)
            elif t == "update_display_data":
                did = c.get("transient", {}).get("display_id")
                for out in cell.outputs:
                    if out.display_id == did:
                        out.data = c["data"]
                        out.metadata = c.get("metadata", {})
                        self._emit_cell("output", cell, output=out)
            elif t == "error":
                out = Output("error", {k: c.get(k, []) for k in ("ename", "evalue", "traceback")})
                cell.outputs.append(out)
                self._emit_cell("output", cell, output=out)

    async def _read_shell(self) -> None:
        while True:
            msg = await self._kc.get_shell_msg()
            pid = msg["parent_header"].get("msg_id")
            if pid not in self._replies:
                self.logs.append(f"mash: unmatched shell {msg['msg_type']} parent={pid}")
            fut = self._replies.get(pid)
            if fut is not None and not fut.done():
                fut.set_result(msg)

    async def _request(self, msg_id: str, timeout: float) -> dict[str, Any]:
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._replies[msg_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._replies.pop(msg_id, None)

    async def complete(self, code: str, cursor: int) -> tuple[list[str], int, int]:
        if self.state != "idle":
            return [], cursor, cursor
        try:
            reply = await self._request(self._kc.complete(code, cursor), 10)
        except (TimeoutError, KernelDied):
            return [], cursor, cursor
        c = reply["content"]
        return c.get("matches", []), c.get("cursor_start", cursor), c.get("cursor_end", cursor)

    async def interrupt(self) -> None:
        if self.state == "busy" and self._km:
            await self._km.interrupt_kernel()

    async def variables(self) -> str:
        await self.ready.wait()
        async with self._lock:
            if self.state == "dead":
                raise KernelDied(f"kernel {self.id} is dead: {self.error}")
            cell = await self._run(self.backend.variables_code, "agent", record=False)
        return cell.text()

    async def snapshot(self) -> Cell | None:
        code = self.backend.snapshot_code(self.session.record_dir / self.id)
        return await self.execute(code, "human") if code else None

    async def _graceful_stop(self) -> None:
        assert self._km is not None
        await self._km.shutdown_kernel()

    async def shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._kc:
            self._kc.stop_channels()
        km = self._km
        if km is None:
            self._set_state("dead")
            return
        seen: dict[int, psutil.Process] = {}
        proc = getattr(km.provisioner, "process", None)

        def track() -> None:
            if proc is None:
                return
            with contextlib.suppress(psutil.Error):
                for child in psutil.Process(proc.pid).children(recursive=True):
                    seen.setdefault(child.pid, child)

        track()
        if km.has_kernel:
            stop = asyncio.create_task(self._graceful_stop())
            deadline = asyncio.get_running_loop().time() + 30
            while not stop.done() and asyncio.get_running_loop().time() < deadline:
                track()
                await asyncio.sleep(0.3)
            if not stop.done():
                stop.cancel()
                with contextlib.suppress(Exception):
                    await km.shutdown_kernel(now=True)
        alive = [d for d in seen.values() if d.is_running()]
        for d in alive:
            with contextlib.suppress(psutil.Error):
                d.terminate()
        _, still = psutil.wait_procs(alive, timeout=10)
        for d in still:
            with contextlib.suppress(psutil.Error):
                d.kill()
        self._set_state("dead")


class Session:
    def __init__(self, name: str, cwd: Path, record_dir: Path, created_by: Author = "human") -> None:
        self.name = name
        self.cwd = cwd
        self.record_dir = record_dir / name
        self.created_by = created_by
        self.kernels: dict[str, KernelHandle] = {}
        self.cells: list[Cell] = []
        self.listeners: list[Listener] = []
        self._tasks: list[asyncio.Task[None]] = []

    def subscribe(self, fn: Listener) -> None:
        self.listeners.append(fn)

    def _emit(self, kind: Any, kernel: KernelHandle | None = None, **kw: Any) -> None:
        ev = SessionEvent(kind, self, kernel, **kw)
        for fn in list(self.listeners):
            fn(ev)

    @property
    def state(self) -> State:
        states = {k.state for k in self.kernels.values()}
        for s in ("busy", "starting", "idle"):
            if s in states:
                return s
        return "dead" if states else "idle"

    @property
    def owner(self) -> Author | None:
        return next((k.owner for k in self.kernels.values() if k.state == "busy"), None)

    def add_kernel(self, language: str, kid: str | None = None) -> KernelHandle:
        backend = get_backend(language)
        if kid is None:
            kid, n = backend.alias, 1
            while kid in self.kernels:
                n += 1
                kid = f"{backend.alias}{n}"
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", kid):
            raise ValueError(
                f"kernel name {kid!r} must start with a letter and use only letters, digits, _ or -"
            )
        if kid in self.kernels:
            raise ValueError(f"session {self.name} already has a kernel named {kid}")
        return self._launch(KernelHandle(self, kid, backend))

    def _launch(self, handle: KernelHandle) -> KernelHandle:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.kernels[handle.id] = handle
        self._emit("state", handle)
        self._tasks.append(asyncio.create_task(handle.start()))
        return handle

    async def restart_kernel(self, kid: str) -> KernelHandle:
        old = self.kernel(kid)
        await old.shutdown()
        return self._launch(KernelHandle(self, old.id, old.backend))

    def kernel(self, kid: str | None = None) -> KernelHandle:
        if kid is None:
            if len(self.kernels) == 1:
                return next(iter(self.kernels.values()))
            if not self.kernels:
                raise ValueError(f"session {self.name} has no kernels; add one with add_kernel")
            raise ValueError(f"session {self.name} has kernels {', '.join(self.kernels)}; pass kernel=")
        if kid in self.kernels:
            return self.kernels[kid]
        with contextlib.suppress(ValueError):
            lang = get_backend(kid).name
            matches = [k for k in self.kernels.values() if k.backend.name == lang]
            if len(matches) == 1:
                return matches[0]
        raise KeyError(f"no kernel {kid} in session {self.name}; have {', '.join(self.kernels) or 'none'}")

    async def execute(self, code: str, author: Author, kernel: str | None = None) -> Cell:
        return await self.kernel(kernel).execute(code, author)

    def history(self, n: int = 10, kernel: str | None = None) -> list[Cell]:
        cells = (
            self.cells if kernel is None else [c for c in self.cells if c.kernel == self.kernel(kernel).id]
        )
        return cells[-n:]

    async def shutdown(self) -> None:
        await asyncio.gather(*(k.shutdown() for k in self.kernels.values()))
