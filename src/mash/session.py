from __future__ import annotations

import asyncio
import contextlib
import os
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

from mash.config import KERNEL_NAME, MATLAB_START_TIMEOUT, kernel_env
from mash.notebook import NotebookRecorder

Author = Literal["human", "agent"]
State = Literal["starting", "idle", "busy", "dead"]
_seq = count(1)


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
            return f"{self.data['ename']}: {self.data['evalue']}"
        if "text/plain" in self.data:
            return self.data["text/plain"]
        for mime, meta in self.metadata.items():
            if mime.startswith("image/"):
                return f"[figure {mime} {meta.get('width', '?')}x{meta.get('height', '?')}]"
        return ""


@dataclass
class Cell:
    index: int
    author: Author
    code: str
    outputs: list[Output] = field(default_factory=list)
    status: Literal["running", "ok", "error"] = "running"
    started: datetime = field(default_factory=datetime.now)
    finished: datetime | None = None

    def text(self) -> str:
        return "".join(o.text() for o in self.outputs if o.kind != "data" or o.data)


@dataclass
class SessionEvent:
    kind: Literal["state", "cell_started", "output", "cell_finished", "log"]
    session: MatlabSession
    cell: Cell | None = None
    output: Output | None = None
    text: str = ""


Listener = Callable[[SessionEvent], None]


class MatlabSession:
    def __init__(self, name: str, cwd: Path, record_dir: Path, created_by: Author = "human") -> None:
        self.name = name
        self.cwd = cwd
        self.record_dir = record_dir / name
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.created_by = created_by
        self.share_name = f"mash_{os.getpid()}_{name}"
        self.state: State = "starting"
        self.owner: Author | None = None
        self.cells: list[Cell] = []
        self.desktop_url: str | None = None
        self.logs: list[str] = []
        self.ready = asyncio.Event()
        self.error: str | None = None
        self.listeners: list[Listener] = []
        self.recorder = NotebookRecorder(self.record_dir / f"{name}.ipynb")
        self._km = AsyncKernelManager(kernel_name=KERNEL_NAME)
        self._kc: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        self._pending: dict[str, Cell] = {}
        self._idle: dict[str, asyncio.Future[None]] = {}
        self._replies: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tasks: list[asyncio.Task[None]] = []

    def subscribe(self, fn: Listener) -> None:
        self.listeners.append(fn)

    def _emit(self, kind: Any, **kw: Any) -> None:
        ev = SessionEvent(kind, self, **kw)
        for fn in list(self.listeners):
            fn(ev)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            await self._km.start_kernel(
                env=kernel_env(self.share_name, self.cwd),
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            threading.Thread(target=self._pump_logs, daemon=True).start()
            self._kc = self._km.client()
            self._kc.start_channels()
            await self._kc.wait_for_ready(timeout=MATLAB_START_TIMEOUT)
            self._tasks = [
                asyncio.create_task(self._guard("iopub", self._read_iopub())),
                asyncio.create_task(self._guard("shell", self._read_shell())),
            ]
            cell = await self._run("%%matlab new_session", "human", record=False)
            if cell.status == "error":
                raise RuntimeError(cell.text())
        except Exception as e:
            self.error = str(e)
            self._set_state("dead")
            self.ready.set()
            return
        self._set_state("idle")
        self.ready.set()

    def _pump_logs(self) -> None:
        proc = getattr(self._km.provisioner, "process", None)
        if proc is None or proc.stdout is None:
            return
        prev = ""
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if "Access MATLAB at" in prev and line.strip().startswith("http"):
                self.desktop_url = line.strip()
            prev = line
            self.logs.append(line)
            del self.logs[:-500]
            if self._loop:
                self._loop.call_soon_threadsafe(lambda ln=line: self._emit("log", text=ln))

    def _set_state(self, state: State, owner: Author | None = None) -> None:
        self.state = state
        self.owner = owner
        self._emit("state")

    async def execute(self, code: str, author: Author) -> Cell:
        await self.ready.wait()
        if self.state == "dead":
            raise RuntimeError(self.error or "session is dead")
        async with self._lock:
            return await self._run(code, author)

    async def _run(self, code: str, author: Author, record: bool = True) -> Cell:
        loop = asyncio.get_running_loop()
        cell = Cell(index=len(self.cells) + 1, author=author, code=code)
        if record:
            self.cells.append(cell)
        self._set_state("busy", author)
        self._emit("cell_started", cell=cell)
        msg_id = self._kc.execute(code)
        self._pending[msg_id] = cell
        self._idle[msg_id] = loop.create_future()
        self._replies[msg_id] = loop.create_future()
        await self._idle[msg_id]
        reply = await self._replies[msg_id]
        del self._replies[msg_id]
        failed = reply["content"]["status"] != "ok" or any(
            o.kind == "error" or (o.kind == "stream" and o.data["name"] == "stderr") for o in cell.outputs
        )
        cell.status = "error" if failed else "ok"
        cell.finished = datetime.now()
        self._set_state("idle")
        self._emit("cell_finished", cell=cell)
        if record:
            self.recorder.append(cell)
        return cell

    async def _guard(self, name: str, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logs.append(f"mash: {name} reader died: {e!r}")
            self.error = f"{name} reader died: {e!r}"
            self._set_state("dead")

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
                self._idle.pop(pid).set_result(None)
            elif t == "stream":
                out = Output("stream", {"name": c["name"], "text": c["text"]})
                cell.outputs.append(out)
                self._emit("output", cell=cell, output=out)
            elif t in ("display_data", "execute_result"):
                did = c.get("transient", {}).get("display_id")
                out = Output("data", c["data"], c.get("metadata", {}), did)
                cell.outputs.append(out)
                self._emit("output", cell=cell, output=out)
            elif t == "update_display_data":
                did = c.get("transient", {}).get("display_id")
                for out in cell.outputs:
                    if out.display_id == did:
                        out.data = c["data"]
                        out.metadata = c.get("metadata", {})
                        self._emit("output", cell=cell, output=out)
            elif t == "error":
                out = Output("error", {k: c[k] for k in ("ename", "evalue", "traceback")})
                cell.outputs.append(out)
                self._emit("output", cell=cell, output=out)

    async def _read_shell(self) -> None:
        while True:
            msg = await self._kc.get_shell_msg()
            pid = msg["parent_header"].get("msg_id")
            if pid not in self._replies:
                self.logs.append(f"mash: unmatched shell {msg['msg_type']} parent={pid}")
            fut = self._replies.get(pid)
            if fut is not None and not fut.done():
                fut.set_result(msg)

    async def complete(self, code: str, cursor: int) -> tuple[list[str], int, int]:
        if self.state != "idle":
            return [], cursor, cursor
        loop = asyncio.get_running_loop()
        msg_id = self._kc.complete(code, cursor)
        self._replies[msg_id] = loop.create_future()
        try:
            reply = await asyncio.wait_for(self._replies[msg_id], 10)
        except TimeoutError:
            self._replies.pop(msg_id, None)
            return [], cursor, cursor
        c = reply["content"]
        return c.get("matches", []), c.get("cursor_start", cursor), c.get("cursor_end", cursor)

    async def interrupt(self) -> None:
        if self.state == "busy":
            await self._km.interrupt_kernel()

    async def whos(self) -> str:
        await self.ready.wait()
        async with self._lock:
            cell = await self._run("whos", "agent", record=False)
        return cell.text()

    async def snapshot(self) -> Cell:
        target = self.record_dir / f"{self.name}.mat"
        return await self.execute(f"save('{target}')", "human")

    def history(self, n: int = 10) -> list[Cell]:
        return self.cells[-n:]

    async def _graceful_stop(self) -> None:
        await self._km.shutdown_kernel()

    async def shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._kc:
            self._kc.stop_channels()
        seen: dict[int, psutil.Process] = {}
        proc = getattr(self._km.provisioner, "process", None)

        def track() -> None:
            if proc is None:
                return
            with contextlib.suppress(psutil.Error):
                for child in psutil.Process(proc.pid).children(recursive=True):
                    seen.setdefault(child.pid, child)

        track()
        if self._km.has_kernel:
            stop = asyncio.create_task(self._graceful_stop())
            deadline = asyncio.get_running_loop().time() + 30
            while not stop.done() and asyncio.get_running_loop().time() < deadline:
                track()
                await asyncio.sleep(0.3)
            if not stop.done():
                stop.cancel()
                with contextlib.suppress(Exception):
                    await self._km.shutdown_kernel(now=True)
        alive = [d for d in seen.values() if d.is_running()]
        for d in alive:
            with contextlib.suppress(psutil.Error):
                d.terminate()
        _, still = psutil.wait_procs(alive, timeout=10)
        for d in still:
            with contextlib.suppress(psutil.Error):
                d.kill()
        self._set_state("dead")
