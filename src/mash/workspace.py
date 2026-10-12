from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mash.session import Author, MatlabSession, SessionEvent

if TYPE_CHECKING:
    from mash.agent import Agent


class Workspace:
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.record_dir = path / "mash-notebooks" / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.sessions: dict[str, MatlabSession] = {}
        self.agent: Agent | None = None
        self.listeners: list[Callable[[SessionEvent], None]] = []
        self._counter = 0
        self._tasks: list[asyncio.Task[None]] = []

    def subscribe(self, fn: Callable[[SessionEvent], None]) -> None:
        self.listeners.append(fn)

    def _forward(self, ev: SessionEvent) -> None:
        for fn in list(self.listeners):
            fn(ev)

    def new_session(self, name: str | None = None, created_by: Author = "human") -> MatlabSession:
        self._counter += 1
        name = name or f"s{self._counter}"
        if name in self.sessions:
            raise ValueError(f"session {name} already exists")
        session = MatlabSession(name, self.path, self.record_dir, created_by)
        session.subscribe(self._forward)
        self.sessions[name] = session
        self._tasks.append(asyncio.create_task(session.start()))
        return session

    def get(self, name: str) -> MatlabSession:
        if name not in self.sessions:
            raise KeyError(f"no session named {name}; have {list(self.sessions)}")
        return self.sessions[name]

    async def close_session(self, name: str) -> None:
        session = self.sessions.pop(name)
        await session.shutdown()

    async def close(self) -> None:
        if self.agent:
            await self.agent.stop()
        for name in list(self.sessions):
            await self.close_session(name)


class Manager:
    def __init__(self) -> None:
        self.workspaces: dict[str, Workspace] = {}

    def new_workspace(self, path: Path, name: str | None = None) -> Workspace:
        path = path.expanduser().resolve()
        name = name or path.name
        base, i = name, 1
        while name in self.workspaces:
            i += 1
            name = f"{base}-{i}"
        ws = Workspace(name, path)
        self.workspaces[name] = ws
        return ws

    def get(self, name: str) -> Workspace:
        if name not in self.workspaces:
            raise KeyError(f"no workspace named {name}")
        return self.workspaces[name]

    async def close(self) -> None:
        for ws in list(self.workspaces.values()):
            await ws.close()
