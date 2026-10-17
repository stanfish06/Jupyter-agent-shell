from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mash.session import Session
from mash.workspace import Workspace


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def mash_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASH_HOME", str(tmp_path / "mash-home"))
    monkeypatch.setenv("MASH_PYTHON", sys.executable)
    monkeypatch.delenv("MASH_DEFAULT_KERNEL", raising=False)


@pytest.fixture
async def workspace(tmp_path: Path) -> AsyncIterator[Workspace]:
    path = tmp_path / "ws"
    path.mkdir()
    ws = Workspace("test", path)
    yield ws
    await ws.close()


async def started(ws: Workspace, *languages: str, name: str = "s1", timeout: float = 60) -> Session:
    session = ws.new_session(name, languages=languages)
    kernels = list(session.kernels.values())
    await asyncio.wait_for(asyncio.gather(*(k.ready.wait() for k in kernels)), timeout)
    for k in kernels:
        assert k.state == "idle", k.error
    return session
