from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
from collections.abc import Callable
from typing import Any


class EngineRepl:
    def __init__(self, share_name: str) -> None:
        self.share_name = share_name
        self.eng: Any = None
        self.error: str | None = None

    async def connect(self) -> bool:
        try:
            engine = importlib.import_module("matlab.engine")
        except ImportError:
            self.error = "MATLAB Engine for Python not installed; run: uv run mash-setup matlab"
            return False
        try:
            self.eng = await asyncio.to_thread(engine.connect_matlab, self.share_name)
        except Exception as e:
            self.error = str(e)
            return False
        return True

    async def eval(self, code: str, on_text: Callable[[str], None]) -> None:
        if self.eng is None:
            on_text(f"engine not connected: {self.error}\n")
            return
        out, err = io.StringIO(), io.StringIO()
        fut = self.eng.eval(code, nargout=0, stdout=out, stderr=err, background=True)
        seen = 0
        while not fut.done():
            await asyncio.sleep(0.05)
            seen = self._flush(out, seen, on_text)
        try:
            fut.result()
        except Exception as e:
            on_text(f"{e}\n")
        self._flush(out, seen, on_text)
        if err.getvalue():
            on_text(err.getvalue())

    @staticmethod
    def _flush(buf: io.StringIO, seen: int, on_text: Callable[[str], None]) -> int:
        val = buf.getvalue()
        if len(val) > seen:
            on_text(val[seen:])
        return len(val)

    def close(self) -> None:
        if self.eng is not None:
            with contextlib.suppress(Exception):
                self.eng.quit()
            self.eng = None
