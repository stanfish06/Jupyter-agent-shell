from __future__ import annotations

import asyncio
import os
import sys
import time

from jupyter_client import AsyncKernelManager

from mash.config import matlab_root

MATLAB_ROOT = str(matlab_root())

CELLS = [
    "x = 1 + 1",
    "disp(x * 10)",
    "y = rand(3)",
    "figure; plot(1:10); title('smoke')",
    "whos",
    "error('boom')",
]


def kernel_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{MATLAB_ROOT}/bin:{env['PATH']}"
    env["MWI_CUSTOM_MATLAB_ROOT"] = MATLAB_ROOT
    env["MWI_USE_EXISTING_LICENSE"] = "True"
    env.setdefault("MWI_LOG_LEVEL", "WARN")
    return env


async def run_cell(kc, code: str) -> None:
    print(f"\n>>> {code}")
    msg_id = kc.execute(code)
    while True:
        msg = await kc.get_iopub_msg(timeout=600)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t = msg["msg_type"]
        c = msg["content"]
        if t == "status" and c["execution_state"] == "idle":
            break
        if t == "stream":
            print(c["text"], end="")
        elif t in ("display_data", "execute_result"):
            data = c["data"]
            print({k: (len(v) if k.startswith("image/") else v[:200]) for k, v in data.items()})
        elif t == "error":
            print("ERROR:", c["ename"], c["evalue"])
    reply = await kc.get_shell_msg(timeout=60)
    print(f"[status={reply['content']['status']}]")


async def main() -> int:
    km = AsyncKernelManager(kernel_name="jupyter_matlab_kernel")
    t0 = time.time()
    await km.start_kernel(env=kernel_env())
    kc = km.client()
    kc.start_channels()
    await kc.wait_for_ready(timeout=300)
    print(f"kernel ready in {time.time() - t0:.1f}s; connection file: {km.connection_file}")
    t0 = time.time()
    await run_cell(kc, "1")
    print(f"MATLAB started in {time.time() - t0:.1f}s")

    for code in CELLS:
        await run_cell(kc, code)
    kc.complete("y(", 2)
    reply = await kc.get_shell_msg(timeout=60)
    print("\ncompletion matches:", reply["content"].get("matches")[:5])

    kc.stop_channels()
    await km.shutdown_kernel(now=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
