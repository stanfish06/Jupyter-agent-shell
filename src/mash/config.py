from __future__ import annotations

import os
import shutil
import socket
import sys
from functools import cache
from glob import glob
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[2] / "node_modules" / ".bin" / "claude-agent-acp"
PREFERRED_MCP_PORT = int(os.environ.get("MASH_MCP_PORT", "8765"))
KERNEL_NAME = "jupyter_matlab_kernel"
MATLAB_START_TIMEOUT = 300

INSTALL_GLOBS = {
    "darwin": ["/Applications/MATLAB_R20*.app"],
    "linux": ["/usr/local/MATLAB/R20*", "/opt/MATLAB/R20*", "~/MATLAB/R20*"],
    "win32": [r"C:\Program Files\MATLAB\R20*"],
}


@cache
def matlab_root() -> Path:
    override = os.environ.get("MASH_MATLAB_ROOT")
    if override:
        return Path(override).expanduser()
    exe = shutil.which("matlab")
    if exe:
        return Path(os.path.realpath(exe)).parents[1]
    patterns = INSTALL_GLOBS.get(sys.platform, INSTALL_GLOBS["linux"])
    found = sorted(p for pat in patterns for p in glob(os.path.expanduser(pat)))
    if found:
        return Path(found[-1])
    raise RuntimeError("MATLAB not found: put matlab on PATH or set MASH_MATLAB_ROOT")


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise RuntimeError("no free TCP port")


def kernel_env(share_name: str, cwd: Path) -> dict[str, str]:
    root = matlab_root()
    env = dict(os.environ)
    env["PATH"] = f"{root / 'bin'}{os.pathsep}{env['PATH']}"
    env["MWI_CUSTOM_MATLAB_ROOT"] = str(root)
    env["MWI_USE_EXISTING_LICENSE"] = "True"
    env.setdefault("MWI_LOG_LEVEL", "WARN")
    env["MWI_MATLAB_STARTUP_SCRIPT"] = f"cd('{cwd}'); matlab.engine.shareEngine('{share_name}');"
    return env
