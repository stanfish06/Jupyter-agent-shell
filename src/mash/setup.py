from __future__ import annotations

import subprocess
import sys

from mash.config import matlab_root


def main() -> None:
    root = matlab_root()
    engine = root / "extern" / "engines" / "python"
    if not engine.is_dir():
        sys.exit(f"no Python engine under {root}; set MASH_MATLAB_ROOT to a MATLAB install")
    print(f"installing MATLAB Engine for Python from {engine}")
    proc = subprocess.run(["uv", "pip", "install", "--python", sys.executable, str(engine)], check=False)
    sys.exit(proc.returncode)
