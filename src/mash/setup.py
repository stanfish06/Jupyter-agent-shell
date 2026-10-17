from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mash.backends import BACKENDS, get_backend, language_status
from mash.config import data_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mash-setup",
        description="Set up language runtimes and Jupyter kernels for mash. "
        "With no arguments, shows which languages are ready. "
        "matlab installs the MATLAB Engine for Python (used by the raw REPL) from a MATLAB install.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=["status"],
        metavar="TARGET",
        help=f"status, or a language: {', '.join(BACKENDS)}",
    )
    args = parser.parse_args()
    for target in args.targets:
        if target == "status":
            print(f"mash data dir: {data_dir()}")
            for b, ok, detail in language_status(Path.cwd()):
                print(f"  {b.name:<7} {'ok' if ok else b.missing:<15} {detail}")
            continue
        try:
            backend = get_backend(target)
            print(f"{backend.title}: {backend.install_summary()}", flush=True)
            backend.install(lambda line: print(line, flush=True))
        except Exception as e:
            sys.exit(f"{target}: {e}")
        ok, detail = backend.status(Path.cwd())
        print(f"{backend.title}: {'ready' if ok else 'not ready'} ({detail})")
