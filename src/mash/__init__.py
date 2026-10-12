from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from mash.tui.app import MashApp

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    MashApp(root.expanduser().resolve()).run()
