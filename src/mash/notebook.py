from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import nbformat
from nbformat.v4 import new_code_cell, new_notebook, new_output

if TYPE_CHECKING:
    from mash.backends import Backend
    from mash.session import Cell


class NotebookRecorder:
    def __init__(self, path: Path, backend: Backend, language_info: dict[str, Any]) -> None:
        self.path = path
        self.backend = backend
        if path.exists():
            self.nb = nbformat.read(path, as_version=4)
        else:
            info = dict(language_info) or {"name": backend.name, "file_extension": backend.file_ext}
            meta = {"kernelspec": backend.notebook_kernelspec(language_info), "language_info": info}
            self.nb = new_notebook(metadata=meta)

    def append(self, cell: Cell) -> None:
        outputs = []
        for o in cell.outputs:
            if o.kind == "stream":
                outputs.append(new_output("stream", name=o.data["name"], text=o.data["text"]))
            elif o.kind == "data" and o.data:
                outputs.append(new_output("display_data", data=o.data, metadata=o.metadata))
            elif o.kind == "error":
                outputs.append(new_output("error", **o.data))
        meta = {
            "mash": {
                "index": cell.index,
                "kernel": cell.kernel,
                "author": cell.author,
                "status": cell.status,
                "started": cell.started.isoformat(timespec="seconds"),
                "finished": cell.finished.isoformat(timespec="seconds") if cell.finished else None,
            }
        }
        count = sum(1 for c in self.nb.cells if c.cell_type == "code") + 1
        self.nb.cells.append(new_code_cell(cell.code, outputs=outputs, execution_count=count, metadata=meta))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(self.nb, self.path)
        self.write_script()

    def write_script(self) -> None:
        sections = []
        for c in self.nb.cells:
            if c.cell_type != "code":
                continue
            m = c.metadata.get("mash", {})
            who = "you" if m.get("author") == "human" else m.get("author", "?")
            when = m.get("started", "")
            header = f"{self.backend.marker} In[{c.execution_count}] {who} {when}"
            sections.append(f"{header}\n{c.source.rstrip()}\n")
        self.path.with_suffix(self.backend.file_ext).write_text("\n".join(sections))
