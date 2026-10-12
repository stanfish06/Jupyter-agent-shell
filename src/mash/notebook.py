from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nbformat
from nbformat.v4 import new_code_cell, new_notebook, new_output

if TYPE_CHECKING:
    from mash.session import Cell

KERNELSPEC = {
    "kernelspec": {"name": "jupyter_matlab_kernel", "display_name": "MATLAB Kernel", "language": "matlab"},
    "language_info": {"name": "matlab", "file_extension": ".m", "mimetype": "text/x-matlab"},
}


class NotebookRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.nb = nbformat.read(path, as_version=4) if path.exists() else new_notebook(metadata=KERNELSPEC)

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
                "author": cell.author,
                "status": cell.status,
                "started": cell.started.isoformat(timespec="seconds"),
                "finished": cell.finished.isoformat(timespec="seconds") if cell.finished else None,
            }
        }
        self.nb.cells.append(
            new_code_cell(cell.code, outputs=outputs, execution_count=cell.index, metadata=meta)
        )
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
            header = f"%% In[{c.execution_count}] {who} {when}"
            sections.append(f"{header}\n{c.source.rstrip()}\n")
        self.path.with_suffix(".m").write_text("\n".join(sections))
