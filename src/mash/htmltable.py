from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser

MAX_ROWS = 40


@dataclass
class ParsedTable:
    caption: str
    header: list[str]
    rows: list[list[str]]


@dataclass
class _Collector(HTMLParser):
    caption: list[str] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    _cell: list[str] | None = None
    _row: list[str] | None = None
    _in_head = False
    _in_table = False
    _in_style = False

    def __post_init__(self) -> None:
        super().__init__()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "style":
            self._in_style = True
        elif tag == "table":
            self._in_table = True
        elif tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False
        elif tag == "thead":
            self._in_head = False
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._in_head and not self.header:
                self.header = self._row
            elif not self._in_head:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._in_style:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif not self._in_table:
            self.caption.append(data)


def parse_html_table(html: str) -> ParsedTable | None:
    if "<table" not in html:
        return None
    c = _Collector()
    c.feed(html)
    if not c.header and not c.rows:
        return None
    caption = " ".join(" ".join(c.caption).split())
    return ParsedTable(caption, c.header, c.rows)


def table_text(table: ParsedTable, max_rows: int = MAX_ROWS) -> str:
    lines = [table.caption] if table.caption else []
    if table.header:
        lines.append("\t".join(table.header))
    lines.extend("\t".join(r) for r in table.rows[:max_rows])
    if len(table.rows) > max_rows:
        lines.append(f"… {len(table.rows) - max_rows} more rows")
    return "\n".join(lines)
