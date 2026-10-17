from __future__ import annotations

import pytest

from mash.setup import main


@pytest.mark.parametrize("argv", [["mash-setup"], ["mash-setup", "status"]])
def test_no_arguments_prints_status(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", argv)
    main()
    out = capsys.readouterr().out
    assert out.startswith("mash data dir:")
    assert [line.split()[0] for line in out.splitlines()[1:]] == ["matlab", "python", "r", "julia"]


def test_unknown_target_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["mash-setup", "cobol"])
    with pytest.raises(SystemExit, match="unknown language 'cobol'"):
        main()
