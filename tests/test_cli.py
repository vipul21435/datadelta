import json
from pathlib import Path

import pytest

from datadelta.cli import main
from datadelta.compare import suffix_diff

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "data"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def pair(tmp_path):
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    _write_jsonl(
        old,
        [
            {"id": 1, "text": "keep this row", "label": "a"},
            {"id": 2, "text": "drop this row", "label": "b"},
            {"id": 3, "text": "Edit this row", "label": "a"},
        ],
    )
    _write_jsonl(
        new,
        [
            {"id": 1, "text": "keep this row", "label": "a"},
            {"id": 3, "text": "edit this row", "label": "a"},
            {"id": 4, "text": "a new row", "label": "c"},
        ],
    )
    return old, new


def test_markdown_report(pair, capsys):
    old, new = pair
    assert main(["diff", str(old), str(new), "--key", "id"]) == 0
    out = capsys.readouterr().out
    assert "| edited | 1 |" in out
    assert "| removed | 1 |" in out
    assert "`text` case" in out


def test_json_report_and_changes_file(pair, tmp_path):
    old, new = pair
    report = tmp_path / "report.json"
    changes = tmp_path / "changes.jsonl"
    code = main(
        [
            "diff",
            str(old),
            str(new),
            "--key",
            "id",
            "--format",
            "json",
            "-o",
            str(report),
            "--changes",
            str(changes),
        ]
    )
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"]["added"] == 1
    assert data["edit_classes"] == {"case": 1}
    assert data["examples"]["case"][0]["key"] == {"id": 3}
    lines = [json.loads(x) for x in changes.read_text(encoding="utf-8").splitlines()]
    assert sorted(x["status"] for x in lines) == ["added", "edited", "removed"]


def test_thresholds_set_exit_code(pair, capsys):
    old, new = pair
    assert main(["diff", str(old), str(new), "--key", "id", "--max-removed", "0"]) == 1
    assert "max-removed" in capsys.readouterr().err
    assert main(["diff", str(old), str(new), "--key", "id", "--max-edited-share", "0.5"]) == 0
    assert main(["diff", str(old), str(new), "--key", "id", "--max-edited-share", "0.2"]) == 1
    assert main(["diff", str(old), str(new), "--key", "id", "--max-added", "1"]) == 0


def test_bad_input_exits_2(tmp_path, capsys):
    bad = tmp_path / "x.txt"
    bad.write_text("hello", encoding="utf-8")
    assert main(["diff", str(bad), str(bad)]) == 2
    assert "error" in capsys.readouterr().err


def test_normalisation_flags(tmp_path, capsys):
    old, new = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(old, [{"t": "Hello  World"}])
    _write_jsonl(new, [{"t": "hello world"}])
    main(["diff", str(old), str(new), "--format", "json"])
    assert json.loads(capsys.readouterr().out)["status"]["edited"] == 1
    main(
        ["diff", str(old), str(new), "--format", "json", "--ignore-case", "--normalize-whitespace"]
    )
    assert json.loads(capsys.readouterr().out)["status"]["unchanged"] == 1


def test_bundled_example_runs(capsys):
    code = main(
        ["diff", str(EXAMPLES / "quiz_v1.jsonl"), str(EXAMPLES / "quiz_v2.jsonl"), "--key", "id"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| unchanged | 2 |" in out and "| edited | 7 |" in out
    assert "Photosynthesis[-, which uses" in out


def test_suffix_diff():
    assert suffix_diff("abc def", "abc") == "abc[- def-]"
    assert suffix_diff("abc", "abcdef") == "abc{+def+}"
    assert suffix_diff("x" * 50 + "y", "x" * 50, context=5) == "...xxxxx[-y-]"


def test_unknown_key_exits_2_and_partial_key_warns(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, [{"id": 1, "t": "x"}, {"t": "no key"}])
    _write_jsonl(b, [{"id": 1, "t": "x"}, {"t": "no key"}])
    assert main(["diff", str(a), str(b), "--key", "nosuchkey"]) == 2
    assert "nosuchkey" in capsys.readouterr().err
    assert main(["diff", str(a), str(b), "--key", "id"]) == 0
    err = capsys.readouterr().err
    assert "1 of 2 old records lack the key" in err
