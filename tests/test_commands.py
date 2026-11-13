import json

import pytest

from datadelta.cli import main
from datadelta.io import read_records


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def labelled(tmp_path):
    old = [
        {"id": i, "text": f"sample sentence number {i}", "label": "pos" if i % 2 else "neg"}
        for i in range(40)
    ]
    new = [dict(r) for r in old]
    for r in new[:8]:
        r["label"] = "neg" if r["label"] == "pos" else "pos"
    new[10]["text"] = new[10]["text"] + " with more words"
    new.append({"id": 99, "text": "a brand new row", "label": "pos", "source": "extra"})
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, old)
    _write_jsonl(b, new)
    return a, b


def test_schema_command(labelled, capsys):
    a, b = labelled
    assert main(["schema", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "| source | field_added |" in out
    assert main(["schema", str(a), str(b), "--fail-on-change"]) == 1
    capsys.readouterr()
    assert main(["schema", str(a), str(a), "--fail-on-change", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["changes"] == []


def test_drift_command(labelled, capsys):
    a, b = labelled
    assert main(["drift", str(a), str(b), "--key", "id", "--format", "json"]) == 0
    rows = {d["field"]: d for d in json.loads(capsys.readouterr().out)}
    assert "id" not in rows
    assert rows["label"]["flips"] == 8
    assert rows["text"]["kind"] == "text"
    assert main(["drift", str(a), str(b), "--key", "id", "--fields", "label"]) == 0
    out = capsys.readouterr().out
    assert "| label |" in out and "| text |" not in out


def test_diff_report_includes_schema_drift_and_flip_threshold(labelled, capsys):
    a, b = labelled
    assert main(["diff", str(a), str(b), "--key", "id"]) == 0
    out = capsys.readouterr().out
    assert "## Schema" in out and "## Drift" in out
    assert "Fields in one version only" in out and "`source`" in out
    assert main(["diff", str(a), str(b), "--key", "id", "--max-flip-share", "0.1"]) == 1
    assert "flip share" in capsys.readouterr().err
    assert main(["diff", str(a), str(b), "--key", "id", "--max-psi", "100"]) == 0
    assert main(["diff", str(a), str(b), "--key", "id", "--max-psi", "-1"]) == 1
    capsys.readouterr()
    assert main(["diff", str(a), str(b), "--key", "id", "--no-drift", "--format", "json"]) == 0
    assert "drift" not in json.loads(capsys.readouterr().out)


def test_examples_are_sampled_with_a_seed(labelled, capsys):
    a, b = labelled
    picks = []
    for seed in (1, 1, 2):
        main(
            [
                "diff",
                str(a),
                str(b),
                "--key",
                "id",
                "--format",
                "json",
                "--examples",
                "2",
                "--seed",
                str(seed),
            ]
        )
        data = json.loads(capsys.readouterr().out)
        picks.append([e["old_index"] for e in data["examples"]["relabel"]])
    assert picks[0] == picks[1]
    assert len(picks[0]) == 2 and picks[0] == sorted(picks[0])


def test_show_command(labelled, capsys):
    a, b = labelled
    assert main(["show", str(a), str(b), "--key", "id", "--key-value", "10"]) == 0
    out = capsys.readouterr().out
    assert "status: edited" in out and "extension" in out and "{+ with more words+}" in out
    assert main(["show", str(a), str(b), "--new-index", "40", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["status"] == "added"
    assert main(["show", str(a), str(b), "--old-index", "0"]) == 0
    assert "status:" in capsys.readouterr().out
    assert main(["show", str(a), str(b), "--key-value", "10"]) == 2
    assert main(["show", str(a), str(b), "--key", "id", "--key-value", "12345"]) == 2


def test_show_with_composite_key(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, [{"x": 1, "y": "a", "v": 1}, {"x": 1, "y": "b", "v": 2}])
    _write_jsonl(b, [{"x": 1, "y": "a", "v": 1}, {"x": 1, "y": "b", "v": 5}])
    assert main(["show", str(a), str(b), "--key", "x,y", "--key-value", '[1, "b"]']) == 0
    out = capsys.readouterr().out
    assert "numeric" in out and "delta +3" in out


def test_parquet_round_trip(tmp_path, capsys):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [{"q": "two plus two", "a": "4"}, {"q": "capital of Peru", "a": "Lima"}]
    path = tmp_path / "x.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert read_records(path) == rows
    other = tmp_path / "y.parquet"
    pq.write_table(pa.Table.from_pylist([rows[0], {"q": "capital of Peru", "a": "lima"}]), other)
    assert main(["diff", str(path), str(other), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["edit_classes"] == {"case": 1}
