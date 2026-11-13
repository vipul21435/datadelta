import json

import pytest

from datadelta.io import FormatError, detect_format, read_records


def test_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2, "text": "x"}\n', encoding="utf-8")
    assert read_records(path) == [{"id": 1}, {"id": 2, "text": "x"}]


def test_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": 1}\n{"id": \n', encoding="utf-8")
    with pytest.raises(FormatError, match=":2:"):
        read_records(path)


def test_json_array_and_wrapper(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    assert read_records(path) == [{"a": 1}, {"a": 2}]
    path.write_text(json.dumps({"version": "2", "data": [{"a": 3}]}), encoding="utf-8")
    assert read_records(path) == [{"a": 3}]


def test_json_that_is_really_jsonl(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert read_records(path) == [{"a": 1}, {"a": 2}]


def test_csv_type_inference(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("id,score,label,note\n1,0.5,pos,\n2,3,neg, padded \n", encoding="utf-8")
    rows = read_records(path)
    assert rows[0] == {"id": 1, "score": 0.5, "label": "pos", "note": None}
    assert rows[1]["note"] == " padded "
    raw = read_records(path, infer_types=False)
    assert raw[0]["id"] == "1"


def test_tsv_and_detection(tmp_path):
    path = tmp_path / "a.tsv"
    path.write_text("a\tb\nx\t2\n", encoding="utf-8")
    assert read_records(path) == [{"a": "x", "b": 2}]
    assert detect_format("x.ndjson") == "jsonl"
    with pytest.raises(FormatError):
        detect_format("x.txt")
