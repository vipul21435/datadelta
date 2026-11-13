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


def test_byte_order_mark_is_ignored(tmp_path):
    bom = "﻿"
    csv_path = tmp_path / "bom.csv"
    csv_path.write_text(bom + "id,name\n1,a\n2,b\n", encoding="utf-8")
    assert read_records(csv_path) == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    jsonl_path = tmp_path / "bom.jsonl"
    jsonl_path.write_text(bom + '{"id": 1}\n', encoding="utf-8")
    assert read_records(jsonl_path) == [{"id": 1}]
    json_path = tmp_path / "bom.json"
    json_path.write_text(bom + '[{"id": 1}]', encoding="utf-8")
    assert read_records(json_path) == [{"id": 1}]


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("7", 7),
        ("-12", -12),
        ("0", 0),
        ("0.5", 0.5),
        ("1e3", 1000.0),
        ("007", "007"),
        ("0010", "0010"),
        ("+5", "+5"),
        ("1_000", "1_000"),
        ("-0", "-0"),
        ("７", "７"),  # full-width digit
        ("٧", "٧"),  # Arabic-Indic digit
        (" 5", " 5"),
        ("inf", "inf"),
        ("nan", "nan"),
        ("1e999", "1e999"),
        ("", None),
    ],
)
def test_csv_inference_keeps_text_that_would_change(tmp_path, cell, expected):
    path = tmp_path / "a.csv"
    path.write_text(f"c\n{cell}\n" if cell else "c,d\n,x\n", encoding="utf-8")
    assert read_records(path)[0]["c"] == expected


def test_zero_padded_codes_are_edits_and_keys_do_not_collide(tmp_path):
    from datadelta.diff import diff

    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("k,code\n01,007\n1,0010\n", encoding="utf-8")
    b.write_text("k,code\n01,7\n1,10\n", encoding="utf-8")
    result = diff(read_records(a), read_records(b), key=["k"])
    assert result.counts()["edited"] == 2
    assert result.method_counts()["key"] == 2


def test_json_objects_are_unwrapped_only_when_unambiguous(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"train": [{"id": 1}], "test": [{"id": 2}]}), encoding="utf-8")
    with pytest.raises(FormatError, match="several lists"):
        read_records(path)
    path.write_text(json.dumps({"id": 1, "tags": ["a", "b"]}), encoding="utf-8")
    assert read_records(path) == [{"id": 1, "tags": ["a", "b"]}]
    path.write_text(json.dumps({"id": 1, "tags": []}), encoding="utf-8")
    assert read_records(path) == [{"id": 1, "tags": []}]


def test_repeated_csv_header_is_an_error(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("id,x,x\n1,a,b\n", encoding="utf-8")
    with pytest.raises(FormatError, match="repeated header"):
        read_records(path)
