import pytest

from datadelta.schema import profile, schema_diff, type_name


def test_type_names():
    assert [type_name(v) for v in (None, True, 1, 1.5, "x", [1], {"a": 1})] == [
        "null",
        "bool",
        "int",
        "float",
        "str",
        "list",
        "dict",
    ]


def test_profile_counts_nulls_missing_and_blanks():
    recs = [{"a": 1, "b": "x"}, {"a": None, "b": "  "}, {"b": "y"}, {"a": 2.5}]
    prof = profile(recs)
    assert prof["a"].present == 3 and prof["a"].nulls == 1
    assert prof["a"].null_rate == pytest.approx(2 / 4)  # one null, one missing
    assert prof["a"].types == {"int": 1, "float": 1}
    assert prof["b"].empty_rate == pytest.approx(1 / 4)


def test_schema_diff_reports_every_kind_of_change():
    old = [{"id": 1, "label": "a", "score": 1, "note": "x", "old_only": 1} for _ in range(10)]
    new = [{"id": 1, "label": "a", "score": 1.5, "note": "x", "new_only": "y"} for _ in range(10)]
    new[0]["label"] = None
    new[1]["note"] = ""
    changes = {(c.field, c.change) for c in schema_diff(old, new).changes}
    assert changes == {
        ("old_only", "field_removed"),
        ("new_only", "field_added"),
        ("score", "type_changed"),
        ("label", "null_rate_changed"),
        ("note", "empty_rate_changed"),
    }
    result = schema_diff(old, new)
    assert result.added_fields == ["new_only"] and result.removed_fields == ["old_only"]
    assert set(result.to_dict()) == {"changes", "old_fields", "new_fields"}


def test_schema_diff_tolerance_and_identity():
    recs = [{"a": i if i else None} for i in range(200)]
    assert schema_diff(recs, recs).changes == []
    shifted = [dict(r) for r in recs]
    shifted[5]["a"] = None  # null rate 1/200 -> 2/200, under the default tolerance
    assert schema_diff(recs, shifted).changes == []
    assert len(schema_diff(recs, shifted, rate_tolerance=0.001).changes) == 1
