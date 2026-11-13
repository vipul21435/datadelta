import pytest

from datadelta.compare import (
    MISSING,
    classify_text,
    compare_records,
    compare_values,
    word_diff,
    word_similarity,
)
from datadelta.normalize import Normalization


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("hello  world", "hello world ", "whitespace"),
        ("ＡBC", "ABC", "unicode"),
        ("Hello World", "hello world", "case"),
        ("Hello, world!", "hello world", "punctuation"),
        ("The answer is 42 because", "The answer is 42", "truncation"),
        ("The answer is 42", "The answer is 42 because six times seven", "extension"),
        ("one two three four", "five six seven eight", "rewrite"),
        ("positive", "negative", "relabel"),
        ("color", "colour", "relabel"),
        ("one two three four five six", "one two three four five seven", "revision"),
    ],
)
def test_text_classes(old, new, expected):
    assert classify_text(old, new) == expected


def test_rewrite_threshold_moves_the_boundary():
    a, b = "a b c d", "a b x y"
    assert word_similarity(a, b) == pytest.approx(0.5)
    assert classify_text(a, b, rewrite_threshold=0.5) == "revision"
    assert classify_text(a, b, rewrite_threshold=0.6) == "rewrite"


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("text", "", "emptied"),
        ("text", None, "emptied"),
        ("text", MISSING, "emptied"),
        ([1], [], "emptied"),
        (None, "text", "filled"),
        (MISSING, 0, "filled"),
        (None, MISSING, "empty_form"),
        ("", None, "empty_form"),
        (1, 3, "numeric"),
        (1, 1.0, "type_change"),
        ("3", 3, "type_change"),
        (True, False, "value"),
        ([1, 2], [2, 1], "value"),
    ],
)
def test_value_classes(old, new, expected):
    change = compare_values("f", old, new)
    assert change is not None and change.edit_class == expected


def test_equal_values_give_no_change():
    assert compare_values("f", {"a": 1, "b": 2}, {"b": 2, "a": 1}) is None
    assert compare_values("f", "x", "x") is None
    assert compare_values("f", 1, True) is not None


def test_numeric_delta():
    change = compare_values("score", 0.25, 0.75)
    assert change.delta == pytest.approx(0.5)


def test_word_diff_marks_changes_and_trims_context():
    old = "a b c d e f g h i j k"
    new = "a b c d e f g h i j changed"
    assert word_diff(old, new, context=2) == "... i j [-k-] {+changed+}"
    assert word_diff("x y", "x z y") == "x {+z+} y"


def test_compare_records_fields_and_normalisation():
    old = {"id": 1, "text": "Hello world", "label": "pos", "gone": "x"}
    new = {"id": 1, "text": "hello world", "label": "neg", "extra": 2}
    changes = {c.field: c.edit_class for c in compare_records(old, new)}
    assert changes == {"text": "case", "label": "relabel", "gone": "emptied", "extra": "filled"}
    folded = compare_records(old, new, Normalization(casefold=True))
    assert "text" not in {c.field for c in folded}


def test_key_order_change_reported_when_significant():
    norm = Normalization(key_order=True)
    changes = compare_records({"a": 1, "b": 2}, {"b": 2, "a": 1}, norm)
    assert [c.edit_class for c in changes] == ["key_order"]
    assert compare_records({"a": 1, "b": 2}, {"b": 2, "a": 1}) == []
