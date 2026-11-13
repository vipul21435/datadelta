import pytest

from datadelta.normalize import (
    Normalization,
    canonical_json,
    loose_hash,
    loose_text,
    record_hash,
    strip_punct,
    tokens,
)


def test_key_order_ignored_by_default():
    a = {"x": 1, "y": "two"}
    b = {"y": "two", "x": 1}
    assert record_hash(a) == record_hash(b)
    keep = Normalization(key_order=True)
    assert record_hash(a, keep) != record_hash(b, keep)


def test_nested_keys_sorted():
    a = {"m": {"b": 1, "a": 2}}
    b = {"m": {"a": 2, "b": 1}}
    assert canonical_json(a) == canonical_json(b) == '{"m":{"a":2,"b":1}}'


def test_whitespace_option():
    a = {"t": "hello   world "}
    b = {"t": "hello world"}
    assert record_hash(a) != record_hash(b)
    ws = Normalization(whitespace=True)
    assert record_hash(a, ws) == record_hash(b, ws)


def test_unicode_option():
    composed = {"t": "café"}
    decomposed = {"t": "café"}
    assert record_hash(composed) != record_hash(decomposed)
    nfc = Normalization(unicode="NFC")
    assert record_hash(composed, nfc) == record_hash(decomposed, nfc)
    with pytest.raises(ValueError):
        Normalization(unicode="NFX")


def test_ignore_fields():
    norm = Normalization(ignore_fields=("updated",))
    assert record_hash({"a": 1, "updated": 1}, norm) == record_hash({"a": 1, "updated": 2}, norm)


def test_int_and_float_are_different():
    assert record_hash({"a": 1}) != record_hash({"a": 1.0})


def test_loose_form():
    assert loose_text("  Hello,   WORLD! ") == "hello world"
    assert strip_punct("a-b.c") == "abc"
    assert loose_hash({"t": "Hello, world"}) == loose_hash({"t": "hello world!"})
    assert loose_hash({"t": "hello"}) != loose_hash({"t": "help"})


def test_tokens_cover_nested_values():
    rec = {"text": "The CAT sat.", "meta": {"tags": ["Pets", "cats"]}, "n": 3, "none": None}
    assert tokens(rec) == {"the", "cat", "sat", "pets", "cats", "3"}
    assert tokens(rec, ignore_fields=("meta",)) == {"the", "cat", "sat", "3"}
