import random

from datadelta.diff import diff


def _by_status(result, status):
    return [c for c in result.changes if c.status == status]


def test_diff_of_identical_is_empty():
    recs = [{"id": i, "t": f"text {i}"} for i in range(10)]
    result = diff(recs, [dict(r) for r in recs])
    assert result.is_empty
    assert result.counts()["unchanged"] == 10
    assert not result.pure_reorder


def test_pure_reorder():
    recs = [{"t": f"text {i}"} for i in range(5)]
    result = diff(recs, recs[::-1])
    assert result.pure_reorder and not result.is_empty
    assert result.moved_count == 4


def test_every_status_on_a_hand_built_pair():
    old = [
        {"q": "What is two plus two?", "a": "four", "label": "math"},
        {"q": "Name a primary colour", "a": "red", "label": "art"},
        {"q": "Capital of France", "a": "Paris", "label": "geo"},
        {"q": "Repeated row", "a": "x", "label": "misc"},
        {"q": "Repeated row", "a": "x", "label": "misc"},
    ]
    new = [
        {"q": "What is two plus two?", "a": "four", "label": "math"},
        {"q": "Name a primary colour", "a": "blue", "label": "art"},
        {"q": "Largest planet", "a": "Jupiter", "label": "astro"},
        {"q": "Repeated row", "a": "x", "label": "misc"},
        {"q": "What is two plus two?", "a": "four", "label": "math"},
    ]
    result = diff(old, new)
    counts = result.counts()
    assert counts == {
        "unchanged": 2,
        "edited": 1,
        "added": 1,
        "removed": 1,
        "duplicated": 1,
        "deduplicated": 1,
    }
    edited = _by_status(result, "edited")[0]
    assert (edited.old_index, edited.new_index, edited.method) == (1, 1, "fuzzy")
    assert edited.classes == ["relabel"]
    dup = _by_status(result, "duplicated")[0]
    assert dup.new_index == 4 and dup.duplicate_of == 0
    dedup = _by_status(result, "deduplicated")[0]
    assert dedup.old_index == 4 and dedup.duplicate_of == 3
    assert _by_status(result, "removed")[0].old_index == 2
    assert _by_status(result, "added")[0].new_index == 2


def test_key_diff_reports_field_changes():
    old = [{"id": 1, "text": "Hello world", "score": 3}, {"id": 2, "text": "bye"}]
    new = [{"id": 1, "text": "Hello world, again", "score": 5}, {"id": 2, "text": ""}]
    result = diff(old, new, key=["id"])
    first, second = result.changes
    assert first.method == "key"
    fields = {f.field: f for f in first.fields}
    assert fields["text"].edit_class == "extension"
    assert fields["score"].delta == 2
    assert second.classes == ["emptied"]
    assert result.class_counts() == {"emptied": 1, "extension": 1, "numeric": 1}
    assert result.field_counts()["score"]["numeric"] == 1


def _random_dataset(rng, n):
    words = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]
    return [
        {"t": " ".join(rng.choice(words) for _ in range(rng.randint(1, 6))), "y": rng.randint(0, 1)}
        for _ in range(n)
    ]


def test_added_and_removed_mirror_each_other():
    rng = random.Random(3)
    for _ in range(50):
        a = _random_dataset(rng, rng.randint(0, 15))
        b = _random_dataset(rng, rng.randint(0, 15))
        ab, ba = diff(a, b), diff(b, a)
        added = {c.new_index for c in ab.changes if c.status == "added"}
        removed = {c.old_index for c in ba.changes if c.status == "removed"}
        assert added == removed
        dup = {c.new_index for c in ab.changes if c.status == "duplicated"}
        dedup = {c.old_index for c in ba.changes if c.status == "deduplicated"}
        assert dup == dedup
        edits_ab = {(c.old_index, c.new_index) for c in ab.changes if c.status == "edited"}
        edits_ba = {(c.new_index, c.old_index) for c in ba.changes if c.status == "edited"}
        assert edits_ab == edits_ba


def test_diff_with_itself_is_empty_on_random_data():
    rng = random.Random(5)
    for _ in range(20):
        a = _random_dataset(rng, rng.randint(0, 30))
        assert diff(a, a).is_empty
