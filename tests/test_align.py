import itertools
import random

import pytest

from datadelta.align import (
    align,
    greedy_pairs,
    jaccard,
    longest_increasing,
    similar_pairs,
)


def _pairs(al):
    return {(m.old, m.new, m.method) for m in al.matches}


def test_identical_lists_align_by_hash():
    recs = [{"t": f"row {i}"} for i in range(5)]
    al = align(recs, list(recs))
    assert _pairs(al) == {(i, i, "hash") for i in range(5)}
    assert al.old_only == [] and al.new_only == [] and al.moved_count == 0


def test_key_alignment_pairs_edited_records():
    old = [{"id": 1, "t": "alpha"}, {"id": 2, "t": "beta"}, {"id": 3, "t": "gamma"}]
    new = [{"id": 3, "t": "gamma"}, {"id": 1, "t": "something else entirely"}, {"id": 4, "t": "d"}]
    al = align(old, new, key=["id"])
    assert _pairs(al) == {(0, 1, "key"), (2, 0, "key")}
    assert al.old_only == [1]
    assert al.new_only == [2]


def test_records_without_key_fall_back_to_content():
    old = [{"id": 1, "t": "a"}, {"t": "no key here"}]
    new = [{"t": "no key here"}, {"id": 1, "t": "a"}]
    al = align(old, new, key=["id"])
    assert _pairs(al) == {(0, 1, "key"), (1, 0, "hash")}


def test_repeated_key_pairs_by_content_then_position():
    old = [{"id": 1, "t": "x"}, {"id": 1, "t": "y"}, {"id": 1, "t": "z"}]
    new = [{"id": 1, "t": "y"}, {"id": 1, "t": "q"}]
    al = align(old, new, key=["id"])
    assert (1, 0, "hash") in _pairs(al)
    assert (0, 1, "key") in _pairs(al)
    assert al.old_only == [2]


def test_loose_pass_pairs_case_and_punctuation_edits():
    old = [{"t": "Hello, world"}, {"t": "unrelated"}]
    new = [{"t": "hello world!"}, {"t": "unrelated"}]
    al = align(old, new)
    assert _pairs(al) == {(0, 0, "loose"), (1, 1, "hash")}


def test_fuzzy_pass_and_threshold():
    old = [{"t": "the quick brown fox jumps over the lazy dog"}]
    new = [{"t": "the quick brown fox leaps over the lazy dog"}]
    al = align(old, new)
    assert [m.method for m in al.matches] == ["fuzzy"]
    assert al.matches[0].similarity == pytest.approx(7 / 9)
    strict = align(old, new, min_similarity=0.9)
    assert strict.matches == [] and strict.old_only == [0] and strict.new_only == [0]
    nofuzzy = align(old, new, fuzzy=False)
    assert nofuzzy.matches == []


def test_duplicates_pair_first_copy_with_first_copy():
    old = [{"t": "a"}, {"t": "a"}]
    new = [{"t": "a"}, {"t": "a"}, {"t": "a"}]
    al = align(old, new)
    assert _pairs(al) == {(0, 0, "hash"), (1, 1, "hash")}
    assert al.new_only == [2]


def test_reordering_is_detected():
    recs = [{"t": f"row {i}"} for i in range(6)]
    shuffled = [recs[i] for i in (0, 1, 5, 2, 3, 4)]
    al = align(recs, shuffled)
    assert al.moved_count == 1
    assert al.moved == {5}


def test_longest_increasing():
    assert longest_increasing([]) == []
    seq = [3, 1, 4, 1, 5, 9, 2, 6]
    pos = longest_increasing(seq)
    values = [seq[p] for p in pos]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert len(pos) == 4


def _random_sets(rng, n, vocab):
    return {i: frozenset(rng.sample(vocab, rng.randint(0, 6))) for i in range(n)}


@pytest.mark.parametrize("threshold", [0.2, 0.34, 0.5, 0.75, 1.0])
def test_prefix_filter_equals_brute_force(threshold):
    rng = random.Random(7)
    vocab = [f"w{k}" for k in range(12)]
    for _ in range(30):
        old = _random_sets(rng, 15, vocab)
        new = _random_sets(rng, 15, vocab)
        fast = {(i, j): s for s, i, j in similar_pairs(old, new, threshold)}
        brute = {
            (i, j): jaccard(a, b)
            for (i, a), (j, b) in itertools.product(old.items(), new.items())
            if a and b and jaccard(a, b) >= threshold - 1e-9
        }
        assert fast.keys() == brute.keys()
        for k, v in brute.items():
            assert fast[k] == pytest.approx(v)


def test_greedy_matching_against_brute_force_optimum_on_unique_scores():
    # With distinct scores, greedy picks the top pair, removes its rows, repeats.
    scored = [(0.9, 0, 0), (0.8, 0, 1), (0.7, 1, 0), (0.6, 1, 1)]
    assert {(i, j) for _, i, j in greedy_pairs(scored)} == {(0, 0), (1, 1)}


def test_similar_pairs_rejects_bad_threshold():
    with pytest.raises(ValueError):
        similar_pairs({}, {}, 0.0)


def _random_records(rng, n):
    words = ["red", "blue", "green", "cat", "dog", "fish", "one", "two", "three", "big"]
    return [
        {"t": " ".join(rng.choice(words) for _ in range(rng.randint(1, 5))), "n": rng.randint(0, 2)}
        for _ in range(n)
    ]


def test_alignment_is_symmetric():
    rng = random.Random(11)
    for _ in range(40):
        a = _random_records(rng, rng.randint(0, 12))
        b = _random_records(rng, rng.randint(0, 12))
        ab = align(a, b, min_similarity=0.3)
        ba = align(b, a, min_similarity=0.3)
        assert {(m.old, m.new) for m in ab.matches} == {(m.new, m.old) for m in ba.matches}
        assert ab.old_only == ba.new_only
        assert ab.new_only == ba.old_only
        assert ab.moved_count == ba.moved_count
