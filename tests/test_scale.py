"""A synthetic 100,000-row diff, with and without a key, inside the test time limit."""

import random
import time

from datadelta.diff import diff


def _versions(n, seed=0):
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(20000)]
    old = [
        {
            "id": i,
            "text": " ".join(rng.choice(vocab) for _ in range(rng.randint(5, 20))),
            "label": rng.choice("abc"),
        }
        for i in range(n)
    ]
    new, removed, edited, relabelled = [], 0, 0, 0
    for rec in old:
        u = rng.random()
        if u < 0.02:
            removed += 1
            continue
        rec = dict(rec)
        if u < 0.07:
            words = rec["text"].split()
            words[rng.randrange(len(words))] = "edited"
            rec["text"] = " ".join(words)
            edited += 1
        elif u < 0.09:
            rec["label"] = {"a": "b", "b": "c", "c": "a"}[rec["label"]]
            relabelled += 1
        new.append(rec)
    added = n // 50
    for i in range(added):
        new.append({"id": n + i, "text": f"fresh row {i} " + rng.choice(vocab), "label": "a"})
    rng.shuffle(new)
    return old, new, removed, edited + relabelled, added


def test_hundred_thousand_rows():
    old, new, removed, edited, added = _versions(100_000)
    start = time.perf_counter()
    keyed = diff(old, new, key=["id"])
    keyed_seconds = time.perf_counter() - start
    counts = keyed.counts()
    assert counts["removed"] == removed
    assert counts["added"] == added
    # An edit that happens to replace a word with itself leaves the record unchanged.
    assert edited - 50 <= counts["edited"] <= edited
    start = time.perf_counter()
    content = diff(old, new)
    content_seconds = time.perf_counter() - start
    assert content.counts() == counts
    assert keyed_seconds + content_seconds < 40
