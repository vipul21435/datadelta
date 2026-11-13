import math
import random
from collections import Counter

import numpy as np
import pytest
from scipy import stats
from scipy.spatial.distance import jensenshannon

from datadelta.diff import diff
from datadelta.drift import (
    drift,
    field_kind,
    js_divergence,
    kolmogorov_sf,
    ks_2samp,
    psi,
    quantile,
)


@pytest.mark.parametrize("x", [0.1, 0.3, 0.5, 0.8, 1.0, 1.17, 1.19, 1.5, 2.0, 3.0])
def test_kolmogorov_sf_matches_scipy(x):
    assert kolmogorov_sf(x) == pytest.approx(stats.kstwobign.sf(x), abs=1e-10)


@pytest.mark.parametrize("seed", range(6))
def test_ks_statistic_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, rng.integers(5, 400)).round(1)  # rounding creates ties
    y = rng.normal(0.2, 1.3, rng.integers(5, 400)).round(1)
    d, p = ks_2samp(list(x), list(y))
    ref = stats.ks_2samp(x, y, method="asymp")
    assert d == pytest.approx(ref.statistic, abs=1e-12)
    en = len(x) * len(y) / (len(x) + len(y))
    assert p == pytest.approx(stats.kstwobign.sf(math.sqrt(en) * d), abs=1e-10)


def test_ks_pvalue_close_to_scipy_for_large_samples():
    rng = np.random.default_rng(1)
    x, y = rng.normal(0, 1, 3000), rng.normal(0.05, 1, 3000)
    _, p = ks_2samp(list(x), list(y))
    assert p == pytest.approx(stats.ks_2samp(x, y, method="asymp").pvalue, abs=0.01)


def test_ks_identical_and_empty():
    assert ks_2samp([1, 2, 3], [1, 2, 3]) == (0.0, 1.0)
    with pytest.raises(ValueError):
        ks_2samp([], [1])


def test_quantile_matches_numpy():
    rng = random.Random(4)
    values = sorted(rng.random() * 100 for _ in range(37))
    for q in (0, 0.05, 0.25, 0.5, 0.9, 1):
        assert quantile(values, q) == pytest.approx(np.quantile(values, q))


def test_psi_hand_worked():
    old = list(range(1, 11))  # deciles of 1..10 put one value per bin
    assert psi(old, old) == pytest.approx(0.0)
    new = [1] * 5 + [10] * 5
    edges = sorted({quantile(old, k / 10) for k in range(1, 10)})
    assert len(edges) == 9
    # First bin gets 5/10, last bin 5/10, the eight middle bins the floor.
    p, q_first, q_mid = 0.1, 0.5, 1e-4
    expected = 2 * (q_first - p) * math.log(q_first / p) + 8 * (q_mid - p) * math.log(q_mid / p)
    assert psi(old, new) == pytest.approx(expected)


def test_psi_is_large_for_shifted_and_small_for_same_distribution():
    rng = np.random.default_rng(2)
    base = list(rng.normal(0, 1, 5000))
    same = list(rng.normal(0, 1, 5000))
    moved = list(rng.normal(1, 1, 5000))
    assert psi(base, same) < 0.02
    assert psi(base, moved) > 0.5


def test_js_matches_scipy():
    p = Counter({"a": 5, "b": 3, "c": 2})
    q = Counter({"a": 1, "b": 3, "d": 6})
    keys = sorted(set(p) | set(q))
    ref = jensenshannon([p[k] for k in keys], [q[k] for k in keys], base=2) ** 2
    assert js_divergence(p, q) == pytest.approx(ref)
    assert js_divergence(p, p) == pytest.approx(0.0)
    assert js_divergence(Counter("a"), Counter("b")) == pytest.approx(1.0)


def test_field_kinds():
    assert field_kind([0, 1], [1, 1]) == "categorical"
    assert field_kind([0.5, 1.5], [2.5]) == "numeric"
    assert field_kind(list(range(100)), [3]) == "numeric"
    assert field_kind([1, 2, 3], [4, 5]) == "numeric"  # every value distinct: an id
    assert field_kind(["a", "b", "c"], ["d"]) == "text"
    assert field_kind(["pos", "neg"], ["neg"]) == "categorical"
    assert field_kind(["a long sentence " * 4], ["short"]) == "text"
    assert field_kind([[1]], [[2]]) == "other"
    assert field_kind([], []) == "other"


def test_drift_report_on_a_labelled_pair():
    old = [
        {"id": i, "label": "pos" if i % 2 else "neg", "text": "word " * (i % 7 + 1)}
        for i in range(60)
    ]
    new = [dict(r) for r in old]
    for r in new[:10]:
        r["label"] = "neg" if r["label"] == "pos" else "pos"
    for r in new:
        r["text"] = r["text"] + "extra words here"
    result = diff(old, new, key=["id"])
    by_field = {d.field: d for d in drift(old, new, result)}
    label = by_field["label"].stats
    assert by_field["label"].kind == "categorical"
    assert label["flips"] == 10 and label["paired"] == 60
    assert label["flip_share"] == pytest.approx(10 / 60)
    cells = {(c["old"], c["new"]): c["count"] for c in label["transitions"]}
    assert sum(cells.values()) == 60
    assert cells[("pos", "neg")] + cells[("neg", "pos")] == 10
    assert cells[("pos", "pos")] + cells[("neg", "neg")] == 50
    text = by_field["text"]
    assert text.kind == "text"
    assert text.stats["length"]["delta"]["mean"] == pytest.approx(len("extra words here"))
    ref = stats.ks_2samp([len(r["text"]) for r in old], [len(r["text"]) for r in new])
    assert text.stats["length"]["ks_statistic"] == pytest.approx(ref.statistic)
    assert text.stats["unigram_js_divergence"] > 0
    assert by_field["id"].kind == "numeric"
    assert by_field["id"].stats["psi"] == pytest.approx(0.0)
    assert "id" not in {d.field for d in drift(old, new, result, exclude=["id"])}
    only = drift(old, new, result, fields=["label"])
    assert [d.field for d in only] == ["label"]


def test_huge_integers_are_not_numeric():
    assert field_kind([10**400], [1.5]) == "other"
    rows = drift([{"n": 10**400}], [{"n": 10**400 + 1}])
    assert rows[0].kind == "other"
