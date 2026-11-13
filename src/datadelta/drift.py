"""Distribution shift per field between two dataset versions.

Field kinds:

- ``categorical``: strings, booleans or integers with at most ``max_categories`` distinct
  values over both versions, no more distinct values than half the values, and no
  string longer than 40 characters. Reported: value
  counts, the Jensen-Shannon divergence of the two distributions and, over paired
  records, the transition matrix of old value to new value (label flips).
- ``numeric``: other ints and floats. Reported: mean, standard deviation, quantiles,
  the two-sample Kolmogorov-Smirnov statistic and p-value, and the PSI.
- ``text``: other strings. Reported: the same statistics on character length, and the
  Jensen-Shannon divergence of the token unigram distributions.

Fields of any other type are skipped.
"""

from __future__ import annotations

import math
import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .diff import DiffResult

Record = dict[str, Any]
_WORD = re.compile(r"\w+")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def kolmogorov_sf(x: float) -> float:
    """P(K > x) for the Kolmogorov distribution (the large-sample limit of sqrt(n) D)."""
    if x <= 0:
        return 1.0
    if x < 1.18:
        # Jacobi theta form, which converges fast for small x.
        w = math.sqrt(2 * math.pi) / x
        s = sum(math.exp(-((2 * k - 1) ** 2) * math.pi**2 / (8 * x * x)) for k in range(1, 8))
        return max(0.0, min(1.0, 1.0 - w * s))
    s = sum((-1) ** (k - 1) * math.exp(-2 * k * k * x * x) for k in range(1, 101))
    return max(0.0, min(1.0, 2.0 * s))


def ks_2samp(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic D and its asymptotic two-sided p-value.

    D = max |F_x(v) - F_y(v)| over all observed v (ties handled). The p-value is
    P(K > sqrt(n m / (n + m)) D) under the Kolmogorov distribution, which is accurate for
    large samples and conservative for heavily tied data.
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        raise ValueError("both samples must be non-empty")
    xs, ys = sorted(x), sorted(y)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        v = xs[i] if xs[i] <= ys[j] else ys[j]
        while i < n and xs[i] <= v:
            i += 1
        while j < m and ys[j] <= v:
            j += 1
        d = max(d, abs(i / n - j / m))
    en = n * m / (n + m)
    return d, kolmogorov_sf(math.sqrt(en) * d)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default method)."""
    if not sorted_values:
        raise ValueError("no values")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def psi(old: Sequence[float], new: Sequence[float], bins: int = 10, floor: float = 1e-4) -> float:
    """Population stability index of ``new`` against ``old``.

    Bins are the ``bins``-quantiles of ``old`` (repeated edges merged, so tied data may
    give fewer bins); each bin is (edge_k, edge_k+1]. With p and q the old and new bin
    shares, floored at ``floor`` so empty bins stay finite,
    PSI = sum over bins of (q - p) * ln(q / p).
    """
    if not old or not new:
        raise ValueError("both samples must be non-empty")
    so = sorted(old)
    edges = sorted({quantile(so, k / bins) for k in range(1, bins)})

    def shares(values: Sequence[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect_left(edges, v)] += 1
        return [max(c / len(values), floor) for c in counts]

    p, q = shares(old), shares(new)
    return sum((qi - pi) * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))


def js_divergence(p: Counter[Any] | dict[Any, float], q: Counter[Any] | dict[Any, float]) -> float:
    """Jensen-Shannon divergence in bits (0 for equal distributions, at most 1).

    JS(P, Q) = H(M) - (H(P) + H(Q)) / 2 with M = (P + Q) / 2; inputs are counts or
    weights and are normalised here.
    """
    tp, tq = sum(p.values()), sum(q.values())
    if tp <= 0 or tq <= 0:
        raise ValueError("both distributions need positive mass")
    total = 0.0
    for k in set(p) | set(q):
        a = p.get(k, 0) / tp
        b = q.get(k, 0) / tq
        mid = (a + b) / 2
        if a > 0:
            total += a * math.log2(a / mid)
        if b > 0:
            total += b * math.log2(b / mid)
    return max(0.0, total / 2)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _values(records: Iterable[Record], name: str) -> list[Any]:
    return [r[name] for r in records if r.get(name) is not None]


def field_kind(
    old_values: Sequence[Any], new_values: Sequence[Any], max_categories: int = 50
) -> str:
    values = list(old_values) + list(new_values)
    if not values:
        return "other"

    def few() -> bool:
        distinct = len(set(values))
        return distinct <= max_categories and distinct <= max(2, len(values) // 2)

    if all(_is_number(v) for v in values):
        if all(isinstance(v, int) for v in values) and few():
            return "categorical"
        return "numeric"
    if all(isinstance(v, (str, bool)) for v in values):
        if few() and all(len(v) <= 40 for v in values if isinstance(v, str)):
            return "categorical"
        if all(isinstance(v, str) for v in values):
            return "text"
    return "other"


def _summary(values: Sequence[float]) -> dict[str, float]:
    s = sorted(values)
    mean = sum(s) / len(s)
    var = sum((v - mean) ** 2 for v in s) / (len(s) - 1) if len(s) > 1 else 0.0
    out = {"n": len(s), "mean": mean, "std": math.sqrt(var)}
    for q in QUANTILES:
        out[f"p{round(q * 100)}"] = quantile(s, q)
    return out


def _shift(old: Sequence[float], new: Sequence[float]) -> dict[str, Any]:
    a, b = _summary(old), _summary(new)
    d, p = ks_2samp(old, new)
    return {
        "old": a,
        "new": b,
        "delta": {k: b[k] - a[k] for k in a if k != "n"},
        "ks_statistic": d,
        "ks_pvalue": p,
        "psi": psi(old, new),
    }


def _unigrams(texts: Iterable[str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for t in texts:
        c.update(_WORD.findall(unicodedata.normalize("NFKC", t).casefold()))
    return c


@dataclass
class FieldDrift:
    field: str
    kind: str
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "kind": self.kind, **self.stats}


def transitions(
    old: Sequence[Record], new: Sequence[Record], result: DiffResult, name: str
) -> Counter[tuple[Any, Any]]:
    """Counts of (old value, new value) for ``name`` over paired records."""
    c: Counter[tuple[Any, Any]] = Counter()
    for m in result.alignment.matches:
        c[(old[m.old].get(name), new[m.new].get(name))] += 1
    return c


def drift(
    old: Sequence[Record],
    new: Sequence[Record],
    result: DiffResult | None = None,
    fields: Sequence[str] | None = None,
    max_categories: int = 50,
    top: int = 10,
    exclude: Sequence[str] = (),
) -> list[FieldDrift]:
    """Per-field shift between ``old`` and ``new``.

    ``result`` (from :func:`datadelta.diff.diff`) adds label transitions over paired
    records. ``fields`` limits the report; by default every field present in both
    versions is summarised, except those in ``exclude`` (such as key fields).
    """
    if fields is None:
        seen_old = {k for r in old for k in r}
        names = [
            k
            for k in dict.fromkeys(k for r in new for k in r)
            if k in seen_old and k not in exclude
        ]
    else:
        names = list(fields)
    out: list[FieldDrift] = []
    for name in names:
        vo, vn = _values(old, name), _values(new, name)
        kind = field_kind(vo, vn, max_categories)
        if kind == "other" or not vo or not vn:
            out.append(FieldDrift(name, "other" if kind == "other" else kind, {"skipped": True}))
            continue
        stats: dict[str, Any] = {}
        if kind == "categorical":
            co, cn = Counter(vo), Counter(vn)
            stats["old_counts"] = dict(co.most_common(top))
            stats["new_counts"] = dict(cn.most_common(top))
            stats["categories"] = len(set(co) | set(cn))
            stats["js_divergence"] = js_divergence(co, cn)
            if result is not None:
                trans = transitions(old, new, result, name)
                pairs = sum(trans.values())
                flips = Counter({k: v for k, v in trans.items() if k[0] != k[1]})
                stats["paired"] = pairs
                stats["flips"] = sum(flips.values())
                stats["flip_share"] = stats["flips"] / pairs if pairs else 0.0
                stats["top_flips"] = [
                    {"old": a, "new": b, "count": n} for (a, b), n in flips.most_common(top)
                ]
        elif kind == "numeric":
            stats.update(_shift([float(v) for v in vo], [float(v) for v in vn]))
        else:
            stats["length"] = _shift([len(v) for v in vo], [len(v) for v in vn])
            uo, un = _unigrams(vo), _unigrams(vn)
            stats["vocabulary"] = {"old": len(uo), "new": len(un)}
            stats["unigram_js_divergence"] = js_divergence(uo, un) if uo and un else None
        out.append(FieldDrift(name, kind, stats))
    return out
