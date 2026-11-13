"""Align the records of two dataset versions.

The cascade, applied to whatever is still unpaired after each step:

1. ``key``: records with equal key values are the same record (optional).
2. ``hash``: equal canonical JSON under the chosen :class:`Normalization`.
3. ``loose``: equal after every normalisation (case, whitespace, NFKC, punctuation).
4. ``fuzzy``: token-set Jaccard similarity at or above a threshold, paired greedily from
   the most similar pair down. Candidates come from an inverted index over only the
   rarest tokens of each set, enough that no pair above the threshold is missed; the idea
   of indexing part of each vector is from Bayardo, Ma and Srikant, "Scaling Up All Pairs
   Similarity Search", WWW 2007.

Copies of the same content are paired first copy with first copy. Every tie is broken by
a rule that does not depend on which version is called old, so aligning B to A gives the
mirror image of aligning A to B.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .normalize import STRICT, Normalization, loose_hash, record_hash, tokens

Record = dict[str, Any]
_EPS = 1e-9


@dataclass(frozen=True)
class Match:
    old: int
    new: int
    method: str
    similarity: float | None = None


@dataclass
class Alignment:
    matches: list[Match]
    old_only: list[int]
    new_only: list[int]
    moved: set[int] = field(default_factory=set)
    """Old indices of matched records outside one longest order-preserving subsequence."""

    @property
    def moved_count(self) -> int:
        return len(self.moved)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _match_equal(
    old_key: Callable[[int], Hashable],
    new_key: Callable[[int], Hashable],
    old_idx: Sequence[int],
    new_idx: Sequence[int],
    method: str,
) -> tuple[list[Match], list[int], list[int]]:
    """Multiset matching on equal keys: the k-th old copy pairs with the k-th new copy."""
    buckets: dict[Hashable, deque[int]] = defaultdict(deque)
    for j in new_idx:
        buckets[new_key(j)].append(j)
    matches: list[Match] = []
    old_left: list[int] = []
    used: set[int] = set()
    for i in old_idx:
        queue = buckets.get(old_key(i))
        if queue:
            j = queue.popleft()
            used.add(j)
            matches.append(Match(i, j, method, 1.0 if method == "hash" else None))
        else:
            old_left.append(i)
    new_left = [j for j in new_idx if j not in used]
    return matches, old_left, new_left


def _prefix_len(size: int, threshold: float) -> int:
    return size - math.ceil(threshold * size - _EPS) + 1


def similar_pairs(
    old_sets: dict[int, frozenset[str]],
    new_sets: dict[int, frozenset[str]],
    threshold: float,
) -> list[tuple[float, int, int]]:
    """Every (similarity, old, new) with token Jaccard >= ``threshold``.

    Tokens are ranked rarest first over both pools. Two sets with Jaccard >= t share at
    least ceil(t * |x|) tokens, so they must share a token within their first
    |x| - ceil(t * |x|) + 1 rarest ones; only pairs that do are scored. The result equals a
    brute-force scan of all pairs.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    df: Counter[str] = Counter()
    for s in old_sets.values():
        df.update(s)
    for s in new_sets.values():
        df.update(s)
    rank = {tok: r for r, tok in enumerate(sorted(df, key=lambda t: (df[t], t)))}

    def prefix(s: frozenset[str]) -> list[str]:
        ordered = sorted(s, key=rank.__getitem__)
        return ordered[: _prefix_len(len(ordered), threshold)]

    index: dict[str, list[int]] = defaultdict(list)
    for j, s in new_sets.items():
        if s:
            for tok in prefix(s):
                index[tok].append(j)

    out: list[tuple[float, int, int]] = []
    for i, s in old_sets.items():
        if not s:
            continue
        size = len(s)
        candidates: set[int] = set()
        for tok in prefix(s):
            candidates.update(index.get(tok, ()))
        for j in candidates:
            other = new_sets[j]
            osize = len(other)
            if osize < threshold * size - _EPS or size < threshold * osize - _EPS:
                continue
            inter = len(s & other)
            sim = inter / (size + osize - inter)
            if sim >= threshold - _EPS:
                out.append((sim, i, j))
    return out


def greedy_pairs(scored: list[tuple[float, int, int]]) -> list[tuple[float, int, int]]:
    """Pick pairs from the most similar down, each record used once.

    Ties go to the pair whose positions are closest, then to the smaller position sum.
    Two candidate pairs that share a record never tie on all three, and the rule is the
    same with old and new swapped, so the result is symmetric.
    """
    scored = sorted(scored, key=lambda p: (-p[0], abs(p[1] - p[2]), p[1] + p[2]))
    used_old: set[int] = set()
    used_new: set[int] = set()
    chosen: list[tuple[float, int, int]] = []
    for sim, i, j in scored:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        chosen.append((sim, i, j))
    return chosen


class _Cache:
    """Per-record hashes and token sets, computed on first use."""

    def __init__(self, records: Sequence[Record], norm: Normalization) -> None:
        self.records = records
        self.norm = norm
        self._hash: dict[int, bytes] = {}
        self._loose: dict[int, bytes] = {}
        self._tokens: dict[int, frozenset[str]] = {}

    def hash(self, i: int) -> bytes:
        h = self._hash.get(i)
        if h is None:
            h = self._hash[i] = record_hash(self.records[i], self.norm)
        return h

    def loose(self, i: int) -> bytes:
        h = self._loose.get(i)
        if h is None:
            h = self._loose[i] = loose_hash(self.records[i], self.norm.ignore_fields)
        return h

    def tokens(self, i: int) -> frozenset[str]:
        t = self._tokens.get(i)
        if t is None:
            t = self._tokens[i] = tokens(self.records[i], self.norm.ignore_fields)
        return t


def _content_align(
    old: _Cache,
    new: _Cache,
    old_idx: list[int],
    new_idx: list[int],
    min_similarity: float,
    fuzzy: bool,
) -> tuple[list[Match], list[int], list[int]]:
    matches: list[Match] = []
    m, old_idx, new_idx = _match_equal(old.hash, new.hash, old_idx, new_idx, "hash")
    matches += m
    if old_idx and new_idx:
        m, old_idx, new_idx = _match_equal(old.loose, new.loose, old_idx, new_idx, "loose")
        matches += m
    if fuzzy and old_idx and new_idx:
        scored = similar_pairs(
            {i: old.tokens(i) for i in old_idx},
            {j: new.tokens(j) for j in new_idx},
            min_similarity,
        )
        chosen = greedy_pairs(scored)
        if chosen:
            used_old = {i for _, i, _ in chosen}
            used_new = {j for _, _, j in chosen}
            matches += [Match(i, j, "fuzzy", sim) for sim, i, j in chosen]
            old_idx = [i for i in old_idx if i not in used_old]
            new_idx = [j for j in new_idx if j not in used_new]
    return matches, old_idx, new_idx


def key_of(record: Record, key: Sequence[str]) -> str | None:
    """The record's key as a canonical string, or None when a key field is missing."""
    values = []
    for name in key:
        if name not in record:
            return None
        values.append(record[name])
    return json.dumps(values, sort_keys=True, ensure_ascii=False, default=str)


def longest_increasing(seq: Sequence[int]) -> list[int]:
    """Positions in ``seq`` of one longest strictly increasing subsequence."""
    tails: list[int] = []
    tail_pos: list[int] = []
    prev = [-1] * len(seq)
    for pos, value in enumerate(seq):
        k = bisect_left(tails, value)
        if k == len(tails):
            tails.append(value)
            tail_pos.append(pos)
        else:
            tails[k] = value
            tail_pos[k] = pos
        prev[pos] = tail_pos[k - 1] if k else -1
    out: list[int] = []
    pos = tail_pos[-1] if tail_pos else -1
    while pos != -1:
        out.append(pos)
        pos = prev[pos]
    return out[::-1]


def align(
    old_records: Sequence[Record],
    new_records: Sequence[Record],
    key: Sequence[str] | None = None,
    norm: Normalization = STRICT,
    min_similarity: float = 0.5,
    fuzzy: bool = True,
) -> Alignment:
    """Pair each old record with at most one new record.

    With ``key``, records whose key fields are equal are paired; if a key value repeats
    within a version, the copies under it are paired by content and then by position.
    Records missing a key field, and every record when no key is given, are aligned by
    content: exact hash, then loose hash, then fuzzy token similarity.
    """
    if not 0.0 < min_similarity <= 1.0:
        raise ValueError("min_similarity must be in (0, 1]")
    oc = _Cache(old_records, norm)
    nc = _Cache(new_records, norm)
    matches: list[Match] = []
    old_only: list[int] = []
    new_only: list[int] = []

    if key:
        old_groups: dict[str | None, list[int]] = defaultdict(list)
        new_groups: dict[str | None, list[int]] = defaultdict(list)
        for i, rec in enumerate(old_records):
            old_groups[key_of(rec, key)].append(i)
        for j, rec in enumerate(new_records):
            new_groups[key_of(rec, key)].append(j)
        for k, olds in old_groups.items():
            if k is None:
                continue
            news = new_groups.get(k)
            if not news:
                old_only += olds
                continue
            if len(olds) == 1 and len(news) == 1:
                matches.append(Match(olds[0], news[0], "key"))
                continue
            m, ol, nl = _content_align(oc, nc, olds, news, min_similarity, fuzzy)
            matches += m
            for i, j in zip(ol, nl, strict=False):
                matches.append(Match(i, j, "key"))
            old_only += ol[len(nl) :]
            new_only += nl[len(ol) :]
        for k, news in new_groups.items():
            if k is not None and k not in old_groups:
                new_only += news
        rest_old = old_groups.get(None, [])
        rest_new = new_groups.get(None, [])
    else:
        rest_old = list(range(len(old_records)))
        rest_new = list(range(len(new_records)))

    if rest_old and rest_new:
        m, ol, nl = _content_align(oc, nc, rest_old, rest_new, min_similarity, fuzzy)
        matches += m
        old_only += ol
        new_only += nl
    else:
        old_only += rest_old
        new_only += rest_new

    matches.sort(key=lambda mt: mt.old)
    order = [mt.new for mt in matches]
    keep = set(longest_increasing(order))
    moved = {mt.old for pos, mt in enumerate(matches) if pos not in keep}
    return Alignment(matches, sorted(old_only), sorted(new_only), moved)
