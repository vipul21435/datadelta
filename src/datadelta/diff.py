"""Diff two dataset versions: align them, then describe every record."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .align import Alignment, align, jaccard
from .compare import FieldChange, compare_records
from .normalize import STRICT, Normalization, record_hash, tokens

Record = dict[str, Any]

STATUSES = ("unchanged", "edited", "added", "removed", "duplicated", "deduplicated")


@dataclass
class RecordChange:
    """What happened to one record.

    ``duplicated``: an unpaired new record whose content equals another new record.
    ``deduplicated``: an unpaired old record whose content equals another old record.
    """

    status: str
    old_index: int | None = None
    new_index: int | None = None
    method: str | None = None
    similarity: float | None = None
    moved: bool = False
    fields: list[FieldChange] = field(default_factory=list)
    duplicate_of: int | None = None
    key: dict[str, Any] | None = None

    @property
    def classes(self) -> list[str]:
        return sorted({f.edit_class for f in self.fields})

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.key is not None:
            out["key"] = self.key
        if self.old_index is not None:
            out["old_index"] = self.old_index
        if self.new_index is not None:
            out["new_index"] = self.new_index
        if self.method is not None:
            out["method"] = self.method
        if self.similarity is not None:
            out["similarity"] = round(self.similarity, 4)
        if self.moved:
            out["moved"] = True
        if self.duplicate_of is not None:
            out["duplicate_of"] = self.duplicate_of
        if self.fields:
            out["classes"] = self.classes
            out["fields"] = [f.to_dict() for f in self.fields]
        return out


@dataclass
class DiffResult:
    old_count: int
    new_count: int
    changes: list[RecordChange]
    alignment: Alignment
    key: list[str] | None = None

    def counts(self) -> dict[str, int]:
        c = Counter(ch.status for ch in self.changes)
        return {s: c.get(s, 0) for s in STATUSES}

    def method_counts(self) -> dict[str, int]:
        c = Counter(m.method for m in self.alignment.matches)
        return {m: c.get(m, 0) for m in ("key", "hash", "loose", "fuzzy")}

    def class_counts(self) -> dict[str, int]:
        """Edited records per edit class (a record with two classes counts in both)."""
        c: Counter[str] = Counter()
        for ch in self.changes:
            if ch.status == "edited":
                c.update(ch.classes)
        return dict(c.most_common())

    def field_counts(self) -> dict[str, Counter[str]]:
        """Per field, how many edited records changed it, by edit class."""
        out: dict[str, Counter[str]] = {}
        for ch in self.changes:
            if ch.status == "edited":
                for f in ch.fields:
                    out.setdefault(f.field, Counter())[f.edit_class] += 1
        return out

    @property
    def moved_count(self) -> int:
        return self.alignment.moved_count

    @property
    def is_empty(self) -> bool:
        """True when nothing was added, removed, edited or moved."""
        counts = self.counts()
        return counts["unchanged"] == len(self.changes) and self.moved_count == 0

    @property
    def pure_reorder(self) -> bool:
        """True when every record is unchanged but some moved."""
        counts = self.counts()
        return counts["unchanged"] == len(self.changes) and self.moved_count > 0


def _unpaired_status(
    indices: list[int],
    hash_of: Callable[[int], bytes],
    matched: set[int],
    single: str,
    copy: str,
) -> list[tuple[int, str, int | None]]:
    """Name unpaired records ``single`` or, if their content repeats, ``copy``."""
    if not indices:
        return []
    first: dict[bytes, int] = {}
    for i in sorted(set(indices) | matched):
        first.setdefault(hash_of(i), i)
    matched_hashes = {hash_of(i) for i in matched}
    seen: set[bytes] = set()
    out = []
    for i in indices:
        h = hash_of(i)
        if h in matched_hashes or h in seen:
            dup = first[h] if first[h] != i else None
            out.append((i, copy, dup))
        else:
            seen.add(h)
            out.append((i, single, None))
    return out


def diff(
    old: Sequence[Record],
    new: Sequence[Record],
    key: Sequence[str] | None = None,
    norm: Normalization = STRICT,
    min_similarity: float = 0.5,
    rewrite_threshold: float = 0.5,
    fuzzy: bool = True,
    block_tokens: int | None = 8,
) -> DiffResult:
    """Align ``old`` and ``new`` and describe every record.

    ``min_similarity`` is the token Jaccard threshold for fuzzy pairing;
    ``rewrite_threshold`` is the word-level similarity under which a text edit counts as
    a rewrite.
    """
    al = align(
        old,
        new,
        key=key,
        norm=norm,
        min_similarity=min_similarity,
        fuzzy=fuzzy,
        block_tokens=block_tokens,
    )
    changes: list[RecordChange] = []
    for m in al.matches:
        if m.method == "hash":
            fields: list[FieldChange] = []  # equal canonical JSON: nothing to compare
        else:
            fields = compare_records(old[m.old], new[m.new], norm, rewrite_threshold)
        similarity = m.similarity
        if similarity is None and not fields:
            similarity = 1.0
        elif similarity is None:
            similarity = jaccard(
                tokens(old[m.old], norm.ignore_fields), tokens(new[m.new], norm.ignore_fields)
            )
        changes.append(
            RecordChange(
                status="edited" if fields else "unchanged",
                old_index=m.old,
                new_index=m.new,
                method=m.method,
                similarity=similarity,
                moved=m.old in al.moved,
                fields=fields,
            )
        )
    matched_old = {m.old for m in al.matches}
    matched_new = {m.new for m in al.matches}
    old_hash = al.old_hash or (lambda i: record_hash(old[i], norm))
    new_hash = al.new_hash or (lambda j: record_hash(new[j], norm))
    for i, status, dup in _unpaired_status(
        al.old_only, old_hash, matched_old, "removed", "deduplicated"
    ):
        changes.append(RecordChange(status=status, old_index=i, duplicate_of=dup))
    for j, status, dup in _unpaired_status(
        al.new_only, new_hash, matched_new, "added", "duplicated"
    ):
        changes.append(RecordChange(status=status, new_index=j, duplicate_of=dup))
    if key:
        for ch in changes:
            if ch.old_index is not None:
                rec = old[ch.old_index]
            elif ch.new_index is not None:
                rec = new[ch.new_index]
            else:  # pragma: no cover - every change has at least one side
                continue
            if all(k in rec for k in key):
                ch.key = {k: rec[k] for k in key}
    changes.sort(
        key=lambda c: (
            c.old_index if c.old_index is not None else len(old),
            c.new_index if c.new_index is not None else -1,
        )
    )
    return DiffResult(len(old), len(new), changes, al, list(key) if key else None)
