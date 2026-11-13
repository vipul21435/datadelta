"""Profile the fields of one dataset version and compare two profiles."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

Record = dict[str, Any]


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


@dataclass
class FieldProfile:
    name: str
    total: int
    present: int = 0
    nulls: int = 0
    empty_strings: int = 0
    types: Counter[str] = field(default_factory=Counter)

    @property
    def null_rate(self) -> float:
        """Share of records where the field is missing or null."""
        return (self.total - self.present + self.nulls) / self.total if self.total else 0.0

    @property
    def empty_rate(self) -> float:
        """Share of records where the field is a blank string."""
        return self.empty_strings / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "null_rate": round(self.null_rate, 6),
            "empty_rate": round(self.empty_rate, 6),
            "types": dict(self.types.most_common()),
        }


def profile(records: Sequence[Record]) -> dict[str, FieldProfile]:
    """Field name -> presence, null count, blank-string count and value types."""
    out: dict[str, FieldProfile] = {}
    n = len(records)
    for rec in records:
        for name, value in rec.items():
            prof = out.get(name)
            if prof is None:
                prof = out[name] = FieldProfile(name, n)
            prof.present += 1
            if value is None:
                prof.nulls += 1
                continue
            prof.types[type_name(value)] += 1
            if isinstance(value, str) and not value.strip():
                prof.empty_strings += 1
    return out


@dataclass
class SchemaChange:
    field: str
    change: str
    old: Any = None
    new: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "change": self.change, "old": self.old, "new": self.new}


@dataclass
class SchemaDiff:
    old: dict[str, FieldProfile]
    new: dict[str, FieldProfile]
    changes: list[SchemaChange]

    @property
    def added_fields(self) -> list[str]:
        return [c.field for c in self.changes if c.change == "field_added"]

    @property
    def removed_fields(self) -> list[str]:
        return [c.field for c in self.changes if c.change == "field_removed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": [c.to_dict() for c in self.changes],
            "old_fields": {k: v.to_dict() for k, v in self.old.items()},
            "new_fields": {k: v.to_dict() for k, v in self.new.items()},
        }


def schema_diff(
    old: Sequence[Record], new: Sequence[Record], rate_tolerance: float = 0.01
) -> SchemaDiff:
    """Fields added or removed, value types changed, and null or blank rates that moved.

    A type change is any difference in the set of non-null value types (so a single
    float among ints counts). Null and blank-string rates are shares of all records; a
    change larger than ``rate_tolerance`` in absolute terms is reported.
    """
    po, pn = profile(old), profile(new)
    changes: list[SchemaChange] = []
    for name in list(po) + [k for k in pn if k not in po]:
        a, b = po.get(name), pn.get(name)
        if b is None:
            changes.append(SchemaChange(name, "field_removed", a.present if a else 0, 0))
            continue
        if a is None:
            changes.append(SchemaChange(name, "field_added", 0, b.present))
            continue
        if set(a.types) != set(b.types):
            changes.append(
                SchemaChange(
                    name, "type_changed", dict(a.types.most_common()), dict(b.types.most_common())
                )
            )
        if abs(a.null_rate - b.null_rate) > rate_tolerance:
            changes.append(
                SchemaChange(
                    name, "null_rate_changed", round(a.null_rate, 6), round(b.null_rate, 6)
                )
            )
        if abs(a.empty_rate - b.empty_rate) > rate_tolerance:
            changes.append(
                SchemaChange(
                    name, "empty_rate_changed", round(a.empty_rate, 6), round(b.empty_rate, 6)
                )
            )
    return SchemaDiff(po, pn, changes)
