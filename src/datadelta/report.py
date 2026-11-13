"""Render a :class:`DiffResult` as Markdown, JSON or per-record JSON Lines."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .compare import MISSING
from .diff import STATUSES, DiffResult, RecordChange


def _short(value: Any, width: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= width else text[: width - 3] + "..."


def summary(result: DiffResult) -> dict[str, Any]:
    return {
        "old_records": result.old_count,
        "new_records": result.new_count,
        "key": result.key,
        "status": result.counts(),
        "moved": result.moved_count,
        "pure_reorder": result.pure_reorder,
        "matched_by": result.method_counts(),
        "edit_classes": result.class_counts(),
        "fields": {name: dict(c.most_common()) for name, c in result.field_counts().items()},
    }


def to_json(result: DiffResult, examples: int = 3) -> dict[str, Any]:
    out = summary(result)
    out["examples"] = {
        cls: [ch.to_dict() for ch in changes] for cls, changes in _examples(result, examples)
    }
    return out


def iter_jsonl(result: DiffResult, include_unchanged: bool = False) -> Iterator[str]:
    for ch in result.changes:
        if ch.status == "unchanged" and not include_unchanged and not ch.moved:
            continue
        yield json.dumps(ch.to_dict(), ensure_ascii=False, default=str)


def _examples(result: DiffResult, n: int) -> list[tuple[str, list[RecordChange]]]:
    """The first ``n`` edited records of each edit class, and of each other status."""
    by_class: dict[str, list[RecordChange]] = {}
    for ch in result.changes:
        if ch.status == "edited":
            for cls in ch.classes:
                by_class.setdefault(cls, []).append(ch)
        elif ch.status != "unchanged":
            by_class.setdefault(ch.status, []).append(ch)
    order = list(result.class_counts()) + [s for s in STATUSES[2:] if s in by_class]
    return [(cls, by_class[cls][:n]) for cls in order if n > 0]


def _value(value: Any) -> str:
    if value is MISSING:
        return "(missing)"
    return _short(json.dumps(value, ensure_ascii=False, default=str))


def _describe(ch: RecordChange) -> str:
    where = []
    if ch.old_index is not None:
        where.append(f"old[{ch.old_index}]")
    if ch.new_index is not None:
        where.append(f"new[{ch.new_index}]")
    head = " -> ".join(where)
    if ch.key is not None:
        shown = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" for k, v in ch.key.items()
        )
        head = f"{shown} {head}"
    if ch.status != "edited":
        if ch.duplicate_of is not None:
            side = "old" if ch.status == "deduplicated" else "new"
            return f"{head}, copy of {side}[{ch.duplicate_of}]"
        return head
    parts = []
    for f in ch.fields:
        if f.word_diff is not None:
            detail = _short(f.word_diff, 90)
        elif f.delta is not None:
            detail = f"{_value(f.old)} -> {_value(f.new)} ({f.delta:+g})"
        else:
            detail = f"{_value(f.old)} -> {_value(f.new)}"
        parts.append(f"`{f.field}` {f.edit_class}: {detail}")
    return f"{head} by {ch.method}: " + "; ".join(parts)


def to_markdown(
    result: DiffResult, old_name: str = "old", new_name: str = "new", examples: int = 3
) -> str:
    counts = result.counts()
    lines = [
        "# datadelta report",
        "",
        f"old: `{old_name}` ({result.old_count} records); "
        f"new: `{new_name}` ({result.new_count} records)",
        "",
        "Alignment: "
        + (f"key `{', '.join(result.key)}`, then content" if result.key else "content"),
        "",
        "| status | records |",
        "|---|---:|",
    ]
    lines += [f"| {s} | {counts[s]} |" for s in STATUSES]
    lines.append(f"| moved (order) | {result.moved_count} |")
    methods = ", ".join(f"{m} {n}" for m, n in result.method_counts().items())
    lines += ["", f"Pairs found by: {methods}."]
    if result.pure_reorder:
        lines += ["", "Every record is unchanged; only the order differs."]
    classes = result.class_counts()
    if classes:
        lines += ["", "## Edit classes", "", "| class | edited records |", "|---|---:|"]
        lines += [f"| {c} | {n} |" for c, n in classes.items()]
        fields = result.field_counts()
        lines += [
            "",
            "## Fields changed",
            "",
            "| field | edited records | classes |",
            "|---|---:|---|",
        ]
        for name, c in sorted(fields.items(), key=lambda kv: -sum(kv[1].values())):
            detail = ", ".join(f"{k} {v}" for k, v in c.most_common())
            lines.append(f"| {name or '(key order)'} | {sum(c.values())} | {detail} |")
    samples = _examples(result, examples)
    if samples:
        lines += ["", "## Examples"]
        for cls, changes in samples:
            lines += ["", f"### {cls}", ""]
            lines += [f"- {_describe(ch)}" for ch in changes]
    return "\n".join(lines) + "\n"
