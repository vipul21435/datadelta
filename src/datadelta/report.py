"""Render a :class:`DiffResult` as Markdown, JSON or per-record JSON Lines."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from typing import Any

from .compare import MISSING
from .diff import STATUSES, DiffResult, RecordChange
from .drift import FieldDrift
from .schema import SchemaDiff


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
        "schema_only_fields": result.schema_only_fields,
    }


def to_json(
    result: DiffResult,
    examples: int = 3,
    seed: int = 0,
    schema: SchemaDiff | None = None,
    drift: Sequence[FieldDrift] | None = None,
) -> dict[str, Any]:
    out = summary(result)
    if schema is not None:
        out["schema"] = [c.to_dict() for c in schema.changes]
    if drift is not None:
        out["drift"] = [d.to_dict() for d in drift]
    out["examples"] = {
        cls: [ch.to_dict() for ch in changes]
        for cls, changes in sample_examples(result, examples, seed)
    }
    return out


def iter_jsonl(result: DiffResult, include_unchanged: bool = False) -> Iterator[str]:
    for ch in result.changes:
        if ch.status == "unchanged" and not include_unchanged and not ch.moved:
            continue
        yield json.dumps(ch.to_dict(), ensure_ascii=False, default=str)


def sample_examples(
    result: DiffResult, n: int, seed: int = 0
) -> list[tuple[str, list[RecordChange]]]:
    """Up to ``n`` records per edit class and per non-unchanged status, drawn at random
    with ``seed`` and listed in file order."""
    rng = random.Random(seed)
    by_class: dict[str, list[RecordChange]] = {}
    for ch in result.changes:
        if ch.status == "edited":
            for cls in ch.classes:
                by_class.setdefault(cls, []).append(ch)
        elif ch.status != "unchanged":
            by_class.setdefault(ch.status, []).append(ch)
    order = list(result.class_counts()) + [s for s in STATUSES[2:] if s in by_class]
    out = []
    for cls in order:
        if n <= 0:
            break
        pool = by_class[cls]
        picked = (
            pool if len(pool) <= n else [pool[i] for i in sorted(rng.sample(range(len(pool)), n))]
        )
        out.append((cls, picked))
    return out


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


def _num(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e6):
            return f"{x:.3g}"
        return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:.1f}"
    return str(x)


def schema_markdown(schema: SchemaDiff) -> list[str]:
    lines = ["## Schema", ""]
    if not schema.changes:
        return lines + ["No schema changes."]
    lines += ["| field | change | old | new |", "|---|---|---|---|"]
    for c in schema.changes:
        lines.append(f"| {c.field} | {c.change} | {_short(c.old, 40)} | {_short(c.new, 40)} |")
    return lines


def drift_markdown(drift: Sequence[FieldDrift], top: int = 5) -> list[str]:
    lines = ["## Drift", ""]
    cats = [d for d in drift if d.kind == "categorical"]
    nums = [d for d in drift if d.kind == "numeric"]
    texts = [d for d in drift if d.kind == "text"]
    if cats:
        lines += [
            "Categorical fields (flips counted over paired records):",
            "",
            "| field | categories | JS divergence | paired | flips | flip share | top flips |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for d in cats:
            st = d.stats
            flips = ", ".join(
                f"{_short(f['old'], 20)} -> {_short(f['new'], 20)} ({f['count']})"
                for f in st.get("top_flips", [])[:top]
            )
            lines.append(
                f"| {d.field} | {st['categories']} | {_num(st['js_divergence'])} | "
                f"{_num(st.get('paired'))} | {_num(st.get('flips'))} | "
                f"{_num(st.get('flip_share'))} | {flips or '-'} |"
            )
        lines.append("")
    if texts:
        lines += [
            "Text fields (length in characters):",
            "",
            "| field | mean old | mean new | median old | median new | KS D | KS p | PSI "
            "| unigram JS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for d in texts:
            ln = d.stats["length"]
            lines.append(
                f"| {d.field} | {_num(ln['old']['mean'])} | {_num(ln['new']['mean'])} | "
                f"{_num(ln['old']['p50'])} | {_num(ln['new']['p50'])} | "
                f"{_num(ln['ks_statistic'])} | {_num(ln['ks_pvalue'])} | {_num(ln['psi'])} | "
                f"{_num(d.stats['unigram_js_divergence'])} |"
            )
        lines.append("")
    if nums:
        lines += [
            "Numeric fields:",
            "",
            "| field | mean old | mean new | p50 old | p50 new | p95 old | p95 new | KS D | KS p "
            "| PSI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for d in nums:
            st = d.stats
            lines.append(
                f"| {d.field} | {_num(st['old']['mean'])} | {_num(st['new']['mean'])} | "
                f"{_num(st['old']['p50'])} | {_num(st['new']['p50'])} | "
                f"{_num(st['old']['p95'])} | {_num(st['new']['p95'])} | "
                f"{_num(st['ks_statistic'])} | {_num(st['ks_pvalue'])} | {_num(st['psi'])} |"
            )
        lines.append("")
    if not (cats or texts or nums):
        lines.append("No fields to summarise.")
    return lines


def to_markdown(
    result: DiffResult,
    old_name: str = "old",
    new_name: str = "new",
    examples: int = 3,
    seed: int = 0,
    schema: SchemaDiff | None = None,
    drift: Sequence[FieldDrift] | None = None,
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
    if result.schema_only_fields:
        shown = ", ".join(f"`{f}`" for f in result.schema_only_fields)
        lines += ["", f"Fields in one version only, left out of record comparison: {shown}."]
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
    if schema is not None:
        lines += [""] + schema_markdown(schema)
    if drift is not None:
        lines += [""] + drift_markdown(drift)
        while lines[-1] == "":
            lines.pop()
    samples = sample_examples(result, examples, seed)
    if samples:
        lines += ["", "## Examples"]
        for cls, changes in samples:
            lines += ["", f"### {cls}", ""]
            lines += [f"- {_describe(ch)}" for ch in changes]
    return "\n".join(lines) + "\n"
