"""Command-line interface: ``datadelta diff | schema | drift | show``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .align import key_of
from .compare import MISSING
from .diff import DiffResult, RecordChange, diff
from .drift import FieldDrift, drift
from .io import FORMATS, FormatError, read_records
from .normalize import Normalization
from .report import (
    drift_markdown,
    iter_jsonl,
    schema_markdown,
    to_json,
    to_markdown,
)
from .schema import schema_diff

Record = dict[str, Any]


def _add_input_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("old", help="old version (JSONL, JSON array, CSV, TSV or Parquet)")
    p.add_argument("new", help="new version")
    p.add_argument("--old-format", choices=FORMATS, help="format of OLD if not clear from its name")
    p.add_argument("--new-format", choices=FORMATS, help="format of NEW if not clear from its name")
    p.add_argument(
        "--no-infer-types", action="store_true", help="keep CSV cells as strings (default: numbers)"
    )


def _add_alignment_options(p: argparse.ArgumentParser) -> None:
    _add_input_options(p)
    p.add_argument("--key", help="comma-separated key fields; without it records align by content")
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="token Jaccard needed to pair two leftover records (default 0.5)",
    )
    p.add_argument(
        "--block-tokens",
        type=int,
        default=8,
        help="index at most this many rarest tokens per record; 0 is exact but slow (default 8)",
    )
    p.add_argument(
        "--rewrite-threshold",
        type=float,
        default=0.5,
        help="word similarity under which a text edit is a rewrite (default 0.5)",
    )
    p.add_argument("--no-fuzzy", action="store_true", help="skip fuzzy pairing")
    p.add_argument("--keep-key-order", action="store_true", help="treat reordered keys as an edit")
    p.add_argument(
        "--normalize-whitespace",
        action="store_true",
        help="treat whitespace-only differences as no change",
    )
    p.add_argument(
        "--unicode",
        choices=("NFC", "NFKC"),
        help="normalise strings to this Unicode form before comparing",
    )
    p.add_argument("--ignore-case", action="store_true", help="treat case differences as no change")
    p.add_argument(
        "--ignore-field", action="append", default=[], help="leave this field out (repeatable)"
    )
    p.add_argument(
        "--schema-fields-as-edits",
        action="store_true",
        help="count a field present in one version only as an edit of every record",
    )


def _add_drift_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-categories",
        type=int,
        default=50,
        help="most distinct values for a field to count as categorical (default 50)",
    )


def _read(args: argparse.Namespace) -> tuple[list[Record], list[Record]]:
    infer = not args.no_infer_types
    old = read_records(args.old, args.old_format, infer_types=infer)
    new = read_records(args.new, args.new_format, infer_types=infer)
    return old, new


def _key(args: argparse.Namespace) -> list[str] | None:
    return [k.strip() for k in args.key.split(",") if k.strip()] if args.key else None


def _run_diff(args: argparse.Namespace, old: list[Record], new: list[Record]) -> DiffResult:
    norm = Normalization(
        key_order=args.keep_key_order,
        whitespace=args.normalize_whitespace,
        unicode=args.unicode,
        casefold=args.ignore_case,
        ignore_fields=tuple(args.ignore_field),
    )
    return diff(
        old,
        new,
        key=_key(args),
        norm=norm,
        min_similarity=args.min_similarity,
        rewrite_threshold=args.rewrite_threshold,
        fuzzy=not args.no_fuzzy,
        block_tokens=args.block_tokens,
        schema_fields_as_edits=args.schema_fields_as_edits,
    )


def _emit(text: str, path: str | None) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n"


def _breaches(
    result: DiffResult, drifts: Sequence[FieldDrift] | None, args: argparse.Namespace
) -> list[str]:
    counts = result.counts()
    out = []
    if args.max_removed is not None and counts["removed"] > args.max_removed:
        out.append(f"removed {counts['removed']} > --max-removed {args.max_removed}")
    if args.max_added is not None and counts["added"] > args.max_added:
        out.append(f"added {counts['added']} > --max-added {args.max_added}")
    if args.max_edited_share is not None:
        share = counts["edited"] / result.old_count if result.old_count else 0.0
        if share > args.max_edited_share:
            out.append(f"edited share {share:.4f} > --max-edited-share {args.max_edited_share}")
    for d in drifts or ():
        st = d.stats
        if args.max_flip_share is not None and st.get("flip_share", 0) > args.max_flip_share:
            out.append(
                f"{d.field}: flip share {st['flip_share']:.4f} > --max-flip-share "
                f"{args.max_flip_share}"
            )
        if args.max_psi is not None:
            value = st.get("psi", st.get("length", {}).get("psi"))
            if value is not None and value > args.max_psi:
                out.append(f"{d.field}: PSI {value:.4f} > --max-psi {args.max_psi}")
    return out


def cmd_diff(args: argparse.Namespace) -> int:
    old, new = _read(args)
    result = _run_diff(args, old, new)
    schema = schema_diff(old, new, args.rate_tolerance)
    drifts = None
    if not args.no_drift:
        drifts = drift(
            old, new, result, max_categories=args.max_categories, exclude=result.key or ()
        )
    if args.format == "json":
        text = _json(to_json(result, args.examples, args.seed, schema, drifts))
    else:
        text = to_markdown(result, args.old, args.new, args.examples, args.seed, schema, drifts)
    _emit(text, args.output)
    if args.changes:
        with open(args.changes, "w", encoding="utf-8") as handle:
            for line in iter_jsonl(result, include_unchanged=args.include_unchanged):
                handle.write(line + "\n")
    breaches = _breaches(result, drifts, args)
    for line in breaches:
        print(f"datadelta: threshold exceeded: {line}", file=sys.stderr)
    return 1 if breaches else 0


def cmd_schema(args: argparse.Namespace) -> int:
    old, new = _read(args)
    schema = schema_diff(old, new, args.rate_tolerance)
    if args.format == "json":
        _emit(_json(schema.to_dict()), args.output)
    else:
        _emit("\n".join(schema_markdown(schema)) + "\n", args.output)
    return 1 if args.fail_on_change and schema.changes else 0


def cmd_drift(args: argparse.Namespace) -> int:
    old, new = _read(args)
    result = _run_diff(args, old, new)
    fields = [f.strip() for f in args.fields.split(",")] if args.fields else None
    drifts = drift(
        old,
        new,
        result,
        fields=fields,
        max_categories=args.max_categories,
        exclude=result.key or (),
    )
    if args.format == "json":
        _emit(_json([d.to_dict() for d in drifts]), args.output)
    else:
        _emit("\n".join(drift_markdown(drifts)).rstrip() + "\n", args.output)
    return 0


def _find(
    result: DiffResult, args: argparse.Namespace, old: list[Record], new: list[Record]
) -> list[RecordChange]:
    if args.old_index is not None:
        hits = [c for c in result.changes if c.old_index == args.old_index]
    elif args.new_index is not None:
        hits = [c for c in result.changes if c.new_index == args.new_index]
    else:
        key = _key(args)
        if not key:
            raise ValueError("--key-value needs --key")
        try:
            value = json.loads(args.key_value)
        except json.JSONDecodeError:
            value = args.key_value
        values = value if isinstance(value, list) and len(key) > 1 else [value]
        wanted = key_of(dict(zip(key, values, strict=False)), key)
        hits = [
            c
            for c in result.changes
            if (c.old_index is not None and key_of(old[c.old_index], key) == wanted)
            or (c.new_index is not None and key_of(new[c.new_index], key) == wanted)
        ]
    if not hits:
        raise ValueError("no record matches that index or key value")
    return hits


def _show_one(ch: RecordChange, old: list[Record], new: list[Record]) -> str:
    lines = [f"status: {ch.status}"]
    if ch.method:
        lines.append(f"paired by: {ch.method} (token similarity {ch.similarity:.3f})")
    if ch.moved:
        lines.append("moved: yes")
    if ch.duplicate_of is not None:
        lines.append(f"copy of: record {ch.duplicate_of} in the same version")
    for label, idx, recs in (("old", ch.old_index, old), ("new", ch.new_index, new)):
        if idx is not None:
            body = json.dumps(recs[idx], indent=2, ensure_ascii=False, default=str)
            lines += ["", f"{label}[{idx}]:", body]
    if ch.fields:
        lines += ["", "changes:"]
        for f in ch.fields:
            lines.append(f"  {f.field or '(key order)'}: {f.edit_class}")
            if f.word_diff is not None:
                lines.append(f"    {f.word_diff}")
            elif f.delta is not None:
                lines.append(f"    {f.old} -> {f.new} (delta {f.delta:+g})")
            else:
                shown_old = "(missing)" if f.old is MISSING else json.dumps(f.old, default=str)
                shown_new = "(missing)" if f.new is MISSING else json.dumps(f.new, default=str)
                lines.append(f"    {shown_old} -> {shown_new}")
    return "\n".join(lines) + "\n"


def cmd_show(args: argparse.Namespace) -> int:
    old, new = _read(args)
    result = _run_diff(args, old, new)
    hits = _find(result, args, old, new)
    if args.format == "json":
        _emit(_json([c.to_dict() for c in hits]), None)
    else:
        sys.stdout.write("\n".join(_show_one(c, old, new) for c in hits))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datadelta", description="What changed between two versions of a dataset."
    )
    parser.add_argument("--version", action="version", version=f"datadelta {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("diff", help="align two versions and report every change")
    _add_alignment_options(p)
    _add_drift_options(p)
    p.add_argument("--format", choices=("md", "json"), default="md", help="report format")
    p.add_argument("-o", "--output", help="write the report here instead of stdout")
    p.add_argument("--changes", help="also write one JSON line per changed record to this file")
    p.add_argument(
        "--include-unchanged", action="store_true", help="list unchanged records in --changes"
    )
    p.add_argument("--examples", type=int, default=3, help="examples per class (default 3)")
    p.add_argument("--seed", type=int, default=0, help="seed for sampling examples (default 0)")
    p.add_argument("--no-drift", action="store_true", help="leave the drift section out")
    p.add_argument(
        "--rate-tolerance",
        type=float,
        default=0.01,
        help="smallest null or blank rate change reported as a schema change (default 0.01)",
    )
    p.add_argument("--max-removed", type=int, help="exit 1 if more records were removed")
    p.add_argument("--max-added", type=int, help="exit 1 if more records were added")
    p.add_argument(
        "--max-edited-share",
        type=float,
        help="exit 1 if edited / old records is above this fraction",
    )
    p.add_argument(
        "--max-flip-share",
        type=float,
        help="exit 1 if any categorical field flips on more than this share of pairs",
    )
    p.add_argument(
        "--max-psi", type=float, help="exit 1 if any numeric or text-length PSI is above this"
    )
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("schema", help="fields added or removed, type and null-rate changes")
    _add_input_options(p)
    p.add_argument("--format", choices=("md", "json"), default="md", help="report format")
    p.add_argument("-o", "--output", help="write the report here instead of stdout")
    p.add_argument(
        "--rate-tolerance",
        type=float,
        default=0.01,
        help="smallest null or blank rate change reported (default 0.01)",
    )
    p.add_argument(
        "--fail-on-change", action="store_true", help="exit 1 if there is any schema change"
    )
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("drift", help="per-field distribution shift")
    _add_alignment_options(p)
    _add_drift_options(p)
    p.add_argument("--fields", help="comma-separated fields to summarise (default: all shared)")
    p.add_argument("--format", choices=("md", "json"), default="md", help="report format")
    p.add_argument("-o", "--output", help="write the report here instead of stdout")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("show", help="show one record pair and its field changes")
    _add_alignment_options(p)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--old-index", type=int, help="0-based position in OLD")
    which.add_argument("--new-index", type=int, help="0-based position in NEW")
    which.add_argument("--key-value", help="key value (JSON or plain text; a list for --key a,b)")
    p.add_argument("--format", choices=("text", "json"), default="text", help="output format")
    p.set_defaults(func=cmd_show)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FormatError, OSError, ValueError) as exc:
        print(f"datadelta: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
