"""Command-line interface: ``datadelta diff OLD NEW``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .diff import DiffResult, diff
from .io import FORMATS, FormatError, read_records
from .normalize import Normalization
from .report import iter_jsonl, to_json, to_markdown


def _add_alignment_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("old", help="old version (JSONL, JSON array, CSV, TSV or Parquet)")
    p.add_argument("new", help="new version")
    p.add_argument("--key", help="comma-separated key fields; without it records align by content")
    p.add_argument("--old-format", choices=FORMATS, help="format of OLD if not clear from its name")
    p.add_argument("--new-format", choices=FORMATS, help="format of NEW if not clear from its name")
    p.add_argument(
        "--no-infer-types", action="store_true", help="keep CSV cells as strings (default: numbers)"
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="token Jaccard needed to pair two leftover records (default 0.5)",
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


def _run_diff(args: argparse.Namespace) -> DiffResult:
    infer = not args.no_infer_types
    old = read_records(args.old, args.old_format, infer_types=infer)
    new = read_records(args.new, args.new_format, infer_types=infer)
    norm = Normalization(
        key_order=args.keep_key_order,
        whitespace=args.normalize_whitespace,
        unicode=args.unicode,
        casefold=args.ignore_case,
        ignore_fields=tuple(args.ignore_field),
    )
    key = [k.strip() for k in args.key.split(",") if k.strip()] if args.key else None
    return diff(
        old,
        new,
        key=key,
        norm=norm,
        min_similarity=args.min_similarity,
        rewrite_threshold=args.rewrite_threshold,
        fuzzy=not args.no_fuzzy,
    )


def _breaches(result: DiffResult, args: argparse.Namespace) -> list[str]:
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
    return out


def cmd_diff(args: argparse.Namespace) -> int:
    result = _run_diff(args)
    if args.format == "json":
        text = json.dumps(to_json(result, args.examples), indent=2, ensure_ascii=False, default=str)
        text += "\n"
    else:
        text = to_markdown(result, args.old, args.new, args.examples)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    if args.changes:
        with open(args.changes, "w", encoding="utf-8") as handle:
            for line in iter_jsonl(result, include_unchanged=args.include_unchanged):
                handle.write(line + "\n")
    breaches = _breaches(result, args)
    for line in breaches:
        print(f"datadelta: threshold exceeded: {line}", file=sys.stderr)
    return 1 if breaches else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datadelta", description="What changed between two versions of a dataset."
    )
    parser.add_argument("--version", action="version", version=f"datadelta {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("diff", help="align two versions and report every change")
    _add_alignment_options(p)
    p.add_argument("--format", choices=("md", "json"), default="md", help="report format")
    p.add_argument("-o", "--output", help="write the report here instead of stdout")
    p.add_argument("--changes", help="also write one JSON line per changed record to this file")
    p.add_argument(
        "--include-unchanged", action="store_true", help="list unchanged records in --changes"
    )
    p.add_argument("--examples", type=int, default=3, help="examples per class (default 3)")
    p.add_argument("--max-removed", type=int, help="exit 1 if more records were removed")
    p.add_argument("--max-added", type=int, help="exit 1 if more records were added")
    p.add_argument(
        "--max-edited-share",
        type=float,
        help="exit 1 if edited / old records is above this fraction",
    )
    p.set_defaults(func=cmd_diff)
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
