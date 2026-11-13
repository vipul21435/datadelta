"""Read dataset files into lists of records (one dict per record)."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

Record = dict[str, Any]

FORMATS = ("jsonl", "json", "csv", "tsv", "parquet")


class FormatError(ValueError):
    """Raised when a file cannot be read as a list of records."""


def detect_format(path: str | Path) -> str:
    """Guess the format from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix in (".parquet", ".pq"):
        return "parquet"
    raise FormatError(f"cannot tell the format of {path!s} from its extension; pass a format")


_INT = re.compile(r"-?(?:0|[1-9][0-9]*)")
_FLOAT = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")


def _infer_scalar(text: str) -> Any:
    """Turn a CSV cell into int, float or None only where nothing is lost.

    An empty cell is None. A cell becomes an int when it is plain ASCII digits with an
    optional minus and no zero padding ("7", "-12"; not "007", "+5", "1_000", "-0"), and a
    float when it has the same form with a fraction or exponent ("0.5", "1e3"). Anything
    else, including non-ASCII digits, stays a string. Floats still lose trailing zeros:
    "1.50" and "1.5" both read as 1.5.
    """
    if text == "":
        return None
    if _INT.fullmatch(text):
        value = int(text)
        return value if str(value) == text else text
    if _FLOAT.fullmatch(text):
        number = float(text)
        if math.isfinite(number):
            return number
    return text


def read_jsonl(path: str | Path) -> list[Record]:
    records: list[Record] = []
    with open(path, encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormatError(f"{path!s}:{lineno}: not valid JSON ({exc.msg})") from exc
            except RecursionError as exc:
                raise FormatError(f"{path!s}:{lineno}: JSON nested too deeply") from exc
            if not isinstance(obj, dict):
                raise FormatError(f"{path!s}:{lineno}: expected a JSON object per line")
            records.append(obj)
    return records


def read_json(path: str | Path) -> list[Record]:
    with open(path, encoding="utf-8-sig") as handle:
        text = handle.read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Many ".json" files on the web are really JSON Lines.
        return read_jsonl(path)
    except RecursionError as exc:
        raise FormatError(f"{path!s}: JSON nested too deeply") from exc
    if isinstance(obj, dict):
        # {"data": [...]} style wrappers: unwrap exactly one non-empty list of objects.
        wrapped = [
            k
            for k, v in obj.items()
            if isinstance(v, list) and v and all(isinstance(item, dict) for item in v)
        ]
        if len(wrapped) > 1:
            names = ", ".join(wrapped)
            raise FormatError(f"{path!s}: several lists of records ({names}); save one per file")
        if not wrapped:
            return [obj]
        obj = obj[wrapped[0]]
    if not isinstance(obj, list) or not all(isinstance(item, dict) for item in obj):
        raise FormatError(f"{path!s}: expected a JSON array of objects")
    return obj


def read_csv(path: str | Path, delimiter: str = ",", infer_types: bool = True) -> list[Record]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = reader.fieldnames or []
        repeated = sorted({name for name in header if header.count(name) > 1})
        if repeated:
            raise FormatError(f"{path!s}: repeated header column(s): {', '.join(repeated)}")
        records: list[Record] = []
        for row in reader:
            if None in row:
                raise FormatError(f"{path!s}:{reader.line_num}: more cells than header columns")
            if infer_types:
                records.append({k: None if v is None else _infer_scalar(v) for k, v in row.items()})
            else:
                records.append(dict(row))
    return records


def read_parquet(path: str | Path) -> list[Record]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise FormatError(
            "reading Parquet needs pyarrow: pip install 'datadelta[parquet]'"
        ) from exc
    return pq.read_table(path).to_pylist()


def read_records(
    path: str | Path, fmt: str | None = None, *, infer_types: bool = True
) -> list[Record]:
    """Read ``path`` as a list of records.

    ``fmt`` is one of ``jsonl``, ``json``, ``csv``, ``tsv`` or ``parquet``; when omitted it
    is taken from the extension. CSV cells that look like integers or floats become numbers
    and empty cells become ``None`` unless ``infer_types`` is false.
    """
    fmt = fmt or detect_format(path)
    if fmt == "jsonl":
        return read_jsonl(path)
    if fmt == "json":
        return read_json(path)
    if fmt == "csv":
        return read_csv(path, ",", infer_types)
    if fmt == "tsv":
        return read_csv(path, "\t", infer_types)
    if fmt == "parquet":
        return read_parquet(path)
    raise FormatError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
