"""Canonical forms, content hashes and tokens for records.

Two records are *equal* when their canonical JSON under the chosen :class:`Normalization`
is identical. The *loose* form applies every normalisation at once (key order, whitespace,
NFKC, case folding, punctuation removal); it is used only to pair records cheaply before
fuzzy matching, never to decide that a record is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Normalization:
    """What counts as "no change" when two values are compared.

    ``key_order``: when true, a record whose keys were reordered counts as changed.
    ``whitespace``: when true, runs of whitespace collapse to one space and ends are trimmed.
    ``unicode``: ``None``, ``"NFC"`` or ``"NFKC"``; strings are normalised to that form.
    ``casefold``: when true, case differences are ignored.
    ``ignore_fields``: top-level fields left out of every comparison.
    """

    key_order: bool = False
    whitespace: bool = False
    unicode: str | None = None
    casefold: bool = False
    ignore_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.unicode not in (None, "NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError(f"unknown Unicode form {self.unicode!r}")


STRICT = Normalization()
LOOSE = Normalization(whitespace=True, unicode="NFKC", casefold=True)


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


@lru_cache(maxsize=65536)
def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def strip_punct(text: str) -> str:
    """Drop every character in a Unicode punctuation category (P*)."""
    if text.isalnum():
        return text
    return "".join(ch for ch in text if not _is_punct(ch))


def norm_text(text: str, norm: Normalization) -> str:
    if norm.unicode:
        text = unicodedata.normalize(norm.unicode, text)
    if norm.whitespace:
        text = collapse_ws(text)
    if norm.casefold:
        text = text.casefold()
    return text


def loose_text(text: str) -> str:
    """The loosest comparison form of a string: NFKC, case folded, no punctuation."""
    return collapse_ws(strip_punct(unicodedata.normalize("NFKC", text).casefold()))


def norm_value(value: Any, norm: Normalization) -> Any:
    """Apply ``norm`` to every string inside ``value`` (dict keys are left alone)."""
    if isinstance(value, str):
        return norm_text(value, norm)
    if isinstance(value, dict):
        return {k: norm_value(v, norm) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [norm_value(v, norm) for v in value]
    return value


def _loose_value(value: Any) -> Any:
    if isinstance(value, str):
        return loose_text(value)
    if isinstance(value, dict):
        return {k: _loose_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_loose_value(v) for v in value]
    return value


def _strip_fields(record: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    drop = set(fields)
    if not drop:
        return record
    return {k: v for k, v in record.items() if k not in drop}


def canonical_json(record: dict[str, Any], norm: Normalization = STRICT) -> str:
    """Serialise ``record`` so that equal records give identical strings."""
    body = norm_value(_strip_fields(record, norm.ignore_fields), norm)
    return json.dumps(
        body,
        sort_keys=not norm.key_order,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def loose_json(record: dict[str, Any], ignore_fields: Iterable[str] = ()) -> str:
    body = _loose_value(_strip_fields(record, ignore_fields))
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(text: str) -> bytes:
    """A 128-bit content digest; kept as bytes and never shown to the user."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def record_hash(record: dict[str, Any], norm: Normalization = STRICT) -> bytes:
    return digest(canonical_json(record, norm))


def loose_hash(record: dict[str, Any], ignore_fields: Iterable[str] = ()) -> bytes:
    return digest(loose_json(record, ignore_fields))


def _leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _leaves(v)
    elif value is not None:
        yield str(value)


def tokens(record: dict[str, Any], ignore_fields: Iterable[str] = ()) -> frozenset[str]:
    """The set of word tokens (NFKC, case folded) found in any value of ``record``."""
    out: set[str] = set()
    for leaf in _leaves(_strip_fields(record, ignore_fields)):
        out.update(_WORD.findall(unicodedata.normalize("NFKC", leaf).casefold()))
    return frozenset(out)


def words(text: str) -> list[str]:
    """Whitespace-separated words, used for word-level diffs."""
    return text.split()
