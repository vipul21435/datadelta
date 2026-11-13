"""Compare two versions of one record field by field and name each edit.

Text edit classes are cumulative: each one is the smallest normalisation that makes the
two strings equal, tried in this order.

=============  ==============================================================
whitespace     equal once runs of whitespace collapse and ends are trimmed
unicode        ...and both are in NFKC form
case           ...and both are case folded
punctuation    ...and Unicode punctuation (categories P*) is removed
truncation     the new text is a strict prefix of the old (whitespace collapsed)
extension      the old text is a strict prefix of the new
relabel        both texts are a single word each (a category label changed)
rewrite        word-level similarity below the rewrite threshold
revision       any other text edit
=============  ==============================================================

Other classes: ``emptied`` (a value became missing, null, "" or an empty container),
``filled`` (the reverse), ``empty_form`` (one empty form became another, such as missing
to null), ``numeric`` (a number changed; the delta is kept), ``type_change`` and ``value``
(any other change, such as a list or a boolean), and ``key_order`` when only the order
of keys differs and key order was declared significant.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from os.path import commonprefix
from typing import Any

from .normalize import STRICT, Normalization, collapse_ws, loose_text, norm_value, words

MISSING: Any = type("Missing", (), {"__repr__": lambda self: "MISSING"})()

TEXT_CLASSES = (
    "whitespace",
    "unicode",
    "case",
    "punctuation",
    "truncation",
    "extension",
    "relabel",
    "rewrite",
    "revision",
)
EDIT_CLASSES = TEXT_CLASSES + (
    "emptied",
    "filled",
    "empty_form",
    "numeric",
    "type_change",
    "value",
    "key_order",
)


@dataclass
class FieldChange:
    field: str
    edit_class: str
    old: Any = MISSING
    new: Any = MISSING
    delta: float | None = None
    similarity: float | None = None
    word_diff: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"field": self.field, "class": self.edit_class}
        if self.old is not MISSING:
            out["old"] = self.old
        if self.new is not MISSING:
            out["new"] = self.new
        if self.delta is not None:
            out["delta"] = self.delta
        if self.similarity is not None:
            out["similarity"] = round(self.similarity, 4)
        if self.word_diff is not None:
            out["word_diff"] = self.word_diff
        return out


def is_empty(value: Any) -> bool:
    if value is MISSING or value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def word_similarity(a: str, b: str) -> float:
    """Ratio of matching words (difflib), 1.0 for two empty strings."""
    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    return SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def word_diff(a: str, b: str, context: int = 4) -> str:
    """A word-level diff in the ``[-removed-]{+added+}`` style of ``git diff --word-diff``.

    Runs of unchanged words longer than ``2 * context`` are shortened to ``...``.
    """
    wa, wb = words(a), words(b)
    parts: list[str] = []
    ops = SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
    for n, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            run = wa[i1:i2]
            head = run[:context] if n > 0 else []
            tail = run[-context:] if n < len(ops) - 1 else []
            if len(run) > len(head) + len(tail):
                parts += head + (["..."] if head or tail else []) + tail
            else:
                parts += run
            continue
        if tag in ("delete", "replace"):
            parts.append("[-" + " ".join(wa[i1:i2]) + "-]")
        if tag in ("insert", "replace"):
            parts.append("{+" + " ".join(wb[j1:j2]) + "+}")
    return " ".join(parts)


def suffix_diff(a: str, b: str, context: int = 30) -> str:
    """Show a truncation or extension as the shared start plus the cut or added tail."""
    head = commonprefix([a, b])
    common = len(head)
    if len(head) > context:
        head = "..." + head[-context:]
    if len(a) > len(b):
        return f"{head}[-{a[common:]}-]"
    return f"{head}{{+{b[common:]}+}}"


def classify_text(a: str, b: str, rewrite_threshold: float = 0.5) -> str:
    ca, cb = collapse_ws(a), collapse_ws(b)
    if ca == cb:
        return "whitespace"
    na, nb = unicodedata.normalize("NFKC", ca), unicodedata.normalize("NFKC", cb)
    if na == nb:
        return "unicode"
    if na.casefold() == nb.casefold():
        return "case"
    if loose_text(a) == loose_text(b):
        return "punctuation"
    if len(cb) < len(ca) and ca.startswith(cb):
        return "truncation"
    if len(ca) < len(cb) and cb.startswith(ca):
        return "extension"
    if len(cb.split()) == 1 and len(ca.split()) == 1:
        return "relabel"
    if word_similarity(a, b) < rewrite_threshold:
        return "rewrite"
    return "revision"


def _vjson(value: Any, norm: Normalization) -> str:
    return json.dumps(
        norm_value(value, norm),
        sort_keys=not norm.key_order,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def compare_values(
    name: str,
    old: Any,
    new: Any,
    norm: Normalization = STRICT,
    rewrite_threshold: float = 0.5,
) -> FieldChange | None:
    """The change from ``old`` to ``new`` in one field, or None if they are equal."""
    if old is not MISSING and new is not MISSING and _vjson(old, norm) == _vjson(new, norm):
        return None
    if is_empty(old) and is_empty(new):
        return FieldChange(name, "empty_form", old, new)
    if is_empty(new):
        return FieldChange(name, "emptied", old, new)
    if is_empty(old):
        return FieldChange(name, "filled", old, new)
    if isinstance(old, str) and isinstance(new, str):
        a, b = norm_value(old, norm), norm_value(new, norm)
        cls = classify_text(a, b, rewrite_threshold)
        if cls == "whitespace":
            shown = None  # a word diff of a whitespace edit is empty; show the values
        elif cls in ("truncation", "extension"):
            shown = suffix_diff(collapse_ws(a), collapse_ws(b))
        else:
            shown = word_diff(a, b)
        return FieldChange(name, cls, old, new, similarity=word_similarity(a, b), word_diff=shown)
    if is_number(old) and is_number(new):
        if type(old) is not type(new) and old == new:
            return FieldChange(name, "type_change", old, new)
        return FieldChange(name, "numeric", old, new, delta=new - old)
    if type(old) is not type(new):
        return FieldChange(name, "type_change", old, new)
    return FieldChange(name, "value", old, new)


def compare_records(
    old: dict[str, Any],
    new: dict[str, Any],
    norm: Normalization = STRICT,
    rewrite_threshold: float = 0.5,
) -> list[FieldChange]:
    """Field changes between two versions of a record, in first-seen field order."""
    ignore = set(norm.ignore_fields)
    names = [k for k in old if k not in ignore]
    names += [k for k in new if k not in ignore and k not in old]
    changes = []
    for name in names:
        change = compare_values(
            name, old.get(name, MISSING), new.get(name, MISSING), norm, rewrite_threshold
        )
        if change is not None:
            changes.append(change)
    if not changes and norm.key_order:
        kept_old = [k for k in old if k not in ignore]
        kept_new = [k for k in new if k not in ignore]
        if kept_old != kept_new:
            changes.append(FieldChange("", "key_order", kept_old, kept_new))
    return changes
