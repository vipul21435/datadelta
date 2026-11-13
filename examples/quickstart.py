"""Use datadelta from Python on the bundled ten-record example.

Run from the repository root:  python examples/quickstart.py
"""

from pathlib import Path

from datadelta.diff import diff
from datadelta.drift import drift
from datadelta.io import read_records
from datadelta.schema import schema_diff

here = Path(__file__).resolve().parent / "data"
old = read_records(here / "quiz_v1.jsonl")
new = read_records(here / "quiz_v2.jsonl")

result = diff(old, new, key=["id"])
print("status counts:", result.counts())
print("edit classes:", result.class_counts())
print("records moved:", result.moved_count)

for change in result.changes:
    if change.status == "edited":
        for f in change.fields:
            print(f"  id={change.key['id']}: {f.field} {f.edit_class}")

print("schema changes:", [(c.field, c.change) for c in schema_diff(old, new).changes])
for d in drift(old, new, result, exclude=["id"]):
    if d.kind == "categorical":
        print(f"{d.field}: {d.stats['flips']} of {d.stats['paired']} paired records changed label")
