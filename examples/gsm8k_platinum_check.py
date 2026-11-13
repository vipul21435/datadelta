"""Check datadelta against the GSM8K-Platinum annotations.

GSM8K-Platinum marks each kept question as consensus, verified or revised. A content
diff from the GSM8K test split should report the revised ones as edited and the others
as unchanged. Fetch the data first with examples/fetch_data.sh, then run:

    python examples/gsm8k_platinum_check.py data/gsm8k
"""

import sys
import time
from collections import Counter
from pathlib import Path

from datadelta.diff import diff
from datadelta.io import read_records

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/gsm8k")
old = read_records(folder / "gsm8k_test.parquet")
new = read_records(folder / "platinum_test.parquet")
start = time.perf_counter()
result = diff(old, new)
seconds = time.perf_counter() - start

print(f"{len(old)} -> {len(new)} records, diff in {seconds:.2f} s")
print("status:", result.counts())
for status in ("unchanged", "edited"):
    tags = Counter(
        new[c.new_index]["cleaning_status"] for c in result.changes if c.status == status
    )
    print(f"{status} records by cleaning_status: {dict(tags)}")
