# Changelog

## 0.1.0

- Readers for JSONL, JSON arrays, CSV and TSV, and Parquet through the `parquet` extra.
- Alignment by key, by canonical content hash (key order, whitespace, Unicode and case options), by a loose hash,
  and by token-set Jaccard similarity with a rarest-token blocking index (`--block-tokens`). Reordering is measured
  with a longest increasing subsequence.
- Record statuses: unchanged, edited, added, removed, duplicated, deduplicated. Field edit classes by fixed rules,
  word-level diffs and numeric deltas. Fields present in one version only are reported as schema changes.
- Schema report: fields added or removed, type set changes, and null and blank rate changes.
- Drift report: label transition matrix and flips, Jensen-Shannon divergence, KS test and PSI on text length and
  numeric values, and the unigram Jensen-Shannon divergence for text.
- Commands `diff`, `schema`, `drift` and `show`; Markdown and JSON reports, per-record JSONL, seeded example
  sampling, and threshold exit codes for CI.
