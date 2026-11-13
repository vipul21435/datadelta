# datadelta

Align the records of two versions of a dataset and report what was added, removed and edited, field by field.

Status: early. The `diff` command works; schema and drift reports are not written yet (see
[What works so far](#what-works-so-far)).

## Why this is hard

Between two releases of a dataset, records move, are edited slightly, and lose or gain fields. A line diff of two
JSONL files marks a reordered file as entirely changed and cannot say which field changed or how; a row count hides
every edit that keeps the count the same. Without a shared key, the old and new copies of a record have to be paired
by content, which means deciding when an edited record is still the same record, and doing it without comparing
every old record with every new one.

## Install

```
pip install git+https://github.com/vipul21435/datadelta
```

Python 3.10 or later, no runtime dependencies. For Parquet input add the optional extra, which brings in pyarrow:
`pip install "datadelta[parquet] @ git+https://github.com/vipul21435/datadelta"`.

## Quickstart

The repository ships a ten-record example pair. With `id` as the key:

```
$ datadelta diff examples/data/quiz_v1.jsonl examples/data/quiz_v2.jsonl --key id --examples 1
# datadelta report

old: `examples/data/quiz_v1.jsonl` (10 records); new: `examples/data/quiz_v2.jsonl` (10 records)

Alignment: key `id`, then content

| status | records |
|---|---:|
| unchanged | 2 |
| edited | 7 |
| added | 1 |
| removed | 1 |
| duplicated | 0 |
| deduplicated | 0 |
| moved (order) | 1 |

Pairs found by: key 9, hash 0, loose 0, fuzzy 0.

## Edit classes

| class | edited records |
|---|---:|
| relabel | 2 |
| case | 1 |
| truncation | 1 |
| numeric | 1 |
| emptied | 1 |
| whitespace | 1 |
...

### truncation

- id=6 old[5] -> new[5] by key: `answer` truncation: Photosynthesis[-, which uses sunlight, water and carbon dioxide.-]

### numeric

- id=7 old[6] -> new[9] by key: `score` numeric: 0.9 -> 0.75 (-0.15)
...
```

(Output shortened at the two `...` lines.) Without `--key` the same pair gives the same counts, with the pairs found
by content instead: 2 by exact hash, 2 by loose hash, 5 by token similarity.

Other outputs: `--format json` for the summary as JSON, `--changes changes.jsonl` for one JSON line per changed
record, and `--max-removed N`, `--max-added N`, `--max-edited-share X` to exit with status 1 when a limit is crossed
(status 2 means the input could not be read).

## How it works

**Alignment** runs a cascade; each step sees only the records the previous steps left unpaired.

1. `--key a,b`: records with equal key values are the same record. Records missing a key field fall through to the
   content steps.
2. Exact content: equal canonical JSON (keys sorted unless `--keep-key-order`, optional `--normalize-whitespace`,
   `--unicode NFC|NFKC`, `--ignore-case`, `--ignore-field F`). Repeated content is paired first copy with first copy.
3. Loose content: equal after every normalisation at once (whitespace, NFKC, case folding, punctuation removed).
4. Token similarity: Jaccard similarity of the records' word sets, at least `--min-similarity` (default 0.5), paired
   greedily from the most similar pair down. Candidates come from an inverted index over only the rarest
   |x| - ceil(t |x|) + 1 tokens of each set x: two sets with Jaccard at least t share at least ceil(t |x|) tokens, so
   they must share one of those, and no pair above the threshold is missed. The tests check this against a
   brute-force scan. Indexing only part of each vector is the idea in Bayardo, Ma and Srikant, "Scaling Up All Pairs
   Similarity Search" (WWW 2007).

Ties are broken by rules that do not depend on which file is called old, so `diff(B, A)` is the mirror of
`diff(A, B)`: what is added one way is removed the other way (a property test checks this on random data).
Reordering is measured as the number of paired records outside one longest order-preserving subsequence.

**Record status**: `unchanged`, `edited`, `added`, `removed`, `duplicated` (an unpaired new record whose content
equals another new record) and `deduplicated` (the same on the old side).

**Edit classes** for a changed field. Text classes are cumulative: each is the smallest normalisation that makes the
two strings equal, tried in this order.

| class | rule |
|---|---|
| whitespace | equal after collapsing whitespace |
| unicode | ... and NFKC |
| case | ... and case folding |
| punctuation | ... and removing Unicode punctuation |
| truncation | new text is a strict prefix of the old |
| extension | old text is a strict prefix of the new |
| relabel | both texts are a single word |
| rewrite | word-level similarity below `--rewrite-threshold` (default 0.5) |
| revision | any other text edit |
| emptied / filled | a value became empty (missing, null, "", [] or {}) / the reverse |
| empty_form | one empty form became another, such as missing to null |
| numeric | a number changed; the delta is reported |
| type_change | the type changed, such as "3" to 3 |
| value | any other change, such as a list or a boolean |

## What works so far

- Reading JSONL, JSON arrays (also `{"data": [...]}` wrappers), CSV and TSV with number inference; Parquet through
  the optional extra (not yet tested).
- The alignment cascade, record statuses, field-level edit classes, word diffs and numeric deltas.
- `datadelta diff` with Markdown and JSON reports, a per-record JSONL file and threshold exit codes.

Not implemented yet: the `schema`, `drift` and `show` commands; schema changes (fields added or removed, type and
null-rate changes); field drift statistics (label transition matrices, KS test and population stability index on
text lengths, Jensen-Shannon divergence of token unigrams, numeric quantile shifts); randomly sampled examples (the
report shows the first records of each class); results on real public data; a measured runtime.

## Related tools

- [datasieve](https://github.com/vipul21435/datasieve) finds duplicates within one dataset.
- [sftlint](https://github.com/vipul21435/sftlint) lints one chat or tool-calling dataset.

## Licence

MIT. See [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
