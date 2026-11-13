# datadelta

Align the records of two versions of a dataset and report what was added, removed and edited, field by field,
with the distribution shift that followed.

## Why this is hard

Between two releases of a dataset, records move, are edited slightly, and lose or gain fields. A line diff of two
JSONL files marks a reordered file as entirely changed and cannot say which field changed or how; a row count hides
every edit that keeps the count the same. Without a shared key, the old and new copies of a record have to be paired
by content, which means deciding when an edited record is still the same record, and doing it without comparing
every old record with every new one. The "Maintenance" questions of Gebru et al.'s *Datasheets for Datasets* ask
whether a dataset will be updated "to correct labeling errors, add new instances, delete instances". Answering that
for a release that has already happened takes exactly this alignment.

## Install

```
pip install git+https://github.com/vipul21435/datadelta
```

Python 3.10 or later, no runtime dependencies. For Parquet input add the optional extra, which brings in pyarrow:
`pip install "datadelta[parquet] @ git+https://github.com/vipul21435/datadelta"`.

## Quickstart

The repository ships a ten-record example pair (`examples/data/`). With `id` as the key:

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
## Drift

Categorical fields (flips counted over paired records):

| field | categories | JS divergence | paired | flips | flip share | top flips |
|---|---:|---:|---:|---:|---:|---|
| topic | 9 | 0.157 | 9 | 1 | 0.111 | biology -> zoology (1) |
...
### truncation

- id=6 old[5] -> new[5] by key: `answer` truncation: Photosynthesis[-, which uses sunlight, water and carbon dioxide.-]
...
```

(Shortened at each `...`.) Without `--key` the same pair gives the same counts, with the pairs found by content
instead: 2 by exact hash, 2 by loose hash, 5 by token similarity.

The four commands:

| command | what it prints |
|---|---|
| `datadelta diff OLD NEW` | record statuses, edit classes, fields changed, schema changes, drift and sampled examples |
| `datadelta schema OLD NEW` | fields added or removed, value types changed, null and blank rates that moved |
| `datadelta drift OLD NEW` | per-field distribution shift only |
| `datadelta show OLD NEW --key-value V` | one record pair (or `--old-index`, `--new-index`) with its word diffs |

`--format json` gives the same content as JSON; `diff --changes changes.jsonl` writes one JSON line per changed
record; `--examples N --seed S` sets how many examples per class are sampled and with which seed. For CI, `diff`
exits 1 when a limit is crossed: `--max-removed N`, `--max-added N`, `--max-edited-share X`, `--max-flip-share X`
(any categorical field), `--max-psi X` (any numeric or text-length field); `schema --fail-on-change` exits 1 on any
schema change. Exit status 2 means the input could not be read. `examples/quickstart.py` shows the Python API.

## How it works

**Reading.** JSONL, a JSON array (or an object holding one list of records), CSV and TSV (cells that look like
numbers become numbers, empty cells become null; `--no-infer-types` keeps strings), and Parquet with the extra.

**Alignment** is a cascade; each step sees only the records the previous steps left unpaired.

1. `--key a,b`: equal key values mean the same record. This is deterministic, rule-based linkage in record-linkage
   terms. Records missing a key field fall through to the content steps; if a key value repeats, its copies are
   paired by content and then by position.
2. Exact content: equal canonical JSON (keys sorted unless `--keep-key-order`; optional `--normalize-whitespace`,
   `--unicode NFC|NFKC`, `--ignore-case`, `--ignore-field F`). Repeated content pairs first copy with first copy.
3. Loose content: equal after every normalisation at once (whitespace, NFKC, case folding, punctuation removed).
4. Token similarity: the Jaccard similarity |x ∩ y| / |x ∪ y| of the records' word sets, at least
   `--min-similarity` t (default 0.5), paired greedily from the most similar pair down. Comparing every pair is
   quadratic, so candidates come from a blocking index, as in record linkage. Tokens are ranked rarest first; two
   sets with Jaccard at least t share at least ceil(t |x|) tokens, so they must share one of the first
   |x| - ceil(t |x|) + 1 rarest tokens of x. Indexing only that part of each set is exact (the tests check it
   against a brute-force scan); the idea of indexing part of each vector comes from Bayardo, Ma and Srikant,
   "Scaling Up All Pairs Similarity Search" (WWW 2007). On long records that prefix is long and the candidate lists
   grow very large. With the exact index (`--block-tokens 0`), the Alpaca content run below was still running after
   2.5 minutes when stopped. By default `--block-tokens 8` indexes at most the 8 rarest tokens per record. That
   bounds the work, but it can miss a qualifying pair whose shared tokens all lie outside both capped prefixes.

Ties are broken by rules that do not depend on which file is called old, so `diff(B, A)` mirrors `diff(A, B)`.
Property tests check on random data that what is added one way is removed the other way, that edited pairs mirror,
and that `diff(A, A)` is empty. Reordering is counted as the number of paired records outside one longest
increasing subsequence of new positions. A field that appears in only one version is reported as a schema change
and left out of record comparison, because otherwise a new column would mark every record as edited
(`--schema-fields-as-edits` turns this off).

**Record status**: `unchanged`, `edited`, `added`, `removed`, `duplicated` (an unpaired new record whose content
equals another new record) and `deduplicated` (the same on the old side).

**Edit classes** for a changed field. Text classes are cumulative: each is the smallest normalisation that makes the
two strings equal, tried in this order.

| class | rule |
|---|---|
| whitespace | equal after collapsing whitespace |
| unicode | ... and NFKC |
| case | ... and case folding |
| punctuation | ... and removing Unicode punctuation (categories P*) |
| truncation | new text is a strict prefix of the old |
| extension | old text is a strict prefix of the new |
| relabel | both texts are a single word |
| rewrite | word-level similarity (difflib ratio over words) below `--rewrite-threshold` (default 0.5) |
| revision | any other text edit |
| emptied / filled | a value became empty (missing, null, blank, [] or {}) / the reverse |
| empty_form | one empty form became another, such as missing to null |
| numeric | a number changed; the delta is reported |
| type_change | the type changed, such as "3" to 3 or 1 to 1.0 |
| value | any other change, such as a list or a boolean |

Text diffs are word-level, in the `[-removed-]{+added+}` style of `git diff --word-diff`; truncations and
extensions show the cut or added tail.

**Schema.** Per field and version: presence, null rate (missing or null, over all records), blank-string rate and
the set of value types. Reported: fields added or removed, any change in the type set, and null or blank rates that
moved by more than `--rate-tolerance` (default 0.01).

**Drift**, per field present in both versions (key fields excluded):

- *Categorical* (strings, booleans or integers with at most `--max-categories` distinct values, no more distinct
  values than half the values, no string over 40 characters): value counts, the Jensen-Shannon divergence of the
  two distributions, and the transition matrix over paired records, whose off-diagonal cells are label flips.
- *Text* (other strings): character-length summary (mean, quantiles), the two-sample Kolmogorov-Smirnov test and
  PSI on lengths, and the Jensen-Shannon divergence of the token unigram distributions.
- *Numeric* (other numbers): mean, standard deviation, 5/25/50/75/95 % quantiles (linear interpolation), KS and PSI.

The formulas:

- Kolmogorov-Smirnov: D = max over v of |F_old(v) - F_new(v)| for the two empirical distribution functions. The
  p-value is the large-sample one, P(K > sqrt(nm / (n + m)) D), with K Kolmogorov distributed:
  P(K > x) = 2 Σ_k (-1)^(k-1) exp(-2 k² x²), or 1 - (sqrt(2π) / x) Σ_k exp(-(2k-1)² π² / (8 x²)) for small x (both
  forms appear on Wikipedia's Kolmogorov-Smirnov test page). The tests match scipy's `ks_2samp` statistic on tied
  data, and scipy's `kstwobign.sf` to 1e-10. scipy also computes exact p-values for small samples (its
  documentation cites Hodges 1958); datadelta does not.
- Population stability index: bins are the deciles of the old values (repeated edges merged), p and q are the old
  and new bin shares floored at 1e-4, and PSI = Σ (q - p) ln(q / p). Yurdakul's thesis *Statistical Properties of
  Population Stability Index* (Western Michigan University, 2018) describes PSI as "a statistic that measures how
  much a variable has shifted over time" and discusses the rule of thumb that values under 0.1 mean little shift.
- Jensen-Shannon divergence, in bits: JS(P, Q) = H(M) - (H(P) + H(Q)) / 2 with M = (P + Q) / 2, between 0 and 1
  (Lin, "Divergence measures based on the Shannon entropy", IEEE Transactions on Information Theory 37(1), 1991).
  The tests compare it with the square of scipy's `jensenshannon`.

## Results on public data

Two pairs, fetched at pinned revisions with `bash examples/fetch_data.sh data`. No rows from either dataset are
committed here. Runtimes are wall-clock for one process on a laptop, reading included.

**GSM8K test split -> GSM8K-Platinum.** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (MIT licence),
`main` test split, 1,319 records. [GSM8K-Platinum](https://huggingface.co/datasets/madrylab/gsm8k-platinum)
(tagged MIT; its card says the added annotations are CC BY-SA 4.0) is a cleaned copy of that split. Its card
reports 110 questions rejected and 10 relabelled, and it adds a `cleaning_status` column.

```
datadelta diff data/gsm8k/gsm8k_test.parquet data/gsm8k/platinum_test.parquet     # 0.56 s
```

| status | records |
|---|---:|
| unchanged | 1,199 |
| edited | 10 (all in `answer`: 9 revision, 1 rewrite) |
| removed | 110 |
| added, duplicated, deduplicated, moved | 0 |

No key was given: 1,199 pairs were found by exact hash and 10 by token similarity. The schema section reports
`cleaning_status` as an added field. `python examples/gsm8k_platinum_check.py data/gsm8k` checks the result against
Platinum's own labels: the 10 edited records are exactly the 10 marked `revised`, and the 1,199 unchanged ones are
the 1,100 marked `consensus` and the 99 marked `verified`. Length drift is negligible (question KS D 0.008, PSI
0.0006; answer KS D 0.014, PSI 0.001).

**Stanford Alpaca -> alpaca-cleaned.** [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
`alpaca_data.json`, 52,002 records; its README says "The dataset is CC BY NC 4.0 (allowing only non-commercial
use)". [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) `alpaca_data_cleaned.json`, 51,760
records; its card metadata says CC BY 4.0, while its licensing section says CC BY-NC 4.0. Alpaca's README says each
instruction is unique, so `instruction` serves as the key:

```
datadelta diff data/alpaca/alpaca_data.json data/alpaca/alpaca_data_cleaned.json --key instruction    # 33.7 s
```

| status | records |
|---|---:|
| unchanged | 1,832 |
| edited | 48,064 |
| added | 1,860 |
| removed | 2,106 |
| duplicated | 4 |

Pairs: 49,895 by key, plus 1 by content within a repeated instruction. Edit classes (records; a record can have
two): rewrite 43,438, revision 3,521, emptied 596, punctuation 564, relabel 213, extension 139, case 83,
whitespace 46, truncation 40, filled 13, unicode 1. By field, `output` carries the rewrites (43,438) and the
revisions (3,520), and `input` carries the 596 emptied values. The schema section reports the blank `input` rate
moving from 0.602 to 0.630. The drift section shows the outputs getting much longer: mean length 270.3 -> 678.9
characters, median 185 -> 477, KS D 0.339, PSI 0.711. For `input`, KS D is 0.028 and PSI 0.008.

The same pair aligned by content alone, the harder case:

```
datadelta diff data/alpaca/alpaca_data.json data/alpaca/alpaca_data_cleaned.json --format json    # 43.6 s
```

This finds 1,832 pairs by exact hash, 692 by loose hash and 12,852 by token similarity, and leaves 36,626 removed
and 36,380 added. Most rewritten outputs share fewer than half their words with the original, so whole-record
Jaccard falls under 0.5. When records have a stable identifier, `--key` is the right tool. Content alignment is for
when they do not, and its result depends on `--min-similarity`.

**Synthetic scale.** `tests/test_scale.py` builds 100,000 records (2 % removed, 7 % edited or relabelled, 2 % added,
shuffled), diffs them with and without a key, and checks that both give the same counts. The whole test, data
generation included, took 8.5 s (`pytest tests/test_scale.py --durations=1`).

## Limitations

- KS p-values are asymptotic, with no exact small-sample computation and no tie correction, and text lengths are
  often heavily tied.
- PSI bins come from the old version's deciles. When the old values are nearly constant, almost everything falls
  in one bin and PSI can be near 0 while KS shows a clear shift; the quickstart's `score` field does this (KS D 1,
  PSI 0).
- The capped blocking index (`--block-tokens 8`) can miss qualifying pairs on long records, and the exact index can
  be very slow on large pools of long, similar records.
- Greedy pairing is not a maximum-weight matching.
- Field comparison is top-level: a change inside a nested list or object is one `value` edit, with no path into it.
  There is no notion of a renamed field either: a rename shows as one field removed and one added.
- Categorical detection is the fixed rule described under Drift; `--max-categories` is its only control.
- Both versions are held in memory as Python objects. The largest input tried is the 44 MB alpaca-cleaned file.

## Related tools

- [datasieve](https://github.com/vipul21435/datasieve) finds duplicates within one dataset; datadelta compares two
  versions.
- [sftlint](https://github.com/vipul21435/sftlint) lints one chat or tool-calling dataset.

## Licence

MIT. See [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
