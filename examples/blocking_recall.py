"""How many qualifying pairs does the capped blocking index miss?

Takes the records left after the exact and loose hash passes of a content diff, ranks
their tokens rarest first as the index does, and checks every pair that a key pairs and
that has token Jaccard >= the threshold: the capped index finds such a pair only if the
two records share a token inside both of their capped prefixes.

    python examples/blocking_recall.py OLD NEW --key instruction [--block-tokens 8]
"""

import argparse
import math
from collections import Counter, defaultdict

from datadelta.align import jaccard, key_of
from datadelta.io import read_records
from datadelta.normalize import loose_hash, record_hash, tokens

parser = argparse.ArgumentParser()
parser.add_argument("old")
parser.add_argument("new")
parser.add_argument("--key", required=True)
parser.add_argument("--block-tokens", type=int, default=8)
parser.add_argument("--min-similarity", type=float, default=0.5)
args = parser.parse_args()
key = args.key.split(",")
t = args.min_similarity

old, new = read_records(args.old), read_records(args.new)


def unpaired(hash_of, olds, news):
    """Indices left after multiset matching on ``hash_of`` (first copy with first copy)."""
    buckets = defaultdict(list)
    for j in news:
        buckets[hash_of(new[j])].append(j)
    used, left_old = set(), []
    for i in olds:
        bucket = buckets.get(hash_of(old[i]))
        if bucket:
            used.add(bucket.pop(0))
        else:
            left_old.append(i)
    return left_old, [j for j in news if j not in used]


olds, news = unpaired(record_hash, range(len(old)), range(len(new)))
olds, news = unpaired(loose_hash, olds, news)
old_sets = {i: tokens(old[i]) for i in olds}
new_sets = {j: tokens(new[j]) for j in news}
df = Counter()
for s in list(old_sets.values()) + list(new_sets.values()):
    df.update(s)
rank = {tok: r for r, tok in enumerate(sorted(df, key=lambda tok: (df[tok], tok)))}


def capped_prefix(s):
    ordered = sorted(s, key=rank.__getitem__)
    exact = len(ordered) - math.ceil(t * len(ordered) - 1e-9) + 1
    return set(ordered[: min(exact, args.block_tokens) if args.block_tokens else exact])


by_key = defaultdict(list)
for j in news:
    by_key[key_of(new[j], key)].append(j)
qualifying = missed = 0
for i in olds:
    k = key_of(old[i], key)
    if k is None:
        continue
    for j in by_key.get(k, ()):
        if old_sets[i] and new_sets[j] and jaccard(old_sets[i], new_sets[j]) >= t - 1e-9:
            qualifying += 1
            if not capped_prefix(old_sets[i]) & capped_prefix(new_sets[j]):
                missed += 1
share = missed / qualifying if qualifying else 0.0
print(f"left after hash passes: {len(olds)} old, {len(news)} new")
print(f"key-matched pairs with Jaccard >= {t}: {qualifying}")
print(f"missed by --block-tokens {args.block_tokens}: {missed} ({share:.2%})")
