#!/usr/bin/env bash
# Download the two public dataset pairs used in the README, each pinned to a revision.
# Usage: bash examples/fetch_data.sh [target-dir]   (default: ./data, about 65 MB)
set -euo pipefail
dest="${1:-data}"
mkdir -p "$dest/gsm8k" "$dest/alpaca"

hf="https://huggingface.co/datasets"
# GSM8K test split (MIT) and GSM8K-Platinum (MIT; annotations CC BY-SA 4.0).
curl -sSL -o "$dest/gsm8k/gsm8k_test.parquet" \
  "$hf/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"
curl -sSL -o "$dest/gsm8k/platinum_test.parquet" \
  "$hf/madrylab/gsm8k-platinum/resolve/e762492455a1cf7967de89f05b6bef72fc713b66/main/test-00000-of-00001.parquet"

# Stanford Alpaca (CC BY-NC 4.0) and the community-cleaned version (see its card for the licence).
curl -sSL -o "$dest/alpaca/alpaca_data.json" \
  "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/f13496211289def0ff88ae673389ae14a818b4b3/alpaca_data.json"
curl -sSL -o "$dest/alpaca/alpaca_data_cleaned.json" \
  "$hf/yahma/alpaca-cleaned/resolve/12567cabf869d7c92e573c7c783905fc160e9639/alpaca_data_cleaned.json"

echo "downloaded to $dest"
