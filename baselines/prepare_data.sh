#!/usr/bin/env bash
# Preprocess datasets for the Phase-1 verl baseline into parquet files.
#   GSM8K -> ~/data/gsm8k/{train,test}.parquet
#   MATH  -> ~/data/math/{train,test}.parquet   (needed only for DATASET=gsm8k_math)
#
# Override the destination root with DATA_DIR (default: ~/data).
set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$HOME/data}"

python examples/data_preprocess/gsm8k.py        --local_save_dir "$DATA_DIR/gsm8k"
python examples/data_preprocess/math_dataset.py --local_save_dir "$DATA_DIR/math"

echo "--- prepared data ---"
ls -la "$DATA_DIR/gsm8k" "$DATA_DIR/math"
