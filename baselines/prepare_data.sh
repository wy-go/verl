#!/usr/bin/env bash
# Preprocess datasets for the verl baseline into parquet files.
#   GSM8K -> $DATA_DIR/gsm8k/{train,test}.parquet   (always)
#   MATH  -> $DATA_DIR/math/{train,test}.parquet    (only when DATASET=gsm8k_math)
#
# Env knobs:
#   DATASET   gsm8k | gsm8k_math   (default: gsm8k — Phase-1 default; MATH not needed)
#   DATA_DIR  destination root     (default: ~/data)
set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-gsm8k}"
DATA_DIR="${DATA_DIR:-$HOME/data}"

python examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR/gsm8k"

# MATH is only needed for DATASET=gsm8k_math. It pulls a third-party HF mirror
# (DigitalLearningGmbH/MATH-lighteval) that has been flaky; skip it by default
# so a Phase-1 (gsm8k) prep never aborts on an unrelated MATH download failure.
case "$DATASET" in
  gsm8k) ;;
  gsm8k_math)
    python examples/data_preprocess/math_dataset.py --local_save_dir "$DATA_DIR/math"
    ;;
  *) echo "DATASET must be gsm8k or gsm8k_math, got: $DATASET" >&2; exit 1 ;;
esac

echo "--- prepared data ---"
ls -la "$DATA_DIR/gsm8k"
if [ "$DATASET" = gsm8k_math ]; then
  ls -la "$DATA_DIR/math"
fi
