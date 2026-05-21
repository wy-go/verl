#!/usr/bin/env bash
# Phase-1 verl baseline launcher — Qwen3-8B GRPO on 8×H100 (FSDP + sglang).
#
# Thin, opinionated wrapper around examples/grpo_trainer/run_qwen3_8b_fsdp.sh.
# Pins the rollout backend to sglang because vLLM 0.8.5 is ABI-broken against
# torch 2.8 on this box (see baselines/README.md, section "Known issues").
#
# Env knobs:
#   DATASET        gsm8k | gsm8k_math        (default: gsm8k — matches verl ref log)
#   SMOKE          1 = fast 1-epoch pipeline check, no checkpoints (default: 0)
#   INFER_BACKEND  vllm | sglang | trtllm    (default: sglang)
#   DATA_DIR       dataset root              (default: ~/data)
#   TOTAL_EPOCHS / SAVE_FREQ / TEST_FREQ / MODEL_PATH / EXPERIMENT_NAME ... pass through
#
# Extra Hydra overrides may be appended as CLI args, e.g.:
#   baselines/run_qwen3_8b_grpo.sh actor_rollout_ref.rollout.n=8
set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET=${DATASET:-gsm8k}
SMOKE=${SMOKE:-0}
export INFER_BACKEND=${INFER_BACKEND:-sglang}
DATA_DIR="${DATA_DIR:-$HOME/data}"

LOG_DIR="$REPO_ROOT/baselines/logs"
mkdir -p "$LOG_DIR"

# ---- dataset selection -------------------------------------------------
# Build data.*_files explicitly for BOTH cases so DATA_DIR is always honored —
# the canonical script otherwise hard-codes $HOME/data for the gsm8k+math path.
# gsm8k     : GSM8K only — cleanest reproduction of the verl reference log
# gsm8k_math: GSM8K + MATH
DATA_ARGS=()
case "$DATASET" in
  gsm8k)
    DATA_ARGS=(
      data.train_files="['$DATA_DIR/gsm8k/train.parquet']"
      data.val_files="['$DATA_DIR/gsm8k/test.parquet']"
    )
    ;;
  gsm8k_math)
    DATA_ARGS=(
      data.train_files="['$DATA_DIR/gsm8k/train.parquet', '$DATA_DIR/math/train.parquet']"
      data.val_files="['$DATA_DIR/gsm8k/test.parquet', '$DATA_DIR/math/test.parquet']"
    )
    ;;
  *) echo "DATASET must be gsm8k or gsm8k_math, got: $DATASET" >&2; exit 1 ;;
esac

# ---- smoke-test shortcut ----------------------------------------------
if [ "$SMOKE" = 1 ]; then
  export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
  export SAVE_FREQ=${SAVE_FREQ:--1}   # -1 = never save (avoids ~80 GB ckpts)
  export TEST_FREQ=${TEST_FREQ:-2}
fi

# ---- naming & logging --------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_grpo_${DATASET}_${INFER_BACKEND}_${TS}}
export TENSORBOARD_DIR="$LOG_DIR/tb/$EXPERIMENT_NAME"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"

echo "experiment : $EXPERIMENT_NAME"
echo "dataset    : $DATASET   rollout: $INFER_BACKEND   smoke: $SMOKE"
echo "console log: $LOG_FILE"
echo "tensorboard: $TENSORBOARD_DIR"

# ---- launch ------------------------------------------------------------
# Later Hydra args win, so these override the canonical script's defaults.
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  trainer.logger='["console","tensorboard"]' \
  trainer.max_actor_ckpt_to_keep=1 \
  "${DATA_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
