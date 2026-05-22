#!/usr/bin/env bash
# Phase-1 verl baseline launcher — Qwen3-8B GRPO on 8×H100 (FSDP + sglang).
#
# Thin, opinionated wrapper around examples/grpo_trainer/run_qwen3_8b_fsdp.sh.
# Pins the rollout backend to sglang because vLLM 0.8.5 is ABI-broken against
# torch 2.9.1 on this box (see baselines/README.md, section "Known issues").
# Run baselines/setup_env.sh once before using this launcher.
#
# Env knobs:
#   DATASET        gsm8k | gsm8k_math        (default: gsm8k — matches verl ref log)
#   SMOKE          1 = fast 1-epoch pipeline check, no checkpoints (default: 0)
#   INFER_BACKEND  vllm | sglang | trtllm    (default: sglang)
#   DATA_DIR       dataset root              (default: ~/data)
#   WANDB_MODE     offline | online          (default: offline; online needs WANDB_API_KEY)
#   WANDB_PROJECT  wandb project name        (default: verl-baselines)
#   TOTAL_EPOCHS / SAVE_FREQ / TEST_FREQ / MODEL_PATH / EXPERIMENT_NAME ... pass through
#
# Auto-applies the HF-Xet and CUDA-libcudart workarounds (see README "Known issues").
# Extra Hydra overrides may be appended as CLI args, e.g.:
#   baselines/run_qwen3_8b_grpo.sh actor_rollout_ref.rollout.n=8
set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# HF Xet downloads bypass HF_ENDPOINT and hang on this box; force classic LFS.
export HF_HUB_DISABLE_XET=1

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
WANDB_PROJECT=${WANDB_PROJECT:-verl-baselines}
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"
# Checkpoints: verl's trainer.default_local_dir defaults to a RELATIVE path
# ("checkpoints/..."), which the Ray worker resolves against its own CWD
# (/usr/bin on this box) -> PermissionError on the first save. Pin it absolute.
CKPT_DIR="$REPO_ROOT/baselines/checkpoints/$EXPERIMENT_NAME"

# wandb: this branch logs to the byted-wandb fork (ByteDance wandb dashboard).
# byted-wandb is protobuf-6-incompatible out of the box; setup_env.sh applies
# baselines/patches/fix_bytedwandb_protobuf6.py to fix it. Logging mode:
#   WANDB_MODE=offline (default) -> writes $WANDB_DIR; push later: `wandb sync`
#   WANDB_MODE=online            -> streams live; needs WANDB_API_KEY exported
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="$LOG_DIR/wandb"
mkdir -p "$WANDB_DIR"

echo "experiment : $EXPERIMENT_NAME"
echo "dataset    : $DATASET   rollout: $INFER_BACKEND   smoke: $SMOKE"
echo "console log: $LOG_FILE"
echo "wandb      : project=$WANDB_PROJECT  mode=$WANDB_MODE  dir=$WANDB_DIR"
echo "checkpoints: $CKPT_DIR"

# ---- env propagation to Ray workers -----------------------------------
# Ray actors (incl. the one that downloads the model) run on a possibly
# pre-existing cluster and only see env vars listed in runtime_env.env_vars
# — they do NOT inherit this shell's `export`s. Forward what they need:
#
#  * CUDA libcudart: torch 2.9.1 bundles CUDA 12.8 libs but the system
#    toolkit is 12.4. Without this the sglang scheduler subprocess loads the
#    wrong libcudart. Prepend torch's cu12 libs to LD_LIBRARY_PATH.
#  * HF Xet: without HF_HUB_DISABLE_XET the actor doing the Qwen3-8B pull
#    uses the Xet CAS protocol and hangs on an unreachable CAS server
#    (README "Known issues" #1). HF_ENDPOINT keeps it on the BD mirror.
#  * WANDB_*: verl's Tracking() runs in the actor, so the wandb backend
#    reads WANDB_MODE / WANDB_DIR / WANDB_API_KEY from the actor's os.environ
#    — they must be forwarded here or wandb logging misbehaves silently.
TORCH_CUDA_LIBS=$(python3 -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(d for d in glob.glob(b+'/*/lib') if 'cu13' not in d)))")
export LD_LIBRARY_PATH="${TORCH_CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
# HF_HUB_DISABLE_XET is quoted ('1') so Hydra keeps it a string — Ray's
# runtime_env.env_vars rejects non-string values (an unquoted 1 is an int).
RAY_ARGS=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_DISABLE_XET='1'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE=${WANDB_MODE}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=${WANDB_DIR}"
)
if [ -n "${HF_ENDPOINT:-}" ]; then
  RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.HF_ENDPOINT=${HF_ENDPOINT}")
fi
# Online wandb needs the API key (and optionally a custom host) in the actor;
# forward each only when set so offline runs stay zero-config.
if [ -n "${WANDB_API_KEY:-}" ]; then
  RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=${WANDB_API_KEY}")
fi
if [ -n "${WANDB_BASE_URL:-}" ]; then
  RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_BASE_URL=${WANDB_BASE_URL}")
fi

# ---- launch ------------------------------------------------------------
# Later Hydra args win, so these override the canonical script's defaults.
#
# FSDP param+optimizer offload is ON. With offload off (the canonical
# default) the actor holds ~65 GB/GPU after the first optimizer step, so the
# colocated sglang server cannot resume its KV-cache memory -> OOM at step 2
# (see README "Known issues"). Offloading to CPU frees the GPU between
# phases; measured throughput cost was negligible (~253 s/step smoke test).
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.default_local_dir="$CKPT_DIR" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  "${DATA_ARGS[@]}" \
  "${RAY_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
