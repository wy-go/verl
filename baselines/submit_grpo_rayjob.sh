#!/usr/bin/env bash
# Run the Qwen3-8B GRPO baseline as a RAY JOB (recorded in the Ray Jobs UI),
# instead of a direct `bash run_qwen3_8b_grpo.sh` (which calls ray.init(address="auto")
# and leaves no Job record). The training itself is UNCHANGED — this only changes how
# it is launched, so the run shows up in the Ray dashboard's Jobs tab.
#
# Run this FROM the Ray cluster head (or any node that can reach the dashboard and is
# authorized to submit). The job DRIVER is pinned to a GPU worker via
# --entrypoint-resources, because the verl env + GPUs live on the worker (the head is
# CPU-only). verl's own ray.init() already injects LD_LIBRARY_PATH/HF/WANDB through its
# runtime_env, so do NOT also pass those keys via --runtime-env-json — Ray refuses to
# merge duplicate runtime_env keys (ValueError). Just submit clean.
#
# Note: sglang server launch on this box is intermittently flaky (see README §2
# "8 attempts"); if a submit dies at LLMServerManager/launch_servers, just resubmit.
#
# Usage:
#   baselines/submit_grpo_rayjob.sh smoke                                  # SMOKE=1, 1 epoch, no ckpt
#   baselines/submit_grpo_rayjob.sh full                                   # full run
#   baselines/submit_grpo_rayjob.sh smoke actor_rollout_ref.rollout.n=8    # + extra hydra overrides
#
# Env knobs:
#   RAY_ADDRESS      Ray dashboard (default http://127.0.0.1:8265; on bytedray use the head's dashboard port)
#   REPO_ROOT        verl repo root (default: parent dir of this script)
#   WORKER_RESOURCE  custom resource the GPU worker advertises (default: worker)
set -euo pipefail

MODE="${1:-smoke}"; shift || true
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKER_RESOURCE="${WORKER_RESOURCE:-worker}"

case "$MODE" in
  smoke) ENVS="SMOKE=1"; SID="qwen3_8b_grpo_smoke" ;;
  full)  ENVS="";        SID="qwen3_8b_grpo_full"  ;;
  *) echo "usage: $0 {smoke|full} [extra hydra args...]" >&2; exit 2 ;;
esac

echo ">> ray job submit '${SID}' -> ${RAY_ADDRESS}  (driver pinned to a '${WORKER_RESOURCE}' node)"
exec ray job submit \
  --address "${RAY_ADDRESS}" \
  --entrypoint-resources "{\"${WORKER_RESOURCE}\": 1}" \
  --submission-id "${SID}_$(date +%Y%m%d_%H%M%S)" \
  --no-wait \
  -- bash -lc "cd '${REPO_ROOT}' && ${ENVS} bash baselines/run_qwen3_8b_grpo.sh $*"
