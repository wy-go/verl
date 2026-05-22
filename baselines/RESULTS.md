# Baseline Results Log

One row per run. Fill in after each run; link the TensorBoard/console log.
Reference targets: <https://verl.readthedocs.io/en/latest/algo/baseline.html>

## Runs

| Date | Experiment | Model | Dataset | GPUs | Rollout | Steps | Val reward | Sec/step | Status | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-21 | smoke test (attempt 8) | Qwen3-8B | gsm8k | 8×H100 | sglang | 7 | **0.817** | ~253 | ✅ pass | `baselines/logs/` |
| 2026-05-22 | phase-1 baseline | Qwen3-8B | gsm8k | 8×H100 | sglang | 105 | **0.950** | ~208 | ✅ pass | `baselines/logs/` |

## Run details

### qwen3_8b_grpo_gsm8k_sglang_20260521_211855 — smoke test

- Date / commit: 2026-05-21 / `wanyu-baselines` (offload fix baked in at `adde3367`)
- Command: `SMOKE=1 bash baselines/run_qwen3_8b_grpo.sh` — run as attempt 8 with
  `param_offload=True optimizer_offload=True` passed on the CLI; `adde3367` makes
  both the launcher default.
- Config deltas vs canonical script: FSDP `param_offload=True` +
  `optimizer_offload=True` (fixes the colocated OOM — see README §2);
  SMOKE → `total_epochs=1`, `save_freq=-1`, `test_freq=2`.
- Hardware: 8×H100-80GB, FSDP full-shard, sglang TP=2 (4 rollout replicas)
- Peak HBM / GPU util: actor ~54 GB allocated / ~65 GB reserved per GPU; ~70 GB
  used at 99% util during training; ~182 GB CPU RAM holding offloaded state
- Sec/step: 318 → 294 → 276 → 272 → 268 → 254 → 253 (first-step warmup
  amortizes; settles ~253)
- Reward curve (training reward mean): step1 0.318 → step7 0.848
- Validation reward (GSM8K acc@1): step0 0.320 · step2 0.535 · step4 0.760 ·
  step6 0.807 · final 0.817
- Reference target & source: low/mid-0.9s for an 8B-class model on GSM8K —
  verl baseline page <https://verl.readthedocs.io/en/latest/algo/baseline.html>
- Reproduced? partial — smoke test only (1 epoch / 7 steps); pipeline validated
  end-to-end, full Phase-1 run (15 epochs) pending
- Issues / notes: took 8 attempts to get a clean run. Launcher bugs fixed and
  committed along the way: HF-Xet workaround not propagated to Ray workers,
  env-var int/string quoting, `TENSORBOARD_DIR` not propagated, and FSDP offload
  for the colocated sglang-resume OOM. See README §2 "Known issues".

### qwen3_8b_grpo_gsm8k_sglang_20260522_002218 — Phase-1 baseline

- Date / commit: 2026-05-22 / `wanyu-baselines` (launcher at `53fec55f`)
- Command: `bash baselines/run_qwen3_8b_grpo.sh` (GSM8K, sglang, defaults)
- Config deltas vs canonical script: GSM8K-only data (canonical ships
  gsm8k+math); FSDP `param_offload=True` + `optimizer_offload=True`;
  `max_actor_ckpt_to_keep=1`; absolute `default_local_dir`. Otherwise the
  canonical `examples/grpo_trainer/run_qwen3_8b_fsdp.sh`: 15 epochs,
  `test_freq=5`, `save_freq=20`.
- Hardware: 8×H100-80GB, FSDP full-shard, sglang TP=2 (4 rollout replicas)
- Wall time: ~6h05m for 105 steps (15 epochs)
- Peak HBM / GPU util: actor ~55 GB allocated / ~66 GB reserved per GPU;
  ~203 GB CPU RAM holding offloaded param + optimizer state
- Sec/step: ~208 avg (range ~190–230; the ~226/230 outliers are
  checkpoint-save steps). Throughput ~2.6k tok/s mid-run.
- Reward curve (training reward mean): ~0.94 by step 25, ~0.97 by step 50,
  then holds 0.96–0.98 (final step-105 `critic/rewards/mean` 0.969). Stable —
  entropy 0.07, grad norm ~0.03, ppo_kl ~2e-5 across all 15 epochs.
- Validation reward (GSM8K acc@1, every 5 steps): step0 0.318 → step5 0.793 →
  step25 0.927 → step50 0.945 → step90 0.951 (peak 0.9515) → **step105 0.950**
- Reference target & source: low/mid-0.9s for an 8B-class model on GSM8K —
  verl baseline page <https://verl.readthedocs.io/en/latest/algo/baseline.html>
- Reproduced? **yes** — final GSM8K acc@1 0.950 (peak 0.9515) lands squarely in
  the expected mid-0.90s band; full 15-epoch schedule, stable throughout.
- Checkpoints: `baselines/checkpoints/qwen3_8b_grpo_gsm8k_sglang_20260522_002218/`
  — `global_step_105` is the final actor (`max_actor_ckpt_to_keep=1`).
- Issues / notes: clean run — no OOM, no crashes, no early stop. The earlier
  step-20 crash (relative `default_local_dir`) was already fixed in `53fec55f`
  before this run.

## Run detail template

Copy this block per run.

```
### <experiment_name>
- Date / commit:
- Command:
- Config deltas vs canonical script:
- Hardware: 8×H100-80GB, FSDP, sglang TP=2
- Peak HBM / GPU util:
- Sec/step (rollout / update split if known):
- Reward curve: start -> end (training reward mean)
- Validation reward: best @ step
- Reference target & source:
- Reproduced? (yes/no/partial):
- Issues / notes:
```
