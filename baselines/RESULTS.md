# Baseline Results Log

One row per run. Fill in after each run; link the TensorBoard/console log.
Reference targets: <https://verl.readthedocs.io/en/latest/algo/baseline.html>

## Runs

| Date | Experiment | Model | Dataset | GPUs | Rollout | Steps | Val reward | Sec/step | Status | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _TBD_ | _smoke test_ | Qwen3-8B | gsm8k | 8×H100 | sglang | ~7 | — | — | pending | — |
| _TBD_ | _phase-1 baseline_ | Qwen3-8B | gsm8k | 8×H100 | sglang | ~105 | — | — | pending | — |

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
