# verl Baseline & Reward-Curve Reproduction

Working notes for reproducing verl reward curves and collecting performance
baselines on this machine. Phase 1 is a single-node smoke/baseline run; later
phases scale up the model and GPU count.

> This directory holds **our experiment notes** — it is intentionally separate
> from the verl source tree and from the repo's `CLAUDE.md` (which is the
> verl-project contribution policy and must not be edited for our purposes).

- Plan & how-to: this file
- Per-run results log: [`RESULTS.md`](RESULTS.md)
- One-time environment setup: [`setup_env.sh`](setup_env.sh)
- Data prep: [`prepare_data.sh`](prepare_data.sh)
- Phase-1 launcher: [`run_qwen3_8b_grpo.sh`](run_qwen3_8b_grpo.sh)

---

## 1. Machine snapshot

| Resource | Value |
| --- | --- |
| GPUs | 8 × NVIDIA H100 80GB HBM3 (single node) |
| CPU / RAM | 23 cores / ~1.8 TB |
| Disk (`/`) | ~270 GB free |
| Driver / CUDA | 535.129 · system toolkit 12.4 · torch bundles CUDA 12.8 |
| torch | 2.9.1+cu128 |
| rollout engines | sglang 0.5.8 ✅ &nbsp;·&nbsp; vLLM 0.8.5 ❌ (see below) |
| train backends | FSDP ✅ &nbsp;·&nbsp; Megatron ✅ &nbsp;·&nbsp; flash-attn 2.8.3 (rebuilt from source) |
| verl | editable install at `/opt/tiger/rl/verl` |

> The stock image ships an older, broken stack (torch 2.8.0 + sglang 0.5.2rc2).
> [`setup_env.sh`](setup_env.sh) brings it to the versions above — run it once on
> a fresh box. The table and the rest of this file reflect the post-setup state.

## 2. Known issues / gotchas

The stock image does **not** work out of the box. `setup_env.sh` and the
launcher scripts handle the items below — this section is the *why*.

- **HF Xet downloads hang.** The Hugging Face Xet protocol bypasses
  `HF_ENDPOINT` (the Bytedance mirror) and stalls on unreachable CAS servers.
  `HF_HUB_DISABLE_XET=1` forces classic LFS through the mirror — `prepare_data.sh`
  and `run_qwen3_8b_grpo.sh` export it; pass it for manual `huggingface-cli` pulls.
- **vLLM is broken.** `import vllm` fails with
  `ImportError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb` —
  vLLM 0.8.5's prebuilt extension does not match torch 2.9.1. **Use
  `INFER_BACKEND=sglang`** for all rollouts. To restore vLLM, install a build
  matching torch 2.9.
- **flash-attn ABI mismatch.** The prebuilt flash-attn 2.8.3 wheel is built for
  torch 2.8; against torch 2.9.1 it fails with `undefined symbol:
  _ZNK3c106SymInt6sym_neERKS0_`. `setup_env.sh` rebuilds 2.8.3 from source
  (~12 min) into user site-packages, which shadows the stale system copy.
- **sglang scheduler libcudart mismatch.** torch 2.9.1 bundles CUDA 12.8 libs;
  the system toolkit is 12.4. The sglang scheduler subprocess otherwise loads
  the wrong `libcudart`. The launcher prepends torch's cu12 libs to
  `LD_LIBRARY_PATH` and propagates them to Ray workers via `runtime_env`.
- **wandb / protobuf clash.** sglang 0.5.8 pulls protobuf 6.x, which rejects the
  byted-wandb fork's old generated `*_pb2.py`. `setup_env.sh` swaps it for stock
  `wandb==0.26.1`. We log to **tensorboard** by default anyway (no auth needed);
  `console` is always on.
- **Low CPU count (23 cores for 8 GPUs).** Fine for GSM8K (cheap regex reward),
  but watch Ray dataloader / reward-worker warnings. Avoid CPU-heavy custom
  reward functions without bumping `data.dataloader_num_workers` carefully.
- **Disk pressure.** One 8B FSDP checkpoint *with optimizer state* is ~80 GB,
  and `/` has only ~270 GB free. Keep `trainer.max_actor_ckpt_to_keep` small and
  use `save_freq=-1` for smoke tests. The launcher sets
  `max_actor_ckpt_to_keep=1` (one ~80 GB checkpoint retained at a time).

## 3. Phase 1 baseline — Qwen3-8B GRPO on GSM8K

Canonical example: `examples/grpo_trainer/run_qwen3_8b_fsdp.sh`
(GRPO is critic-less — see `examples/grpo_trainer/README.md`).

### System design (why these choices)

8B dense on 8×H100-80GB is **small enough that FSDP is the right tool** —
Megatron's TP/PP/EP machinery only pays off at larger scale or for MoE. Match
the parallelism to the scale; don't over-engineer.

| Component | Setting | Rationale |
| --- | --- | --- |
| Train backend | FSDP, full-shard over 8 GPUs | 8B params shard cleanly; ~2 GB params/GPU, optimizer fits without offload |
| Param/optim offload | **off** | H100-80GB has headroom at this size; offload only costs throughput |
| Rollout engine | sglang, `tensor_model_parallel_size=2` | 2-way TP → 4 data-parallel rollout replicas across 8 GPUs |
| Placement | colocated (actor + rollout share GPUs) | `gpu_memory_utilization=0.6` reserves 60% HBM for the KV cache, rest for FSDP state + activations |
| Sequence parallel | `sp_size=1` | not needed at 1k prompt + 2k response lengths |
| Batching | dynamic bsz + sequence balancing (defaults) | even token load across ranks |

### Key hyperparameters (from the canonical script)

| Knob | Value |
| --- | --- |
| `algorithm.adv_estimator` | `grpo` |
| `data.train_batch_size` | 1024 prompts/step |
| `actor_rollout_ref.rollout.n` | 5 samples/prompt → 5120 trajectories/step |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 256 |
| `data.max_prompt_length` / `max_response_length` | 1024 / 2048 |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 |
| KL | `use_kl_loss=True`, `kl_loss_coef=0.001`, `low_var_kl`; `use_kl_in_reward=False` |
| `entropy_coeff` | 0 |
| `total_epochs` / `test_freq` / `save_freq` | 15 / 5 / 20 |

GSM8K train set is 7,473 prompts → ~7 optimizer steps/epoch → ~105 steps for
15 epochs. `gsm8k_math` (~15k prompts) is ~14 steps/epoch. Confirm wall-clock
step time from the first few steps and record it in `RESULTS.md`.

### Expected outcome

Validation runs every `test_freq=5` steps. For an 8B-class model the GSM8K
test score should climb into the low/mid-0.9s. **Verify the exact target
against the verl baseline page** before declaring a reproduction:
<https://verl.readthedocs.io/en/latest/algo/baseline.html>
(reference GRPO training log linked from `examples/grpo_trainer/README.md`).

## 4. How to run

```bash
# 0. one-time on a fresh box: bring the env to the known-good versions (~15 min)
bash baselines/setup_env.sh

# 1. (optional) pre-download the model so the first run isn't gated on a 16 GB pull
HF_HUB_DISABLE_XET=1 huggingface-cli download Qwen/Qwen3-8B

# 2. preprocess data → ~/data/gsm8k/{train,test}.parquet  (GSM8K only by default)
bash baselines/prepare_data.sh

# 3a. smoke test first — 1 epoch, no checkpoints, validates the whole pipeline
SMOKE=1 bash baselines/run_qwen3_8b_grpo.sh

# 3b. full Phase-1 baseline (GSM8K only, matches verl reference log)
bash baselines/run_qwen3_8b_grpo.sh

# variants
DATASET=gsm8k_math bash baselines/prepare_data.sh          # prep gsm8k + MATH, then:
DATASET=gsm8k_math bash baselines/run_qwen3_8b_grpo.sh     # train on gsm8k + math
INFER_BACKEND=vllm  bash baselines/run_qwen3_8b_grpo.sh    # only after vLLM is fixed
```

Always run the **smoke test first** — it catches data-path, OOM, and backend
issues in ~7 steps instead of failing hours into a full run.

## 5. Monitoring

- Console log: `baselines/logs/<experiment>.log` (tee'd by the launcher).
- TensorBoard: `tensorboard --logdir baselines/logs/tb` then watch
  - validation reward (the reproduction target)
  - training reward mean (the reward curve)
  - response length, KL, entropy, actor grad norm (stability)
- `nvidia-smi` / `nvitop` for GPU utilization and HBM headroom.

## 6. Scaling roadmap

When the model grows, change the parallelism strategy deliberately — pick the
backend and parallel dims to match scale, not habit.

| Model | Params (active) | Train backend | Parallelism | GPUs | Notes |
| --- | --- | --- | --- | --- | --- |
| **Qwen3-8B** dense | 8B | FSDP | full-shard; rollout TP=2 | 8×H100 | **Phase 1 (current)** |
| Qwen3-30B-A3B MoE | 30B (3B) | Megatron (or FSDP) | expert parallelism for MoE layers; rollout TP=2–4 | 8–16×H100 | EP is the win for MoE |
| Qwen2.5-32B dense | 32B | FSDP + offload, or Megatron TP=4 | TP=4; rollout TP=4–8 | 8–16×H100 | enable param/optim offload on FSDP |
| ~70B dense | 70B | Megatron | TP=8 + PP=2 | 16–32×H100, multi-node | |
| Qwen3-235B-A22B | 235B (22B) | Megatron | TP + PP + EP | 64+×H100, multi-node | scale demo script exists |
| DeepSeek-V3 | 671B | Megatron | TP + PP + EP | 256+ GPUs | scale demo script exists |

Each row has a matching `examples/grpo_trainer/run_*.sh` script — start from it
rather than hand-rolling configs. For multi-node, verl launches via Ray; set
`trainer.nnodes` and start a Ray cluster first.

## 7. References

- verl docs: <https://verl.readthedocs.io/en/latest/index.html>
- Baseline metrics: <https://verl.readthedocs.io/en/latest/algo/baseline.html>
- GRPO guide: `examples/grpo_trainer/README.md`
- Persistent context for the AI assistant lives in the Claude memory store
  (`MEMORY.md` index) — not in the repo's `CLAUDE.md`.
