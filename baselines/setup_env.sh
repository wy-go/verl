#!/usr/bin/env bash
# One-time environment setup for the verl baseline on this 8xH100 box.
#
# The stock image ships torch 2.8.0 + sglang 0.5.2rc2, which does not work here:
# the prebuilt flash-attn is ABI-broken against a newer torch, and sglang needs
# a newer build. This script brings the env to the known-good set:
#
#   torch 2.9.1  +  sglang 0.5.8  +  matching deps  +  flash-attn rebuilt 2.8.3
#
# Safe to re-run. Takes ~15 min, dominated by the flash-attn source build.
# See baselines/README.md "Known issues" for the reason behind each step.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. torch first — flash-attn (step 4) compiles against the installed torch.
pip install 'torch==2.9.1'

# 2. sglang + training deps, batched so pip resolves them together. torch is
#    pinned so the resolver cannot quietly move it. sglang 0.5.8 pulls
#    protobuf 6.x, which breaks the old byted-wandb (handled in step 3).
pip install \
  'sglang[srt,openai]==0.5.8' \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' \
  'transformers~=4.57.0' \
  'numpy==2.2.6' 'pandas==2.2.3' \
  tensorboard \
  'torch==2.9.1'

# 3. wandb: keep the byted-wandb fork so runs log to the ByteDance wandb
#    dashboard. byted-wandb 0.13.x targets protobuf <5, so installing it
#    transiently downgrades protobuf; capture the current (6.x) version first
#    and restore it afterward -- sglang 0.5.8 / grpcio-tools require 6.x.
#    fix_bytedwandb_protobuf6.py then rewrites the byteddatabus + wandb proto
#    stubs so `import wandb` works under protobuf 6 with no downgrade.
#    See baselines/patches/README.md. The pip "incompatible" warnings about
#    the protobuf pins are cosmetic -- the patch is what makes it work.
PROTOBUF_VERSION=$(python3 -c 'import google.protobuf as p; print(p.__version__)')
pip uninstall -y wandb byted-wandb || true
pip install 'byted-wandb==0.13.95'
# byteddatabus (a byted-wandb dep) is often pre-baked into the read-only system
# site-packages; the protobuf-6 patch must rewrite its proto stub, so install a
# copy into the writable user site (it shadows the system one on sys.path).
# --ignore-installed skips uninstalling the read-only system copy.
pip install --user --no-deps --ignore-installed byteddatabus
pip install "protobuf==${PROTOBUF_VERSION}"
python3 "$SCRIPT_DIR/patches/fix_bytedwandb_protobuf6.py"

# 4. flash-attn: the prebuilt 2.8.3 wheel is torch-2.8 ABI and fails against
#    torch 2.9.1 (undefined symbol _ZNK3c106SymInt6sym_neERKS0_). Rebuild 2.8.3
#    from source. --force-reinstall because a stale copy also lives in the
#    read-only system site-packages and would otherwise satisfy the install.
pip uninstall -y flash-attn flash_attn || true
MAX_JOBS="${MAX_JOBS:-23}" FLASH_ATTENTION_FORCE_BUILD=TRUE \
  pip install -v --force-reinstall --no-deps --no-cache-dir \
    --no-build-isolation --no-binary flash-attn 'flash-attn==2.8.3'

# 5. verify the whole stack imports together in one process.
python3 - <<'PY'
import torch, flash_attn, flash_attn_2_cuda, sglang, transformers, tensordict, wandb, verl
from torch.utils.tensorboard import SummaryWriter  # noqa: F401
print("torch       ", torch.__version__)
print("flash_attn  ", flash_attn.__version__)
print("sglang      ", sglang.__version__)
print("transformers", transformers.__version__)
print("tensordict  ", tensordict.__version__)
print("wandb       ", wandb.__version__)
print("cuda        ", torch.cuda.is_available(), torch.cuda.device_count(), "GPU(s)")
print("env OK")
PY
