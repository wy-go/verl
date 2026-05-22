# byted-wandb under protobuf 6.x

This branch (`wanyu-baselines-wandb`) logs training runs to the **byted-wandb**
fork so they land on the ByteDance wandb dashboard. byted-wandb is not
protobuf-6 compatible out of the box; [`fix_bytedwandb_protobuf6.py`](fix_bytedwandb_protobuf6.py)
makes it work with **no downgrade** of protobuf, sglang, torch, or anything
else. This file is the *why* and *how*.

## The conflict

| Package | protobuf requirement |
| --- | --- |
| `byted-wandb` (every version, 0.13.0–0.13.95) | `protobuf <5` (its dep `byteddatabus` tightens this to `<4`) |
| `grpcio-tools` 1.75.1 (pulled by the sglang 0.5.8 stack) | `protobuf >=6.31.1, <7` |
| installed in this env | `protobuf 6.33.6` |

The ranges do not overlap, so **no byted-wandb version is pip-compatible** with
this environment, and there is no newer byted-wandb (the fork is frozen at the
0.13.x line). Downgrading protobuf to satisfy byted-wandb would break
`grpcio-tools` and cascade into the sglang / Ray stack — not acceptable.

But the incompatibility is shallow: it is *generated-code* breakage, not a
wire-format or API problem. It can be patched in place.

## The two failure modes

`import wandb` (byted-wandb) fails in two distinct spots under protobuf 6.x:

1. **`byteddatabus`** — a byted-wandb dependency. Its `databus/py3_collector_pb2.py`
   is legacy-style generated code that calls `_descriptor.FileDescriptor(...)`
   and `_descriptor.FieldDescriptor(...)` directly. protobuf ≥3.20 forbids
   this:

   ```
   TypeError: Descriptors cannot be created directly.
   If this call came from a _pb2.py file, your generated code is out of date
   and must be regenerated with protoc >= 3.19.0.
   ```

   wandb imports `databus`, so this is the **first** failure — it happens
   before any wandb code runs.

2. **wandb's own proto shims** — `wandb/proto/wandb_{base,internal,server,telemetry}_pb2.py`.
   Each dispatches on the protobuf major version:

   ```python
   protobuf_version = google.protobuf.__version__[0]   # first char only
   if protobuf_version == "3":
       from wandb.proto.v3.wandb_internal_pb2 import *
   elif protobuf_version == "4":
       from wandb.proto.v4.wandb_internal_pb2 import *
   ```

   For protobuf `6.33.6`, `__version__[0]` is `"6"` — neither branch runs, the
   shim imports nothing, and the downstream `from ...wandb_internal_pb2 import
   <Message>` raises `ImportError`. The wheel **already ships `v4/` stubs**
   (modern `_builder` API) that load fine under protobuf 5/6 — the dispatcher
   just never selects them.

## The fix

[`fix_bytedwandb_protobuf6.py`](fix_bytedwandb_protobuf6.py) patches both, in
the active environment's `site-packages`, idempotently:

- **databus** — regenerates `py3_collector_pb2.py` with the modern
  descriptor-pool / `builder` API. The serialized `FileDescriptor` bytes are
  extracted from the original file and reused **verbatim**, so the proto
  schema and the wire format are byte-for-byte identical; only the codegen
  API is modernised.
- **wandb** — widens each version dispatcher to `protobuf_version in ("4",
  "5", "6")`, so protobuf 4/5/6 all load the bundled `v4/` stubs.

The original of every changed file is kept next to it as
`*.protobuf-pre6.bak`. The script detects already-patched files and skips
them, so it is safe to re-run.

## How it is applied

`baselines/setup_env.sh` runs it automatically — it installs byted-wandb,
installs a writable user-site copy of `byteddatabus` (which is often pre-baked
**read-only** into the system site-packages, where the patch could not rewrite
its stub), restores protobuf to 6.x, and invokes this script. To apply or
re-apply manually:

```bash
python baselines/patches/fix_bytedwandb_protobuf6.py
python -c "import wandb; print(wandb.__version__)"   # expect: 0.13.95
```

> **Re-run after any reinstall.** The patch lives in `site-packages`; a
> `pip install`/upgrade of `wandb` or `byteddatabus` restores the unpatched
> files. Re-run the script (or re-run `setup_env.sh`) afterward.

## Verification status

- ✅ **Offline** — verified end to end on byted-wandb 0.13.95 + protobuf
  6.33.6: `import wandb`, and a full `wandb.init` → `wandb.log` →
  `wandb.finish` round-trip writing and re-parsing the `.wandb` datastore.
  `baselines/run_qwen3_8b_grpo.sh` defaults to `WANDB_MODE=offline`; push to
  the dashboard afterward with `wandb sync baselines/logs/wandb/<run>`.
- ⏳ **Online** — not yet verified. Live streaming additionally exercises the
  gRPC sync path and byted auth deps (which *import* cleanly under protobuf 6).
  Set `WANDB_MODE=online` and provide `WANDB_API_KEY`; confirm on the first run.

## Reverting

To go back to stock wandb (the `wanyu-baselines` branch's choice):

```bash
pip uninstall -y byted-wandb && pip install 'wandb==0.26.1'
```
