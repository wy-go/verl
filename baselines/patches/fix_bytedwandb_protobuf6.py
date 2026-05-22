#!/usr/bin/env python3
"""Make byted-wandb importable under protobuf 6.x -- with no downgrade.

byted-wandb (ByteDance's wandb fork, the 0.13.x line) targets protobuf <5.
The verl baseline env runs protobuf 6.x: sglang 0.5.8 pulls grpcio-tools,
which hard-requires protobuf>=6.31.1. Two things then break `import wandb`:

  1. byteddatabus -- a byted-wandb dependency -- ships a legacy-style
     `py3_collector_pb2.py` that calls `_descriptor.FileDescriptor(...)` and
     `_descriptor.FieldDescriptor(...)` directly. protobuf >=3.20 forbids
     this with `TypeError: Descriptors cannot be created directly`. wandb
     imports databus, so this fails FIRST -- before any wandb code runs.

  2. wandb's own proto shims, `wandb/proto/*_pb2.py`, dispatch on the
     protobuf major version: each branches on "3" and "4" only. Under
     protobuf 5/6 no branch matches, the shim imports nothing, and the
     downstream `from ...wandb_internal_pb2 import <Message>` raises
     ImportError.

This script fixes both, in-place and idempotently:

  * databus: regenerates `py3_collector_pb2.py` with the modern
    descriptor-pool / builder API. The serialized FileDescriptor bytes are
    extracted from the original file and reused verbatim, so the proto
    schema and the wire format are byte-for-byte unchanged.
  * wandb:   widens each version dispatcher so protobuf 4/5/6 all load the
    bundled `wandb/proto/v4/` stubs (modern builder API, fine on 6.x).

Run it after `pip install byted-wandb` -- baselines/setup_env.sh does this.
Re-run it after any reinstall of wandb / byteddatabus: a reinstall restores
the unpatched files. The script is safe to re-run (it detects already-patched
files and skips them); originals are kept next to each file as *.protobuf-pre6.bak.

Verified: byted-wandb 0.13.95 + protobuf 6.33.6, offline
`wandb.init` / `wandb.log` / `wandb.finish` round-trip. Background and the
dependency math are in baselines/patches/README.md.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys

BAK_SUFFIX = ".protobuf-pre6.bak"


def _pkg_dir(name: str) -> str | None:
    """Directory of an installed top-level package, or None if absent."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def patch_wandb_dispatchers() -> int:
    """Route protobuf 4/5/6 to wandb's bundled v4 proto stubs."""
    wandb_dir = _pkg_dir("wandb")
    proto_dir = os.path.join(wandb_dir, "proto") if wandb_dir else None
    if not proto_dir or not os.path.isdir(proto_dir):
        print("  wandb/proto not found -- skipping (is byted-wandb installed?)")
        return 0

    old = 'protobuf_version == "4"'
    new = 'protobuf_version in ("4", "5", "6")'
    changed = 0
    for fn in sorted(os.listdir(proto_dir)):
        if not fn.endswith("_pb2.py"):
            continue
        path = os.path.join(proto_dir, fn)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if "protobuf_version" not in src:
            continue  # not a version dispatcher (e.g. an actual stub)
        if new in src:
            print(f"  {fn}: already patched")
            continue
        if old not in src:
            print(f"  {fn}: unrecognised dispatcher form -- left untouched")
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src.replace(old, new))
        print(f"  {fn}: patched (protobuf 4/5/6 -> v4 stubs)")
        changed += 1
    return changed


def _extract_serialized_pb(path: str) -> bytes | None:
    """Pull the serialized FileDescriptor bytes out of a legacy *_pb2.py."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "serialized_pb":
                    return ast.literal_eval(kw.value)
    return None


_TEMPLATE = '''\
# -*- coding: utf-8 -*-
# Regenerated for protobuf 6.x by baselines/patches/fix_bytedwandb_protobuf6.py.
# The original is saved alongside as this file + "{bak}". The serialized
# FileDescriptor below is byte-identical to byteddatabus's original -- only the
# protobuf codegen API is modernised, so the schema and wire format are unchanged.
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile({pb!r})

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, {mod!r}, globals())
'''


def patch_databus_stub() -> int:
    """Rewrite byteddatabus's legacy py3_collector_pb2.py to the modern API."""
    databus_dir = _pkg_dir("databus")
    if databus_dir is None:
        print("  databus not found -- skipping (no byteddatabus installed?)")
        return 0
    path = os.path.join(databus_dir, "py3_collector_pb2.py")
    if not os.path.isfile(path):
        print("  databus/py3_collector_pb2.py not found -- skipping")
        return 0

    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if "AddSerializedFile" in src:
        print("  py3_collector_pb2.py: already modern -- skipped")
        return 0

    pb = _extract_serialized_pb(path)
    if pb is None:
        print("  py3_collector_pb2.py: could not extract serialized_pb -- ABORT")
        return -1

    bak = path + BAK_SUFFIX
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as fh:
            fh.write(src)
    mod = os.path.splitext(os.path.basename(path))[0]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_TEMPLATE.format(pb=pb, mod=mod, bak=BAK_SUFFIX))

    # drop stale bytecode for the rewritten module
    cache = os.path.join(databus_dir, "__pycache__")
    if os.path.isdir(cache):
        for f in os.listdir(cache):
            if f.startswith("py3_collector_pb2"):
                os.remove(os.path.join(cache, f))
    print(f"  py3_collector_pb2.py: regenerated (original -> *{BAK_SUFFIX})")
    return 1


def main() -> int:
    print("Patching byted-wandb for protobuf 6.x compatibility ...")
    print("[1/2] byteddatabus proto stub")
    if patch_databus_stub() < 0:
        return 1
    print("[2/2] wandb proto dispatchers")
    patch_wandb_dispatchers()
    print("Done. Verify with: python -c 'import wandb; print(wandb.__version__)'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
