#!/usr/bin/env python3
"""Scan the model registry and auto-build missing TensorRT engines from ONNX.

Usage (from repo root, cwd must resolve `openpilot` package):
  python openpilot/sunnypilot/modeld_v2/gpu_backend/build_plans.py [--dry-run]

The ONNX files carry their own weights (verified: CD210 vision 267
initializers/46MB, FiletOFish 205/34MB), so trtexec from ONNX is sufficient;
tinygrad.pkl is only needed for the tinygrad fallback path.

Models without an ONNX (e.g. BigCombo: only plan+metadata on this rig) are
reported as "manual" — copy the plan from the other fork or fetch the
official supercombo.onnx and rebuild.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def trtexec(onnx: Path, out: Path, fp16: bool = True, workspace_mb: int = 4096) -> int:
  prec = "--fp16" if fp16 else "--noTF32"
  cmd = [
    "trtexec", f"--onnx={onnx}", f"--saveEngine={out}", prec,
    f"--memPoolSize=workspace:{workspace_mb}", "--noDataTransfers",
  ]
  print(f"[build] {' '.join(cmd)}")
  return subprocess.run(cmd).returncode


def main() -> int:
  sys.path.insert(0, str(REPO_ROOT))
  from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import REGISTRY

  ap = argparse.ArgumentParser()
  ap.add_argument("--dry-run", action="store_true", help="only print the plan")
  args = ap.parse_args()

  rc = 0
  for name, prof in sorted(REGISTRY.items()):
    root = Path(prof.engine_dir)
    if prof.is_ready():
      eng = prof.merged_engine or prof.vision_engine
      print(f"[ok   ] {name:<14} mode={prof.mode:<7} {eng}")
      continue

    # ── find ONNX sources ──
    onnx_pairs: list[tuple[Path, Path]] = []
    if prof.mode == "merged":
      merged_plan = root / prof.merged_engine if prof.merged_engine else None
      for onnx in sorted(root.glob("*supercombo*.onnx")):
        if merged_plan is not None and not merged_plan.exists():
          onnx_pairs.append((onnx, merged_plan))
    else:
      for src, cand in (("driving_vision.onnx", prof.vision_engine),
                        ("driving_policy.onnx", prof.policy_engine)):
        onnx = root / src
        plan = root / cand if cand else None
        if onnx.exists() and plan is not None and not plan.exists():
          onnx_pairs.append((onnx, plan))
    for onnx in sorted(root.glob("*.onnx")):
      if onnx.name in ("dmonitoring_model.onnx",):
        continue

    if not onnx_pairs:
      print(f"[manual] {name:<14} mode={prof.mode:<7} no engine & no ONNX "
            f"(copy .plan here or supply ONNX)")
      rc = 1
      continue

    plan_names = " + ".join(p.name for _, p in onnx_pairs)
    print(f"[build ] {name:<14} mode={prof.mode:<7} {plan_names}")
    if args.dry_run:
      continue
    for onnx, plan in onnx_pairs:
      ret = trtexec(onnx, plan)
      if ret != 0:
        print(f"[fail ] {onnx.name} -> {plan.name} rc={ret}")
        rc = 1
  print(f"\nregistry: {len(REGISTRY)} profiles")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())