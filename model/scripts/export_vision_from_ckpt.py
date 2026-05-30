"""Export a chart-vision model to ONNX from a saved .ckpt.

Safety net for the case where training completed and checkpointed the best
weights, but the process crashed (e.g. macOS DataLoader teardown SIGSEGV)
before the ONNX export ran.

Usage:
    python -m scripts.export_vision_from_ckpt \
        --ckpt artifacts/chart-vision-v1.ckpt --out artifacts --version v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fertiluna_vision.export_onnx import export
from fertiluna_vision.model import build_model
from fertiluna_vision.train import VisionTrainResult


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--version", type=str, default="v1")
    args = p.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    width = blob.get("width", 1.0)
    model = build_model(width=width)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    result = VisionTrainResult(
        model=model,
        metrics={
            "n_params": sum(p.numel() for p in model.parameters()),
            "best_val_mae_present": blob.get("best_val_mae"),
            "final": blob.get("metrics", {}),
            "history": [],
        },
        config=blob.get("config", {}),
    )
    manifest = export(result, args.out, version=args.version)
    print(f"\nExported {manifest['files']['model']['path']} "
          f"({manifest['files']['model']['bytes']/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
