"""Train the FertiLuna model and export to ONNX.

Usage:
    python -m scripts.train_and_export \
        --out artifacts \
        --n-samples 50000 \
        --version v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fertiluna.export_onnx import export
from fertiluna.train import TrainConfig, save_metrics, train


def main() -> int:
    p = argparse.ArgumentParser(description="Train + export FertiLuna model.")
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", type=str, default="v1")
    args = p.parse_args()

    cfg = TrainConfig(n_samples=args.n_samples, seed=args.seed)
    result = train(cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    save_metrics(result, args.out / f"metrics-{args.version}.json")
    manifest = export(result, args.out, version=args.version)

    print("\n=== summary ===")
    print(f"  accuracy: {manifest['metrics']['accuracy']:.4f}")
    print(f"  log_loss: {manifest['metrics']['log_loss']:.4f}")
    print(f"  artifacts in: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
