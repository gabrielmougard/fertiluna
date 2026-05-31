"""Train the FertiLuna chart-vision model and export to ONNX.

================================================================================
RECOMMENDED "BEST MODEL" RECIPE (compute time NOT a concern — go for quality)
================================================================================

The model maps a cycle-chart screenshot -> per-day BBT + LH (normalized value +
presence). It trains on synthetic charts we render ourselves, so labels are free
and exact. To maximize quality, train on a LARGE, freshly-rendered dataset with
a higher-capacity (still <100M) backbone for many epochs.

1) (one-time) Install a CUDA build of PyTorch on the GPU laptop, e.g. CUDA 12.1:

     uv pip install --upgrade "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121
     # (also: uv pip install onnxscript onnx onnxruntime pillow matplotlib numpy)

2) Render big train + val datasets ONCE (uint8 .npz; reused every epoch):

     python -m scripts.build_vision_dataset --out data --n 120000 --seed 1  --workers 16
     python -m scripts.build_vision_dataset --out data --n 10000  --seed 99 --workers 16

   (Tip: bump --n higher if you have disk/RAM; more synthetic variety = better
    generalization to real app screenshots. ~120k train is a strong target.)

3) Train the best model (width 3.0 ≈ 4.7M params, still tiny & <100M) and export:

     python -m scripts.train_and_export_vision \
         --train-npz data/charts-120000-seed1.npz \
         --val-npz   data/charts-10000-seed99.npz \
         --width 3.0 --epochs 40 --batch-size 128 --num-workers 16 \
         --out artifacts --version v1

   On CUDA this uses AMP mixed precision + cudnn autotuning automatically.
   Best weights are checkpointed to artifacts/chart-vision-v1.ckpt after every
   improvement, so a crash never loses the model. If export didn't run, recover:

     python -m scripts.export_vision_from_ckpt \
         --ckpt artifacts/chart-vision-v1.ckpt --out artifacts --version v1

Outputs in artifacts/:
   chart-vision-v1.onnx            (the model; copied to the web app's public/)
   chart-vision-manifest-v1.json   (IO spec, normalization, metrics, checksum)

--------------------------------------------------------------------------------
Quick smoke test (any machine, no GPU needed):

    python -m scripts.train_and_export_vision \
        --train-size 256 --val-size 64 --epochs 1 --num-workers 0 --version v1test
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fertiluna_vision.export_onnx import export
from fertiluna_vision.train import VisionTrainConfig, train


def main() -> int:
    p = argparse.ArgumentParser(description="Train + export FertiLuna chart-vision model.")
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--train-size", type=int, default=20_000)
    p.add_argument("--val-size", type=int, default=2_000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-npz", type=str, default=None)
    p.add_argument("--val-npz", type=str, default=None)
    p.add_argument("--log-every", type=int, default=200,
                   help="Print a running-loss line every N training steps (0=off).")
    p.add_argument("--version", type=str, default="v1")
    p.add_argument("--lr", type=float, default=3e-3,
                   help="Max learning rate. Use a much smaller value (e.g. 2e-4) when fine-tuning with --init-ckpt.")
    p.add_argument("--init-ckpt", type=str, default=None,
                   help="Fine-tune: load weights from this .ckpt instead of random init (width must match).")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze the conv backbone; train only width_refine + heads (fast retarget).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = str(args.out / f"chart-vision-{args.version}.ckpt")

    cfg = VisionTrainConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        width=args.width,
        lr=args.lr,
        num_workers=args.num_workers,
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        log_every=args.log_every,
        checkpoint_path=ckpt,
        init_ckpt=args.init_ckpt,
        freeze_backbone=args.freeze_backbone,
    )
    result = train(cfg)

    with open(args.out / f"vision-metrics-{args.version}.json", "w") as f:
        json.dump(result.metrics, f, indent=2)

    manifest = export(result, args.out, version=args.version)
    mb = manifest["files"]["model"]["bytes"] / 1e6
    print("\n=== vision summary ===")
    print(f"  params:   {result.metrics['n_params']/1e6:.2f}M")
    print(f"  onnx:     {mb:.2f} MB")
    print(f"  val MAE:  {result.metrics['best_val_mae_present']:.4f} (normalized)")
    print(f"  presence F1: {result.metrics['final']['val_presence_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
