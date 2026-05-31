"""Train the FertiLuna chart-vision model and export to ONNX.

================================================================================
JUST RUN IT (auto-tunes batch size, workers, learning rate for this box)
================================================================================

    python -m scripts.build_vision_dataset --out data --n 200000 --seed 1  --style blend
    python -m scripts.build_vision_dataset --out data --n 20000  --seed 99 --style blend

    python -m scripts.train_and_export_vision \\
        --train-npz data/charts-blend-200000-seed1 \\
        --val-npz   data/charts-blend-20000-seed99 \\
        --width 3.0 --epochs 40 --version v1

That's it. batch_size, num_workers, and lr are auto-picked from detected
VRAM + CPU count. Env vars (PYTORCH_CUDA_ALLOC_CONF, OMP_NUM_THREADS) are
set automatically. AMP uses bf16 if your GPU supports it. The script prints
the auto-tune plan before training so you can see what was picked.

Override any auto-picked value by passing it explicitly:
    --batch-size 512 --num-workers 24 --lr 2e-2

Outputs in artifacts/:
    chart-vision-v1.onnx           (the model; copied to the web app's public/)
    chart-vision-manifest-v1.json  (IO spec, normalization, metrics, checksum)
    chart-vision-v1.ckpt           (best torch weights; resumable)

--------------------------------------------------------------------------------
Quick smoke test (any machine, no GPU needed):

    python -m scripts.train_and_export_vision \\
        --train-size 256 --val-size 64 --epochs 1 --num-workers 0 --version smoke
--------------------------------------------------------------------------------

Fine-tune from an existing checkpoint (e.g. pre-trained on generic, fine-tune
on a Premom-heavy blend — see model/README.md §1.5):

    python -m scripts.train_and_export_vision \\
        --train-npz data/charts-blend-40000-seed1 \\
        --val-npz   data/charts-blend-5000-seed99 \\
        --width 3.0 --epochs 15 --lr 2e-4 \\
        --init-ckpt artifacts/chart-vision-base.ckpt \\
        --out artifacts --version v1
"""

from __future__ import annotations

# These env vars must be set BEFORE `import torch` (and therefore before any
# fertiluna_vision import). PYTORCH_CUDA_ALLOC_CONF is read at first CUDA
# alloc; OMP_NUM_THREADS / MKL_NUM_THREADS are read by the OpenMP / MKL
# runtime when torch first uses them. setdefault leaves explicit user overrides
# alone — pass e.g. `OMP_NUM_THREADS=4 python -m ...` to override.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

from fertiluna_vision.export_onnx import export
from fertiluna_vision.train import VisionTrainConfig, auto_tune, train


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train + export FertiLuna chart-vision model. "
                    "batch_size, num_workers, lr are auto-tuned to your box "
                    "unless explicitly passed.",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--train-size", type=int, default=20_000)
    p.add_argument("--val-size", type=int, default=2_000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--train-npz", type=str, default=None)
    p.add_argument("--val-npz", type=str, default=None)
    p.add_argument("--version", type=str, default="v1")
    p.add_argument("--init-ckpt", type=str, default=None,
                   help="Fine-tune: load weights from this .ckpt instead of random init (width must match).")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze the conv backbone; train only width_refine + heads (fast retarget).")

    # Auto-tuned by default — explicit values override.
    p.add_argument("--batch-size", type=int, default=None,
                   help="Auto-picked from VRAM × model width if not set.")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Auto-picked from CPU count if not set "
                        "(capped at 16 — past that, a memmap dataset just wastes RSS).")
    p.add_argument("--lr", type=float, default=None,
                   help="Auto-scaled linearly from chosen batch_size if not set. "
                        "For fine-tuning with --init-ckpt, pass an explicit small value (e.g. 2e-4).")

    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="DataLoader batches kept in flight per worker (4–8 typical).")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                   help="Mixed-precision dtype on CUDA. bf16 (default) is preferred "
                        "on Ampere+/Hopper/Blackwell: same speed as fp16, no GradScaler, "
                        "better numerical stability.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the model in torch.compile(). Off by default — for "
                        "a 4.7M-param CNN the warmup rarely pays off, since "
                        "cudnn.benchmark + bf16 + TF32 already capture most of "
                        "the speedup.")
    p.add_argument("--compile-mode",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   default="reduce-overhead",
                   help="torch.compile mode. 'reduce-overhead' (default) enables "
                        "CUDA Graphs (the main win for small models) and warms up "
                        "in ~30-60s. 'max-autotune' benchmarks every Triton kernel "
                        "variant — 5-15 MINUTES of compile time for usually near-"
                        "zero gain on small CNNs. Avoid unless you've measured.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = str(args.out / f"chart-vision-{args.version}.ckpt")

    # ---- auto-tune --------------------------------------------------------
    # Fill in only the knobs the user didn't pass. Print the plan so the user
    # can see what we picked and how to override.
    plan = auto_tune(args.width)
    batch_size = args.batch_size if args.batch_size is not None else plan["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else plan["num_workers"]
    # If lr wasn't passed, scale the auto-tune baseline by the *actual* batch
    # (which may differ from auto if the user explicitly set --batch-size).
    if args.lr is not None:
        lr = args.lr
    else:
        lr = 3e-3 * (batch_size / 64.0)
        lr = max(1e-4, min(5e-2, lr))

    print(
        "[auto-tune] "
        f"VRAM={plan['_vram_gb']:.0f} GB  CPUs={plan['_cpu']}  width={args.width}  ->  "
        f"batch_size={batch_size}{'' if args.batch_size is None else ' (user)'}  "
        f"num_workers={num_workers}{'' if args.num_workers is None else ' (user)'}  "
        f"lr={lr:.2e}{'' if args.lr is None else ' (user)'}",
        flush=True,
    )
    print(
        "[auto-tune] env: "
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}  "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}",
        flush=True,
    )
    print(
        "[auto-tune] override any value by passing it explicitly: "
        "--batch-size N / --num-workers N / --lr X",
        flush=True,
    )
    # ----------------------------------------------------------------------

    cfg = VisionTrainConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        epochs=args.epochs,
        batch_size=batch_size,
        width=args.width,
        lr=lr,
        num_workers=num_workers,
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        checkpoint_path=ckpt,
        init_ckpt=args.init_ckpt,
        freeze_backbone=args.freeze_backbone,
        prefetch_factor=args.prefetch_factor,
        amp_dtype=args.amp_dtype,
        compile=args.compile,
        compile_mode=args.compile_mode,
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
