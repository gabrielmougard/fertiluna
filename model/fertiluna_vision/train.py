"""Training loop for the chart-vision model.

Losses:
    presence: BCEWithLogits over all (series, day) cells.
    value:    smooth-L1 (Huber) on the soft-argmax value (already in [0,1]),
              masked to the days where a point is actually present.

Metrics:
    val_mae_present  — mean abs error of normalized value on present days
    val_presence_f1  — F1 of the presence prediction
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import PRESENCE_THRESHOLD
from .dataset import ChartDataset
from .model import build_model, count_params


# ---------------------------------------------------------------------------
# DataLoader IPC: use the file-system sharing strategy.
#
# By default PyTorch ships worker→main batches through shared memory backed by
# /dev/shm. Docker containers (including vast.ai's) usually cap /dev/shm at
# 64 MB, while a single 384-image batch is ~400 MB. The first worker then
# hangs forever in shm_open(), main hangs forever in queue.get(), and you see
# the model on GPU at 0% util with no batches ever produced.
#
# 'file_system' uses on-disk file descriptors instead of shm, which has no
# such cap. Marginally slower than shm when shm has headroom, but the only
# strategy that works reliably in containers. Must be set BEFORE any
# DataLoader workers are spawned — module import time is safe.
# ---------------------------------------------------------------------------
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    # Already-set or unsupported (e.g. on Windows) — best-effort, ignore.
    pass


def auto_tune(width: float) -> dict:
    """Pick sensible defaults for this box. Returns {batch_size, num_workers, lr}.

    Heuristics (empirically calibrated against the existing README baseline of
    `width=3.0 fits batch=32 in 8 GB VRAM`):
      * Per-sample peak activation memory ≈ 250 MB × (width / 3.0).
      * Use 75% of VRAM, round batch down to multiple of 32, clamp to [32, 2048].
      * num_workers = clamp(cpu_count // 12, 4, 16). Past 16 a memmap dataset
        wastes RSS on workers that mostly block waiting for the GPU.
      * lr scales linearly with batch from the README baseline (3e-3 at batch=64),
        clamped to [1e-4, 5e-2].
    """
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        per_sample_mb = 250.0 * (max(0.5, width) / 3.0)
        usable_mb = vram_gb * 1024.0 * 0.75
        b = int(usable_mb / per_sample_mb)
        batch = max(32, min(2048, (b // 32) * 32))
    else:
        vram_gb = 0.0
        batch = 32

    cpu = os.cpu_count() or 4
    workers = max(4, min(16, cpu // 12)) if cpu >= 8 else max(2, cpu - 1)

    # Linear LR scaling from baseline (lr=3e-3 at batch=64), clamped for sanity.
    lr = 3e-3 * (batch / 64.0)
    lr = max(1e-4, min(5e-2, lr))

    return {
        "batch_size": batch,
        "num_workers": workers,
        "lr": lr,
        "_vram_gb": vram_gb,
        "_cpu": cpu,
    }


def _worker_init(worker_id: int) -> None:
    """Per-DataLoader-worker init.

    Without this, each forked worker inherits the parent's OMP/MKL thread
    count (= host CPU count on a fat box). With `num_workers=N` and
    OMP_NUM_THREADS=N you end up with N×N threads fighting the scheduler.
    Pinning each worker to a single intra-op thread is the single biggest
    fix for "training stalls on a 192-core box" pathology.
    """
    torch.set_num_threads(1)
    try:
        # `torch.set_num_interop_threads` can only be set before any parallel
        # work — fine here because the worker process is fresh.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed = (torch.initial_seed() + worker_id) % (2**31)
    np.random.seed(seed)


@dataclass
class VisionTrainConfig:
    train_size: int = 20_000
    val_size: int = 2_000
    epochs: int = 8
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    width: float = 1.0
    num_workers: int = 4
    seed: int = 42
    value_loss_weight: float = 5.0
    presence_loss_weight: float = 1.0
    scale_loss_weight: float = 0.5
    # Optional pre-rendered datasets (much faster than on-the-fly rendering).
    train_npz: Optional[str] = None
    val_npz: Optional[str] = None
    # If set, the best weights are checkpointed here after every improvement so
    # a crash during DataLoader teardown can't lose the trained model.
    checkpoint_path: Optional[str] = None
    # Fine-tuning: start from these weights instead of random init. The model
    # width must match the checkpoint's width.
    init_ckpt: Optional[str] = None
    # If True, freeze the conv backbone (stem + blocks) so only width_refine and
    # the value/present heads adapt — fastest, safest way to retarget a trained
    # model to a new chart style without forgetting learned features.
    freeze_backbone: bool = False
    # Batches the loader keeps in flight per worker. Larger = better GPU feed
    # at the cost of host RAM. 4–8 is the sweet spot on a memmap-backed set.
    prefetch_factor: int = 4
    # AMP dtype on CUDA: "bf16" (default on Hopper/Blackwell/Ada — no scaler
    # needed, more numerically stable), or "fp16" (legacy, needs GradScaler).
    amp_dtype: str = "bf16"
    # Wrap the model in torch.compile(). Off by default because the warmup
    # cost rarely pays off for a 4.7M-param CNN: cudnn.benchmark + bf16 + TF32
    # already capture most of the available speedup. Enable only for very long
    # runs where ~5–15% throughput compounds.
    compile: bool = False
    # torch.compile mode. "reduce-overhead" enables CUDA Graphs (the main win
    # for small models, ~30–60s warmup) without per-matmul Triton autotuning.
    # "max-autotune" benchmarks every Triton kernel variant for every matmul —
    # 5–15 minutes of compile time, and for a small CNN the ATen kernels
    # usually win anyway. "default" is the cheapest compile (~10–20s).
    compile_mode: str = "reduce-overhead"
    # Soft cap warning threshold. Above this, num_workers is almost certainly
    # hurting more than helping for a memmap-backed dataset.
    workers_warn_above: int = 32


@dataclass
class VisionTrainResult:
    model: nn.Module
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def _device() -> torch.device:
    # Prefer CUDA (fastest), then Apple MPS, then CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _masked_value_loss(value_pred, value_gt, present_gt):
    # value_pred is already a position in [0,1] (soft-argmax), no sigmoid.
    per = F.smooth_l1_loss(value_pred, value_gt, reduction="none", beta=0.05)
    mask = present_gt
    denom = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / denom


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    abs_err_sum = 0.0
    present_count = 0.0
    tp = fp = fn = 0.0
    scale_correct = 0.0
    scale_total = 0.0
    for x, value_gt, present_gt, scale_gt in loader:
        x = x.to(device)
        value_gt = value_gt.to(device)
        present_gt = present_gt.to(device)
        scale_gt = scale_gt.to(device)
        value_pred, present_logit, scale_logit = model(x)
        pred_val = value_pred  # already in [0,1]
        pred_present = (torch.sigmoid(present_logit) >= PRESENCE_THRESHOLD).float()

        m = present_gt
        abs_err_sum += (torch.abs(pred_val - value_gt) * m).sum().item()
        present_count += m.sum().item()

        tp += ((pred_present == 1) & (present_gt == 1)).sum().item()
        fp += ((pred_present == 1) & (present_gt == 0)).sum().item()
        fn += ((pred_present == 0) & (present_gt == 1)).sum().item()

        # scale accuracy (only over labeled samples, scale_gt >= 0)
        valid = scale_gt >= 0
        if valid.any():
            pred_scale = scale_logit.argmax(dim=1)
            scale_correct += ((pred_scale == scale_gt) & valid).sum().item()
            scale_total += valid.sum().item()

    mae = abs_err_sum / max(1.0, present_count)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    scale_acc = scale_correct / max(1.0, scale_total)
    return {"val_mae_present": mae, "val_presence_f1": f1,
            "presence_precision": prec, "presence_recall": rec,
            "val_scale_acc": scale_acc}


def train(cfg: Optional[VisionTrainConfig] = None) -> VisionTrainResult:
    cfg = cfg or VisionTrainConfig()
    device = _device()
    print(f"[vision] device={device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.train_npz:
        from .dataset import CachedChartDataset

        train_ds = CachedChartDataset(cfg.train_npz, augment=True)
        val_ds = CachedChartDataset(cfg.val_npz, augment=False) if cfg.val_npz else \
            ChartDataset(cfg.val_size, seed=cfg.seed + 7)
        print(f"[vision] cached train set: {len(train_ds)} samples")
    else:
        train_ds = ChartDataset(cfg.train_size, seed=cfg.seed)
        val_ds = ChartDataset(cfg.val_size, seed=cfg.seed + 7)

    use_cuda = device.type == "cuda"
    if use_cuda:
        # cudnn autotuner: inputs are fixed-size, so this picks fast kernels.
        torch.backends.cudnn.benchmark = True
        # Enable TF32 matmul on Ampere+ for ~2x throughput on fp32 ops
        # outside the AMP region (no accuracy impact for this task).
        torch.set_float32_matmul_precision("high")
        cap = torch.cuda.get_device_capability(0)
        print(f"[vision] cuda device: {torch.cuda.get_device_name(0)} "
              f"(sm_{cap[0]}{cap[1]}, "
              f"bf16={'yes' if torch.cuda.is_bf16_supported() else 'no'}, "
              f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")

    # Warn loudly about num_workers >> what a memmap-backed dataset can use.
    # Past ~32, each extra worker just adds OMP threads + RSS for ~no IO gain.
    if cfg.num_workers > cfg.workers_warn_above:
        print(f"[vision] WARNING: num_workers={cfg.num_workers} is unusually high "
              f"for a memmap dataset; 8–16 is typically optimal. Each worker "
              f"forks torch + opens the memmap; >32 workers usually thrashes "
              f"the box and stalls startup. Consider --num-workers 16.")
    # Belt-and-suspenders: even with our worker_init pinning, the *parent's*
    # OMP_NUM_THREADS leaks into the pin_memory thread and a few BLAS calls.
    # Cap it once here if the env didn't already.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Use 'forkserver' for DataLoader workers on Linux. The parent has by now
    # initialized CUDA + cuBLAS + cuDNN (via the device_capability print and
    # set_float32_matmul_precision), and plain fork() of a CUDA-initialized
    # parent is a known source of silent hangs / deadlocks on big boxes.
    # forkserver spawns a clean intermediary that has never touched CUDA,
    # and workers fork from that — fast AND safe.
    mp_ctx = None
    if cfg.num_workers > 0:
        try:
            mp_ctx = mp.get_context("forkserver")
        except (ValueError, AttributeError):
            mp_ctx = None  # macOS / Windows fall back to default (spawn)

    dl_kwargs = dict(
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=use_cuda,
        worker_init_fn=_worker_init if cfg.num_workers > 0 else None,
    )
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = cfg.prefetch_factor
        if mp_ctx is not None:
            dl_kwargs["multiprocessing_context"] = mp_ctx
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          drop_last=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        **dl_kwargs)

    model = build_model(width=cfg.width).to(device)
    n_params = count_params(model)
    print(f"[vision] params: {n_params:,} ({n_params/1e6:.2f}M)")

    if cfg.init_ckpt:
        blob = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)
        ckpt_width = blob.get("width")
        if ckpt_width is not None and abs(float(ckpt_width) - cfg.width) > 1e-6:
            raise ValueError(
                f"--init-ckpt width {ckpt_width} != --width {cfg.width}; "
                "they must match to load the weights."
            )
        model.load_state_dict(blob["state_dict"])
        print(f"[vision] initialized from {cfg.init_ckpt} "
              f"(base val_mae={blob.get('best_val_mae')})")

    if cfg.freeze_backbone:
        frozen = 0
        for p in model.stem.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
        for p in model.blocks.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
        print(f"[vision] froze backbone (stem+blocks): {frozen:,} params "
              "— only width_refine + heads will train")

    # Keep a handle on the un-compiled module for state_dict / checkpoint /
    # export — torch.compile wraps it and renames every key with a `_orig_mod.`
    # prefix that breaks downstream loads.
    raw_model = model
    if cfg.compile and use_cuda:
        print(f"[vision] torch.compile(model, mode={cfg.compile_mode!r})  "
              f"(first batch warmup ~"
              f"{'10-20s' if cfg.compile_mode == 'default' else '30-60s' if cfg.compile_mode == 'reduce-overhead' else '5-15 MIN'})")
        if cfg.compile_mode == "max-autotune":
            print("[vision] NOTE: max-autotune benchmarks every Triton kernel "
                  "variant for every matmul. For a small CNN this usually costs "
                  "many minutes of compile time for ~0% gain — prefer "
                  "'reduce-overhead' unless you've measured a win.")
        model = torch.compile(model, mode=cfg.compile_mode)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, cfg.epochs * (len(train_ds) // cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    bce = nn.BCEWithLogitsLoss()
    # scale classifier: ignore_index=-1 means generic (arbitrary-axis) charts
    # contribute no scale gradient — only premom charts supervise the unit.
    ce_scale = nn.CrossEntropyLoss(ignore_index=-1)

    # Mixed precision (CUDA only). bf16 is preferred on Ampere+/Hopper/Blackwell:
    # same speed as fp16, wider dynamic range, no GradScaler needed. Fall back to
    # fp16 if the card pre-dates bf16 (or the user forced it).
    if use_cuda and cfg.amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_cuda and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"[vision] amp dtype={amp_dtype} scaler={'on' if use_scaler else 'off'} "
          f"batch={cfg.batch_size} workers={cfg.num_workers} "
          f"prefetch={cfg.prefetch_factor}")

    best_mae = float("inf")
    best_state = None
    history = []

    total_batches = max(1, len(train_ds) // cfg.batch_size)
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        # Tell the user the loop has entered — the time between this line
        # and the "first batch" line is worker spawn + prefetch fill + cold
        # page cache reads + cudnn.benchmark warmup. Can be 2-5 minutes on
        # the first epoch with a large memmap dataset; near-instant on later
        # epochs once the kernel has cached the file.
        print(f"[vision] epoch {epoch+1}/{cfg.epochs}: starting "
              f"({total_batches} batches; workers warming up...)", flush=True)
        last_log = t0
        for batch_idx, (x, value_gt, present_gt, scale_gt) in enumerate(train_dl):
            if batch_idx == 0:
                dt = time.time() - t0
                print(f"  first batch fetched in {dt:.1f}s "
                      f"(worker spawn + prefetch + cudnn autotune)", flush=True)
            x = x.to(device, non_blocking=use_cuda)
            value_gt = value_gt.to(device, non_blocking=use_cuda)
            present_gt = present_gt.to(device, non_blocking=use_cuda)
            scale_gt = scale_gt.to(device, non_blocking=use_cuda)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_cuda,
            ):
                value_pred, present_logit, scale_logit = model(x)
                l_val = _masked_value_loss(value_pred, value_gt, present_gt)
                l_pres = bce(present_logit, present_gt)
                # scale CE only over labeled (premom) samples; if a batch has
                # none, contribute 0 (avoids NaN from all-ignored CE).
                if (scale_gt >= 0).any():
                    l_scale = ce_scale(scale_logit, scale_gt)
                else:
                    l_scale = present_logit.sum() * 0.0
                loss = (
                    cfg.value_loss_weight * l_val
                    + cfg.presence_loss_weight * l_pres
                    + cfg.scale_loss_weight * l_scale
                )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                # bf16 (or fp32) path — gradients stay in range, no scaler needed.
                loss.backward()
                opt.step()
            sched.step()
            running += loss.item()
            nb += 1

            now = time.time()
            if now - last_log >= 15.0:
                # Periodic in-epoch heartbeat so you can see things are alive
                # and gauge throughput without waiting for the full epoch.
                samples = (batch_idx + 1) * cfg.batch_size
                rate = samples / (now - t0)
                print(f"  [epoch {epoch+1}] batch {batch_idx+1}/{total_batches} "
                      f"loss={running/nb:.4f} ({rate:.0f} samples/s)", flush=True)
                last_log = now

        metrics = evaluate(model, val_dl, device)
        dt = time.time() - t0
        print(
            f"[vision] epoch {epoch+1}/{cfg.epochs} "
            f"loss={running/max(1,nb):.4f} "
            f"val_mae={metrics['val_mae_present']:.4f} "
            f"val_f1={metrics['val_presence_f1']:.4f} "
            f"val_scale_acc={metrics['val_scale_acc']:.3f} "
            f"({dt:.0f}s)"
        )
        history.append({"epoch": epoch + 1, **metrics})
        if metrics["val_mae_present"] < best_mae:
            best_mae = metrics["val_mae_present"]
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            # Persist immediately so a teardown crash never loses the model.
            if cfg.checkpoint_path:
                torch.save(
                    {
                        "state_dict": best_state,
                        "width": cfg.width,
                        "best_val_mae": best_mae,
                        "metrics": metrics,
                        "config": cfg.__dict__,
                    },
                    cfg.checkpoint_path,
                )

    if best_state is not None:
        raw_model.load_state_dict(best_state)

    final = evaluate(model, val_dl, device)

    # Explicitly tear down DataLoader workers BEFORE returning. On macOS +
    # Python 3.12, leaving persistent workers to be garbage-collected during
    # interpreter teardown can raise a spurious worker SIGSEGV. Deleting the
    # iterators/loaders here shuts workers down cleanly while we're still in a
    # controlled state.
    try:
        del train_dl
        del val_dl
    except Exception:
        pass
    import gc

    gc.collect()

    return VisionTrainResult(
        model=raw_model.cpu(),
        metrics={
            "n_params": n_params,
            "best_val_mae_present": best_mae,
            "final": final,
            "history": history,
        },
        config=cfg.__dict__,
    )
