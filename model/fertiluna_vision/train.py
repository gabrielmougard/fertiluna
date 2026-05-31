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

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import PRESENCE_THRESHOLD
from .dataset import ChartDataset
from .model import build_model, count_params


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

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=use_cuda,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
        pin_memory=use_cuda,
    )

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

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, cfg.epochs * (len(train_ds) // cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    bce = nn.BCEWithLogitsLoss()
    # scale classifier: ignore_index=-1 means generic (arbitrary-axis) charts
    # contribute no scale gradient — only premom charts supervise the unit.
    ce_scale = nn.CrossEntropyLoss(ignore_index=-1)

    # Mixed precision (CUDA only) — big speedup, negligible accuracy impact.
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_mae = float("inf")
    best_state = None
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for x, value_gt, present_gt, scale_gt in train_dl:
            x = x.to(device, non_blocking=use_cuda)
            value_gt = value_gt.to(device, non_blocking=use_cuda)
            present_gt = present_gt.to(device, non_blocking=use_cuda)
            scale_gt = scale_gt.to(device, non_blocking=use_cuda)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
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

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            nb += 1

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        model.load_state_dict(best_state)

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
        model=model.cpu(),
        metrics={
            "n_params": n_params,
            "best_val_mae_present": best_mae,
            "final": final,
            "history": history,
        },
        config=cfg.__dict__,
    )
