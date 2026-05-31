"""Export the trained chart-vision model to ONNX for in-browser inference.

Produces:
    chart-vision-v1.onnx     — input image (1,3,IMG_H,IMG_W) float32 (normalized),
                                outputs value (1,N_SERIES,N_DAYS) and present
                                (1,N_SERIES,N_DAYS) as RAW LOGITS. The browser
                                applies sigmoid (cheap) so we keep the graph
                                export-clean and identical to training.
    chart-vision-manifest-v1.json — version, IO spec, normalization constants,
                                image size, series names, metrics, checksums.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .constants import (
    IMG_H,
    IMG_W,
    N_DAYS,
    N_SERIES,
    NORM_MEAN,
    NORM_STD,
    PRESENCE_THRESHOLD,
    SERIES_NAMES,
    BBT_SCALES,
    LH_RANGE,
)
from .train import VisionTrainResult


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export(result: VisionTrainResult, out_dir: Path, version: str = "v1") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = result.model.eval()

    onnx_path = out_dir / f"chart-vision-{version}.onnx"
    dummy = torch.zeros(1, 3, IMG_H, IMG_W, dtype=torch.float32)

    print(f"[export] tracing chart-vision -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["value", "present", "scale"],
        dynamic_axes={
            "image": {0: "batch"},
            "value": {0: "batch"},
            "present": {0: "batch"},
            "scale": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Guarantee a SINGLE self-contained .onnx file (no external .onnx.data
    # sidecar). The torch exporter sometimes splits large initializers into an
    # external-data file, which complicates browser fetching/caching. Reload and
    # re-save with all tensor data inlined into the one protobuf.
    import onnx

    loaded = onnx.load(str(onnx_path), load_external_data=True)
    onnx.convert_model_to_external_data(loaded, location="__never__", size_threshold=2**31)
    # The line above marks tensors as external only above a 2GiB threshold (i.e.
    # effectively never), forcing everything to stay inline on save.
    onnx.save_model(loaded, str(onnx_path), save_as_external_data=False)
    # Remove any sidecar the tracer may have written.
    sidecar = Path(str(onnx_path) + ".data")
    if sidecar.exists():
        sidecar.unlink()

    # ── parity check: torch vs onnxruntime ──
    print("[export] parity check (torch vs onnxruntime) ...")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3, IMG_H, IMG_W)).astype(np.float32)
    with torch.no_grad():
        tv, tp, ts = model(torch.from_numpy(x))
    tv = tv.numpy()
    tp = tp.numpy()
    ts = ts.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ov, op, os_ = sess.run(["value", "present", "scale"], {"image": x})
    v_diff = float(np.max(np.abs(tv - ov)))
    p_diff = float(np.max(np.abs(tp - op)))
    s_diff = float(np.max(np.abs(ts - os_)))
    print(f"[export] value  max|torch-onnx| = {v_diff:.6e}")
    print(f"[export] present max|torch-onnx| = {p_diff:.6e}")
    print(f"[export] scale   max|torch-onnx| = {s_diff:.6e}")
    if max(v_diff, p_diff, s_diff) > 1e-3:
        raise RuntimeError("ONNX vision model diverges from torch")

    manifest = {
        "version": version,
        "task": "chart-to-series",
        "image": {"height": IMG_H, "width": IMG_W, "channels": 3,
                  "norm_mean": list(NORM_MEAN), "norm_std": list(NORM_STD),
                  "layout": "NCHW"},
        "output": {
            "n_series": N_SERIES,
            "series_names": SERIES_NAMES,
            "n_days": N_DAYS,
            "value": "already normalized [0,1] (soft-argmax vertical position); use directly, NO sigmoid",
            "present": "raw logit; apply sigmoid -> probability",
            "presence_threshold": PRESENCE_THRESHOLD,
            "scale": "BBT-axis class logits; argmax -> index into bbt_scales below",
            "bbt_scales": [
                {"label": name, "min": lo, "max": hi}
                for name, (lo, hi) in BBT_SCALES
            ],
            "lh_range": {"min": LH_RANGE[0], "max": LH_RANGE[1]},
        },
        "files": {
            "model": {
                "path": onnx_path.name,
                "sha256": _sha256(onnx_path),
                "bytes": onnx_path.stat().st_size,
            }
        },
        "metrics": result.metrics,
        "parity": {"value_max_abs_diff": v_diff, "present_max_abs_diff": p_diff,
                   "scale_max_abs_diff": s_diff},
    }
    manifest_path = out_dir / f"chart-vision-manifest-{version}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export] manifest -> {manifest_path}")
    return manifest
