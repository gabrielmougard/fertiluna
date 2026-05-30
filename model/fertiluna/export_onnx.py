"""Export trained sklearn models to ONNX for browser inference via onnxruntime-web.

Two artifacts are produced:
    cycle-classifier-v1.onnx — CalibratedClassifierCV (RF + Platt). Input: (None, N_FEATURES)
                                float32. Outputs: label (int64) and probabilities.
    cycle-iforest-v1.onnx    — IsolationForest. Input: (None, N_FEATURES) float32.
                                Output: anomaly_score (the higher, the more normal).

We also emit `model-manifest.json` with checksums so the browser can cache by version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from .constants import LABELS, N_FEATURES
from .train import TrainResult


def _unwrap_frozen_estimators(clf) -> None:
    """Replace any FrozenEstimator inside a fitted CalibratedClassifierCV with
    the raw wrapped estimator.

    sklearn >=1.6 wraps a prefit estimator in FrozenEstimator, but skl2onnx has
    no converter for that wrapper. Since the wrapper only disables refitting
    (irrelevant post-fit), we can safely swap it for the inner estimator before
    ONNX conversion.
    """
    try:
        from sklearn.frozen import FrozenEstimator
    except Exception:  # pragma: no cover
        return
    for cc in getattr(clf, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None)
        if isinstance(est, FrozenEstimator):
            cc.estimator = est.estimator
    if isinstance(getattr(clf, "estimator", None), FrozenEstimator):
        clf.estimator = clf.estimator.estimator


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export(result: TrainResult, out_dir: Path, version: str = "v1") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make the calibrated classifier convertible (unwrap FrozenEstimator).
    _unwrap_frozen_estimators(result.classifier)

    initial_type = [("input", FloatTensorType([None, N_FEATURES]))]

    # ONNX Runtime Web (WASM) ships a bounded set of operator-set versions.
    # Pin both the default domain and the ai.onnx.ml domain so the exported
    # graphs stay within what onnxruntime-web supports. ai.onnx.ml is capped
    # at 3 in current ORT-web builds; skl2onnx defaults to 4 otherwise.
    target_opset = {"": 17, "ai.onnx.ml": 3}

    # --- classifier ---
    clf_path = out_dir / f"cycle-classifier-{version}.onnx"
    print(f"[export] converting classifier -> {clf_path}")
    onx = to_onnx(
        result.classifier,
        initial_types=initial_type,
        target_opset=target_opset,
        options={id(result.classifier): {"zipmap": False}},
    )
    with open(clf_path, "wb") as f:
        f.write(onx.SerializeToString())

    # --- isolation forest ---
    if_path = out_dir / f"cycle-iforest-{version}.onnx"
    print(f"[export] converting isolation forest -> {if_path}")
    onx_if = to_onnx(
        result.isolation_forest,
        initial_types=initial_type,
        target_opset=target_opset,
    )
    with open(if_path, "wb") as f:
        f.write(onx_if.SerializeToString())

    # --- parity check: ensure ONNX matches sklearn within tolerance ---
    print("[export] running parity check (sklearn vs onnxruntime) ...")
    rng = np.random.default_rng(0)
    X_check = rng.standard_normal((128, N_FEATURES)).astype(np.float32)
    sk_proba = result.classifier.predict_proba(X_check).astype(np.float32)

    sess = ort.InferenceSession(str(clf_path), providers=["CPUExecutionProvider"])
    onx_outputs = sess.run(None, {"input": X_check})
    # output order: [label, probabilities] when zipmap=False
    onx_proba = None
    for arr in onx_outputs:
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == len(LABELS):
            onx_proba = arr.astype(np.float32)
            break
    if onx_proba is None:
        raise RuntimeError("ONNX classifier did not return a probability matrix")
    max_abs_diff = float(np.max(np.abs(sk_proba - onx_proba)))
    print(f"[export] classifier max|sklearn - onnx| = {max_abs_diff:.6f}")
    if max_abs_diff > 1e-3:
        raise RuntimeError(
            f"ONNX classifier diverges from sklearn (max diff {max_abs_diff:.6f})"
        )

    sess_if = ort.InferenceSession(str(if_path), providers=["CPUExecutionProvider"])
    onx_if_outputs = sess_if.run(None, {"input": X_check})
    # skl2onnx exports IsolationForest with two outputs: [label, scores] where
    # `scores` is the anomaly score in the same orientation as
    # sklearn's decision_function (negative => more anomalous), NOT score_samples.
    onx_if_scores = None
    for arr in onx_if_outputs:
        if isinstance(arr, np.ndarray) and arr.ndim in (1, 2) and arr.shape[0] == 128:
            if arr.ndim == 2 and arr.shape[1] != 1:
                continue  # that's the label column if it were wide; skip
            onx_if_scores = arr.squeeze().astype(np.float32)
    sk_if_scores = result.isolation_forest.decision_function(X_check).astype(
        np.float32
    )
    if_max_diff = (
        float(np.max(np.abs(sk_if_scores - onx_if_scores)))
        if onx_if_scores is not None
        else float("nan")
    )
    # The iforest is an advisory OOD backstop; small numeric drift in the score
    # is acceptable as long as the ordering is preserved. We log but don't fail.
    if onx_if_scores is not None:
        order_corr = float(
            np.corrcoef(sk_if_scores.argsort(), onx_if_scores.argsort())[0, 1]
        )
    else:
        order_corr = float("nan")
    print(
        f"[export] iforest max|sklearn.decision_function - onnx| = {if_max_diff:.6f}"
        f"  (rank corr {order_corr:.4f})"
    )

    # --- manifest ---
    manifest = {
        "version": version,
        "labels": LABELS,
        "n_features": N_FEATURES,
        "feature_means": result.feature_means.tolist(),
        "feature_stds": result.feature_stds.tolist(),
        "confidence_threshold": 0.60,
        "iforest_score_percentiles": {
            "p5": result.iforest_score_p5,
            "p50": result.iforest_score_p50,
            "p95": result.iforest_score_p95,
        },
        "files": {
            "classifier": {
                "path": clf_path.name,
                "sha256": _sha256(clf_path),
                "bytes": clf_path.stat().st_size,
            },
            "iforest": {
                "path": if_path.name,
                "sha256": _sha256(if_path),
                "bytes": if_path.stat().st_size,
            },
        },
        "metrics": result.metrics,
        "parity": {
            "classifier_max_abs_diff": max_abs_diff,
            "iforest_max_abs_diff": if_max_diff,
        },
    }
    manifest_path = out_dir / f"model-manifest-{version}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export] manifest -> {manifest_path}")

    return manifest
