"""Training pipeline for the FertiLuna cycle classifier.

Trains three artifacts:
    classifier (calibrated)  — main RF + sigmoid (Platt) calibration. Output:
                                per-class probabilities used by the UI to gate
                                the "données insuffisantes" decision (< 0.6 max
                                proba → low-confidence display).
    isolation_forest         — unsupervised anomaly score on the feature vector.
                                Used as an out-of-distribution backstop in the UI.

Why calibration: raw RF probabilities are pushed toward 0/1; without calibration
the "< 0.6 → unknown" threshold isn't statistically meaningful. CalibratedClassifierCV
with sigmoid fits Platt scaling per class on held-out folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

from .constants import LABELS, N_FEATURES
from .features import batch_extract
from .synthetic import generate_dataset


@dataclass
class TrainConfig:
    n_samples: int = 50_000
    seed: int = 42
    test_size: float = 0.2
    # Held-out fraction (of the train split) used only to fit the calibrator.
    calib_size: float = 0.2
    rf_n_estimators: int = 120
    rf_max_depth: Optional[int] = 12
    rf_min_samples_leaf: int = 12
    rf_max_features: str = "sqrt"
    rf_class_weight: str = "balanced"
    iforest_contamination: float = 0.10
    iforest_n_estimators: int = 150


@dataclass
class TrainResult:
    classifier: CalibratedClassifierCV
    isolation_forest: IsolationForest
    metrics: dict = field(default_factory=dict)
    feature_means: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    feature_stds: np.ndarray = field(default_factory=lambda: np.ones(N_FEATURES))
    # Distribution of iforest decision_function scores on the (in-distribution)
    # training set. The browser uses these to map a raw anomaly score into a
    # human-friendly "this curve is unusual" percentile.
    iforest_score_p5: float = 0.0
    iforest_score_p50: float = 0.0
    iforest_score_p95: float = 0.0


def _build_dataset(cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    temps, lh, y, _truths = generate_dataset(n=cfg.n_samples, seed=cfg.seed)
    X = batch_extract(temps, lh)
    return X, y


def train(cfg: Optional[TrainConfig] = None) -> TrainResult:
    cfg = cfg or TrainConfig()
    print(f"[train] generating {cfg.n_samples} synthetic cycles ...")
    X, y = _build_dataset(cfg)
    print(f"[train] dataset: X={X.shape}  y={y.shape}  classes={np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    # Split the train portion into a fit set (trains the forest) and a
    # calibration set (fits the Platt scaler). Using a single prefit forest
    # keeps the exported ONNX graph to ONE tree ensemble instead of the five
    # that CalibratedClassifierCV(cv=5) would create — critical for a model
    # that has to download and cache in the browser.
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=cfg.calib_size,
        random_state=cfg.seed,
        stratify=y_train,
    )

    base_rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        max_features=cfg.rf_max_features,
        class_weight=cfg.rf_class_weight,
        n_jobs=-1,
        random_state=cfg.seed,
    )

    print("[train] fitting random forest on fit set ...")
    base_rf.fit(X_fit, y_fit)

    print("[train] calibrating (Platt sigmoid) on held-out calibration set ...")
    # sklearn >=1.6 replaced cv="prefit" with the FrozenEstimator wrapper:
    # wrap the already-fitted forest so CalibratedClassifierCV only fits the
    # Platt scaler and reuses the single underlying ensemble.
    from sklearn.frozen import FrozenEstimator

    clf = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_rf), method="sigmoid"
    )
    clf.fit(X_calib, y_calib)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    acc = float(np.mean(y_pred == y_test))
    ll = float(log_loss(y_test, y_proba, labels=list(range(len(LABELS)))))
    report = classification_report(
        y_test, y_pred, target_names=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABELS))))

    print(f"[train] accuracy:  {acc:.4f}")
    print(f"[train] log_loss:  {ll:.4f}")
    print("[train] confusion matrix (rows=true, cols=pred):")
    print(cm)
    for i, name in enumerate(LABELS):
        prec = report[name]["precision"]
        rec = report[name]["recall"]
        f1 = report[name]["f1-score"]
        print(f"  {name:28s}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")

    print("[train] fitting isolation forest for OOD detection ...")
    iforest = IsolationForest(
        n_estimators=cfg.iforest_n_estimators,
        contamination=cfg.iforest_contamination,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    iforest.fit(X_train)

    if_scores = iforest.decision_function(X_train)
    if_p5, if_p50, if_p95 = (
        float(np.percentile(if_scores, 5)),
        float(np.percentile(if_scores, 50)),
        float(np.percentile(if_scores, 95)),
    )
    print(
        f"[train] iforest decision_function percentiles "
        f"p5={if_p5:.4f} p50={if_p50:.4f} p95={if_p95:.4f}"
    )

    metrics = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "accuracy": acc,
        "log_loss": ll,
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in LABELS
        },
        "confusion_matrix": cm.tolist(),
        "config": cfg.__dict__,
    }

    feature_means = X_train.mean(axis=0).astype(np.float32)
    feature_stds = X_train.std(axis=0).astype(np.float32) + 1e-6

    return TrainResult(
        classifier=clf,
        isolation_forest=iforest,
        metrics=metrics,
        feature_means=feature_means,
        feature_stds=feature_stds,
        iforest_score_p5=if_p5,
        iforest_score_p50=if_p50,
        iforest_score_p95=if_p95,
    )


def save_metrics(result: TrainResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.metrics, f, indent=2)
    print(f"[train] metrics written to {path}")
