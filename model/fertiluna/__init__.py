"""FertiLuna cycle-analysis model.

Package layout:
    constants  — shared constants (cycle length, label codes, feature names)
    synthetic  — physiologically-grounded synthetic cycle generator
    features   — feature extraction (matches the TS port in src/lib/features.ts)
    train      — training pipeline (RF + Platt calibration + Isolation Forest backstop)
    export_onnx — sklearn -> ONNX export
"""

__version__ = "0.1.0"
