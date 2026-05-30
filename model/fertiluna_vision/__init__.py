"""FertiLuna chart-vision model.

A small CNN that maps a cycle-chart screenshot to per-day BBT and LH values
(normalized curve shape + presence mask), trained entirely on synthetic charts
we render ourselves — so labels are free and exact.

Package layout:
    constants   — image size, output dims, series indices (kept in sync with TS)
    render      — synthetic chart renderer (matplotlib) + ground-truth labels
    dataset     — torch Dataset wrapping the renderer
    model       — the CNN (MobileNet-style backbone + regression/presence heads)
    train       — training loop with masked losses
    export_onnx — torch -> ONNX + manifest (same pattern as the cycle classifier)
"""

__version__ = "0.1.0"
