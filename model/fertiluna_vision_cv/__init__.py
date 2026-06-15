"""FertiLuna chart-vision — classical CV alternative.

A hybrid computer-vision digitizer that produces the SAME outputs as
`fertiluna_vision` (the CNN) but using only OpenCV primitives — no neural net,
no torch, no ONNX runtime. The aim is the smallest, most debuggable pipeline
that's good enough for the Premom-family fertility charts the product targets.

Pipeline (see pipeline.py):
    image
      └─> preprocess        : load, EXIF-rotate, upscale to working width
      └─> plot_region       : detect chart bounding box (color-ink + gridlines)
      └─> axis_calibration  : top/bottom y, scale class (C vs F via digit templates)
      └─> color_segmentation: HSV masks for BBT-blue / LH-orange / Level-purple
      └─> marker_detection  : centroids of marker glyphs per series
      └─> day_axis          : cluster x-coords into day cells
      └─> map → (value, present, scale) — identical schema to the CNN

Public:
    run_pipeline(image) -> ChartResult
    render_overlay(image, result) -> overlaid PIL image (assessment)
    CLI: `python -m fertiluna_vision_cv.cli {assess|infer} ...`
"""

from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import ChartResult, run_pipeline
from .debug_overlay import render_overlay

__all__ = ["ChartResult", "run_pipeline", "render_overlay"]
