"""OCR backend abstraction.

Three backends, all browser/WASM-friendly:

  * NoOCR        — skip OCR entirely.
  * TemplateOCR  — cv2.putText-rendered Hershey-font templates matched
                   glyph-by-glyph. Zero extra deps; ships with OpenCV(.js).
                   Best-effort accuracy on real fonts.
  * PaddleOCRBackend — PP-OCRv3 English recognition model via
                   onnxruntime (Python) / onnxruntime-web (browser).
                   The recognizer-only flavor: this package's CV pipeline
                   already locates text bboxes, so we don't need the
                   detector or classifier. Bundle: ~10 MB ONNX + 50 KB
                   char dict. Accurate on real-app sans-serif fonts.

The PaddleOCR backend is built by running:
    python -m scripts.build_paddleocr_onnx --out ../public/models
which downloads PP-OCRv3 and converts via paddle2onnx. The browser-side
TS wrapper at src/lib/paddleOcr.ts loads the SAME files via ORT-Web.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class OCRBackend(Protocol):
    name: str

    def ocr(self, img_bgr: np.ndarray) -> str: ...


class NoOCR:
    """No-op backend. Returns empty string for everything.

    The pipeline still works: axis_ticks falls back to pure-geometric
    detection (AP filter on label rows), and table_extract returns
    presence markers instead of OCR'd text.
    """
    name = "noop"

    def ocr(self, img_bgr: np.ndarray) -> str:
        return ""


class TemplateOCR:
    """Lightweight cv2.putText-template matching. Ships in OpenCV core, so
    portable to OpenCV.js / WASM without extra weight."""
    name = "template"

    def ocr(self, img_bgr: np.ndarray) -> str:
        # Lazy import keeps this module load-cheap when only NoOCR is used.
        from .axis_ticks import _ocr_label_image
        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr
        return _ocr_label_image(gray)


class PaddleOCRBackend:
    """PP-OCRv3 English recognition via ONNXRuntime.

    Expects the ONNX file + char dict produced by
    `scripts/build_paddleocr_onnx.py`. By default looks in
    `<repo>/public/models` so it shares the same path the browser uses,
    but you can pass explicit paths.
    """
    name = "paddle"

    def __init__(
        self,
        model_path: str | None = None,
        dict_path: str | None = None,
        version: str = "v1",
    ) -> None:
        from pathlib import Path
        from .paddle_ocr import PaddleOCRRec
        if model_path is None or dict_path is None:
            # Default to <repo>/public/models alongside the other ONNX assets.
            here = Path(__file__).resolve()
            for parent in here.parents:
                cand = parent / "public" / "models"
                if cand.exists():
                    if model_path is None:
                        model_path = str(cand / f"paddle-ocr-rec-{version}.onnx")
                    if dict_path is None:
                        dict_path = str(cand / f"paddle-ocr-dict-{version}.txt")
                    break
        if not (model_path and Path(model_path).exists()):
            raise RuntimeError(
                f"PaddleOCR ONNX not found at {model_path}. Run "
                "`python -m scripts.build_paddleocr_onnx --out ../public/models` "
                "from the model/ directory first."
            )
        # Cache the recognizer (ONNX session) by file paths. Constructing a
        # PaddleOCRBackend per image — which run_pipeline does — would
        # otherwise reload the ~9 MB model every call. The cache makes the
        # eval loop and the per-request server path reuse one warm session.
        self._rec = _get_cached_rec(model_path, dict_path)

    def ocr(self, img_bgr: np.ndarray) -> str:
        return self._rec.ocr(img_bgr)

    def ocr_with_conf(self, img_bgr: np.ndarray) -> tuple[str, float]:
        return self._rec.ocr_with_conf(img_bgr)


# Module-level recognizer cache, keyed by (model_path, dict_path). Shared
# across every PaddleOCRBackend so the ONNX session loads at most once.
_REC_CACHE: dict[tuple[str, str], object] = {}


def _get_cached_rec(model_path: str, dict_path: str):
    from .paddle_ocr import PaddleOCRRec
    key = (str(model_path), str(dict_path))
    rec = _REC_CACHE.get(key)
    if rec is None:
        rec = PaddleOCRRec(model_path, dict_path)
        _REC_CACHE[key] = rec
    return rec


def make_ocr(prefer: str = "template") -> OCRBackend:
    """Return the requested backend.

    `prefer`:
        "template" (default) — Hershey template matcher; pure OpenCV.
        "paddle"             — PP-OCRv3 ONNX via ONNXRuntime. Falls back to
                               TemplateOCR if the ONNX file isn't built yet.
        "none"               — skip OCR entirely.
    """
    if prefer == "none":
        return NoOCR()
    if prefer == "paddle":
        try:
            return PaddleOCRBackend()
        except Exception as e:
            print(f"[ocr] PaddleOCR unavailable ({e}); falling back to template.")
    return TemplateOCR()
