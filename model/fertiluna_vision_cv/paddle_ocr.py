"""PaddleOCR PP-OCRv3 recognition backend.

The fertiluna_vision_cv pipeline locates text bboxes itself (axis ticks,
table cells), so we ONLY need PaddleOCR's recognition model — no detection,
no classification. That makes the integration tiny: one ONNX session and a
~30-line preprocessing / CTC-decode pair.

Input contract (PP-OCRv3 English rec model):
    * BGR or RGB crop of the text region (we accept BGR, convert internally).
    * Resize to fixed HEIGHT=48 keeping aspect; width is padded to a multiple
      of 32 (model is fully convolutional in width).
    * Normalize: (pixel/255 - 0.5) / 0.5  →  range ~[-1, 1].
    * Layout: NCHW float32.

Output: (1, T, num_classes) logits.
CTC decode: per timestep argmax; collapse repeats; drop blank (class 0).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_BLANK = "<blank>"
# PP-OCRv3 rec models trail a single space token at end of the dict;
# en_dict.txt has no space, the model adds it at runtime. We mirror that.
_TRAILING_SPACE = True


class PaddleOCRRec:
    """ONNXRuntime-driven PP-OCRv3 recognizer."""

    def __init__(
        self,
        model_path: str | Path,
        dict_path: str | Path,
        input_height: int = 48,
        width_multiple: int = 32,
        max_width: int = 640,
    ) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        # The model's only input — name varies (some PP-OCRv3 exports use
        # "x", others "image"). Read it from the session.
        self.input_name = self.session.get_inputs()[0].name
        self.input_h = input_height
        self.width_multiple = width_multiple
        self.max_width = max_width
        # Build the char vocab. Index 0 is the CTC blank by convention.
        # PP-OCRv3 ALWAYS appends a trailing space class on top of whatever
        # the dict file contains, so the output class count is len(dict)+2
        # (blank + trailing-space). The English dict file already ends with
        # a space line, but PaddleOCR's postprocess adds another regardless.
        chars = Path(dict_path).read_text(encoding="utf-8").splitlines()
        chars.append(" ")
        self.vocab = [_BLANK] + chars

    # ── preprocessing ──────────────────────────────────────────────────────
    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((1, 3, self.input_h, self.width_multiple),
                            dtype=np.float32)
        # resize so height = self.input_h, width scaled proportionally
        new_w = max(self.width_multiple,
                    int(round(w * self.input_h / h)))
        new_w = min(self.max_width, new_w)
        # round UP to nearest width_multiple
        new_w = ((new_w + self.width_multiple - 1) // self.width_multiple) * \
                self.width_multiple
        resized = cv2.resize(rgb, (new_w, self.input_h),
                             interpolation=cv2.INTER_LINEAR)
        # normalize: (px/255 - 0.5) / 0.5  →  [-1, 1]
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        # HWC → CHW; batch dim
        arr = arr.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(arr, dtype=np.float32)

    # ── CTC decode ─────────────────────────────────────────────────────────
    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy CTC: argmax per timestep, collapse repeats, drop blanks.
        Returns (text, mean_conf)."""
        # logits: (1, T, C)
        probs = logits[0]                     # (T, C)
        cls = probs.argmax(axis=1)            # (T,)
        # softmax for confidence — only need it on the argmax column.
        # Stable softmax row-by-row.
        m = probs.max(axis=1, keepdims=True)
        exp = np.exp(probs - m)
        s = exp.sum(axis=1, keepdims=True)
        conf_per_t = (exp / s)[np.arange(len(cls)), cls]
        # collapse repeats + drop blank (idx 0)
        out_chars: list[str] = []
        out_confs: list[float] = []
        prev = -1
        for t, c in enumerate(cls):
            if c == prev:
                continue
            prev = int(c)
            if c == 0 or c >= len(self.vocab):
                continue
            out_chars.append(self.vocab[c])
            out_confs.append(float(conf_per_t[t]))
        text = "".join(out_chars).strip()
        mean_conf = float(np.mean(out_confs)) if out_confs else 0.0
        return text, mean_conf

    # ── public API ─────────────────────────────────────────────────────────
    def ocr(self, img_bgr: np.ndarray) -> str:
        text, _ = self.ocr_with_conf(img_bgr)
        return text

    def ocr_with_conf(self, img_bgr: np.ndarray) -> tuple[str, float]:
        if img_bgr is None or img_bgr.size == 0:
            return "", 0.0
        x = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]
        return self._ctc_decode(out)
