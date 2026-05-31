"""The chart-vision CNN (v2 — soft-argmax height decoding).

A compact depthwise-separable encoder maps the RGB chart to a feature map. The
day axis runs along WIDTH; the value of a curve in a given column is its
VERTICAL position. v1 mean-pooled the height away before the value head, which
destroyed exactly that signal and forced the net to smuggle height info through
channel statistics — fragile, and it could not disambiguate the overlapping
BBT/LH lines near ovulation.

v2 fixes this with per-series soft-argmax over height (a clean, collapsed
heatmap model):

    heat:  (B, S, h, w)                 one height-attention map per series
    prob = softmax(heat, dim=h)          where (vertically) is the curve?
    ys   = linspace(1..0, h)             top of plot = 1.0, bottom = 0.0
    value = Σ_h prob * ys  →  (B, S, w)  expected vertical position == the value

So "value = vertical position" is true BY CONSTRUCTION, each series has its own
channel (overlap disambiguation for free), and the output is always in [0,1]
and sane even on absent days (Fix 3). Presence is a separate per-series head.

Resolution: the encoder stops at /8 (h≈28 for a 224-tall input) so the value
path keeps enough vertical resolution to resolve the ~18 BBT gridlines (Fix 2).

Everything is fixed-shape and conv-only → clean ONNX export, fast in WASM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import IMG_H, IMG_W, N_DAYS, N_SERIES, N_BBT_SCALES


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWSep(nn.Module):
    """Depthwise-separable conv block with optional stride."""

    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, k=3, s=s, groups=cin)
        self.pw = ConvBNAct(cin, cout, k=1, s=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class ChartVisionNet(nn.Module):
    def __init__(self, width: float = 1.0):
        super().__init__()

        def c(ch):
            return max(8, int(round(ch * width)))

        # Encoder: stop at /8 to keep vertical resolution for height decoding.
        self.stem = ConvBNAct(3, c(32), k=3, s=2)  # /2
        self.blocks = nn.Sequential(
            DWSep(c(32), c(64), s=1),
            DWSep(c(64), c(128), s=2),    # /4
            DWSep(c(128), c(128), s=1),
            DWSep(c(128), c(256), s=2),   # /8
            DWSep(c(256), c(256), s=1),
            DWSep(c(256), c(256), s=1),
        )
        feat = c(256)

        # ── value path: per-series height-attention heatmap ──
        # A couple of convs refine the feature map, then a 1x1 produces one
        # heatmap channel per series. We soft-argmax over height in forward().
        self.heat_refine = nn.Sequential(
            DWSep(feat, feat, s=1),
            ConvBNAct(feat, feat, k=1),
        )
        self.heat_head = nn.Conv2d(feat, N_SERIES, 1)

        # ── presence path: collapse height (mean) then a 1D width head ──
        # Presence only needs "is there a curve in this column for this series",
        # so height-pooling is fine here.
        self.pres_refine = nn.Sequential(
            nn.Conv1d(feat, feat, 3, padding=1, groups=feat, bias=False),
            nn.BatchNorm1d(feat),
            nn.ReLU6(inplace=True),
            nn.Conv1d(feat, feat, 1, bias=False),
            nn.BatchNorm1d(feat),
            nn.ReLU6(inplace=True),
        )
        self.present_head = nn.Conv1d(feat, N_SERIES, 1)

        # ── scale path: classify the BBT axis convention (celsius / fahrenheit) ──
        # The axis tick labels + gridline spacing distinguish C (35-37) from
        # F (95-99). A global-pooled feature → small MLP → N_BBT_SCALES logits.
        self.scale_head = nn.Sequential(
            nn.Linear(feat, feat),
            nn.ReLU6(inplace=True),
            nn.Linear(feat, N_BBT_SCALES),
        )

        # temperature for the height softmax (lower = sharper localization)
        self.heat_temp = 0.5

    def forward(self, x):
        # x: (B,3,H,W)
        x = self.stem(x)
        x = self.blocks(x)                 # (B, C, h, w)

        # ── value via per-series soft-argmax over height ──
        heat = self.heat_head(self.heat_refine(x))   # (B, S, h, w)
        B, S, h, w = heat.shape
        prob = F.softmax(heat / self.heat_temp, dim=2)  # over height
        # vertical positions: row 0 (top of plot) = 1.0, last row (bottom) = 0.0
        ys = torch.linspace(1.0, 0.0, h, device=x.device, dtype=x.dtype)
        ys = ys.view(1, 1, h, 1)
        value_w = (prob * ys).sum(dim=2)             # (B, S, w)  in [0,1]
        # resample width -> N_DAYS
        value = F.interpolate(value_w, size=N_DAYS, mode="linear",
                              align_corners=False)   # (B, S, N_DAYS)

        # ── presence ──
        feat_w = x.mean(dim=2)                        # (B, C, w)
        feat_w = self.pres_refine(feat_w)
        present_w = self.present_head(feat_w)         # (B, S, w)
        present = F.interpolate(present_w, size=N_DAYS, mode="linear",
                                align_corners=False)  # logits

        # ── scale (BBT unit) ──
        pooled = x.mean(dim=(2, 3))                   # (B, C) global avg pool
        scale = self.scale_head(pooled)               # (B, N_BBT_SCALES) logits

        # value is a position in [0,1] by construction; present + scale are
        # logits (browser applies sigmoid / argmax respectively).
        return value, present, scale


def build_model(width: float = 1.0) -> ChartVisionNet:
    return ChartVisionNet(width=width)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
