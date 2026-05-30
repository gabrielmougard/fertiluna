"""The chart-vision CNN.

A compact, fully-convolutional encoder (MobileNet-style depthwise-separable
blocks) maps the RGB chart to a feature map, which is then collapsed along the
HEIGHT axis and resampled along the WIDTH axis to exactly N_DAYS columns. Two
small heads produce, per (series, day):
    value_logit   -> sigmoid -> normalized value in [0,1]
    present_logit -> sigmoid -> probability a point exists

Rationale for the architecture:
    - Charts encode the day axis along width, so we keep width resolution and
      pool height — a natural inductive bias for "read the curve column by
      column" (the same idea as the classical column-scan, learned).
    - Depthwise-separable convs keep the model small (<10M params) and ONNX/WASM
      friendly. No attention, no dynamic shapes → clean export, fast in-browser.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import IMG_H, IMG_W, N_DAYS, N_SERIES


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

        self.stem = ConvBNAct(3, c(32), k=3, s=2)  # H/2, W/2
        self.blocks = nn.Sequential(
            DWSep(c(32), c(64), s=1),
            DWSep(c(64), c(128), s=2),   # /4
            DWSep(c(128), c(128), s=1),
            DWSep(c(128), c(256), s=2),  # /8
            DWSep(c(256), c(256), s=1),
            DWSep(c(256), c(384), s=2),  # /16
            DWSep(c(384), c(384), s=1),
        )
        feat = c(384)

        # Collapse the height dimension to 1 (adaptive), keep width.
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        # A small 1D refinement along width, then project width -> N_DAYS.
        self.width_refine = nn.Sequential(
            nn.Conv1d(feat, feat, 3, padding=1, groups=feat, bias=False),
            nn.BatchNorm1d(feat),
            nn.ReLU6(inplace=True),
            nn.Conv1d(feat, feat, 1, bias=False),
            nn.BatchNorm1d(feat),
            nn.ReLU6(inplace=True),
        )

        # Two heads, each producing N_SERIES channels over N_DAYS positions.
        self.value_head = nn.Conv1d(feat, N_SERIES, 1)
        self.present_head = nn.Conv1d(feat, N_SERIES, 1)

    def forward(self, x):
        # x: (B,3,H,W)
        x = self.stem(x)
        x = self.blocks(x)            # (B, C, h, w)
        x = self.height_pool(x)       # (B, C, 1, w)
        x = x.squeeze(2)              # (B, C, w)
        x = self.width_refine(x)      # (B, C, w)
        # resample width -> N_DAYS
        x = F.interpolate(x, size=N_DAYS, mode="linear", align_corners=False)
        value = self.value_head(x)    # (B, N_SERIES, N_DAYS)
        present = self.present_head(x)
        return value, present


def build_model(width: float = 1.0) -> ChartVisionNet:
    return ChartVisionNet(width=width)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
