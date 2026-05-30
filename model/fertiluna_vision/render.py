"""Synthetic cycle-chart renderer.

We render thousands of charts that *look like* the screenshots users will paste
(Inito / Premom / Clearblue / paper sympto-thermal charts) and record the EXACT
per-day values that went into each plot. That gives us a perfectly-labeled
dataset for free — the whole point of the synthetic approach.

A rendered sample yields:
    image    : PIL.Image (RGB)
    value    : (N_SERIES, N_DAYS) float32 in [0,1], normalized within each
               series' own y-range (NaN-free; 0 where absent)
    present  : (N_SERIES, N_DAYS) float32 {0,1}
    meta     : dict (axis ranges etc., for debugging)

Heavy visual augmentation is the key to generalization across apps:
    - random day-axis range / offset (chart may start at ZT 6, J1, etc.)
    - dual y-axis with independent, random value ranges per series
    - random line + marker colors per series, random markers/线 styles
    - optional shaded fertile-window bands, gridlines, legends, axis ticks
    - random fonts/sizes, background tints, jpeg-like noise, blur, scaling
    - random subset of series present (temp only / lh only / both)
    - missing days (gaps) in either series
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageFilter

from .constants import IMG_H, IMG_W, N_DAYS, N_SERIES

# Reuse the physiologically-grounded cycle generator from the cycle model so the
# BBT shapes are realistic (rises, plateaus, anovulation, etc.).
from fertiluna.synthetic import generate_cycle


# Palette of plausible line colors seen across real apps.
_LINE_COLORS = [
    "#5678d6", "#d6447f", "#e66e46", "#7e4b9e", "#2e9e6b",
    "#e0408f", "#3a7bd5", "#ff7043", "#8e44ad", "#16a085",
    "#c0392b", "#2c82c9", "#27ae60", "#d35400", "#9b59b6",
]
_BG_TINTS = ["#ffffff", "#fffdfa", "#fbf7fc", "#f7faff", "#fffaf7", "#fcfcfc"]
_GRID_COLORS = ["#e6e6eb", "#ececf2", "#eeeeee", "#e9e3ee", "#f0eef3"]
_BAND_COLORS = ["#e9d8f4", "#f4d8e6", "#d8e4f4", "#efe6f8"]


@dataclass
class ChartSample:
    image: Image.Image
    value: np.ndarray   # (N_SERIES, N_DAYS) in [0,1]
    present: np.ndarray  # (N_SERIES, N_DAYS) {0,1}
    meta: dict


def _series_curves(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (unnormalized) BBT and LH day-series of length N_DAYS, with
    NaN for missing days. Built from the cycle generator + an LH-like signal."""
    sample = generate_cycle(rng)
    temps = sample.temps[:N_DAYS].astype(np.float64).copy()
    lh = sample.lh[:N_DAYS].astype(np.float64).copy()

    # Occasionally drop a series entirely (temp-only or lh-only charts).
    drop = rng.random()
    if drop < 0.18:
        lh[:] = np.nan          # temp only
    elif drop < 0.30:
        temps[:] = np.nan       # lh only

    # Extra random missing days for realism.
    for arr, p in ((temps, 0.08), (lh, 0.10)):
        mask = rng.random(arr.shape) < p
        arr[mask] = np.nan
    return temps, lh


def _normalize_series(
    raw: np.ndarray, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw values into [0,1] given an axis range [lo,hi]; return (value, present)."""
    present = (~np.isnan(raw)).astype(np.float32)
    value = np.zeros(N_DAYS, dtype=np.float32)
    span = max(1e-6, hi - lo)
    for d in range(N_DAYS):
        if present[d]:
            value[d] = float(np.clip((raw[d] - lo) / span, 0.0, 1.0))
    return value, present


def _fig_to_pil(fig: Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.get_dpi())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_chart(rng: np.random.Generator) -> ChartSample:
    py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))

    temps_raw, lh_raw = _series_curves(rng)

    # ── random per-series axis ranges (the chart's own y-scales) ──
    # BBT axis: pick a window that contains the data with random padding.
    def _axis_range(raw: np.ndarray, default_lo, default_hi, pad_lo, pad_hi):
        vals = raw[~np.isnan(raw)]
        if vals.size == 0:
            return default_lo, default_hi
        lo = float(vals.min()) - py_rng.uniform(*pad_lo)
        hi = float(vals.max()) + py_rng.uniform(*pad_hi)
        if hi - lo < 1e-3:
            hi = lo + 1.0
        return lo, hi

    temp_lo, temp_hi = _axis_range(temps_raw, 36.0, 37.2, (0.05, 0.4), (0.05, 0.4))
    lh_lo, lh_hi = _axis_range(lh_raw, 0.0, 3.0, (0.0, 0.2), (0.1, 0.8))

    value = np.zeros((N_SERIES, N_DAYS), dtype=np.float32)
    present = np.zeros((N_SERIES, N_DAYS), dtype=np.float32)
    value[0], present[0] = _normalize_series(temps_raw, temp_lo, temp_hi)
    value[1], present[1] = _normalize_series(lh_raw, lh_lo, lh_hi)

    # ── figure setup ──
    w_in = py_rng.uniform(5.5, 8.0)
    h_in = py_rng.uniform(3.0, 4.6)
    dpi = py_rng.choice([72, 90, 100])
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    bg = py_rng.choice(_BG_TINTS)
    fig.patch.set_facecolor(bg)
    ax_t = fig.add_axes([0.10, 0.16, 0.80, 0.78])
    ax_t.set_facecolor(bg)

    days = np.arange(1, N_DAYS + 1)

    # random day-axis label offset (chart may label ZT 6, J1, etc.)
    day_label_start = py_rng.choice([1, 1, 1, 5, 6, 7, 8])

    grid_color = py_rng.choice(_GRID_COLORS)
    if py_rng.random() < 0.85:
        ax_t.grid(True, color=grid_color, linewidth=py_rng.uniform(0.5, 1.1))

    # shaded fertile-window bands
    if py_rng.random() < 0.55:
        n_bands = py_rng.choice([1, 2])
        for _ in range(n_bands):
            start = py_rng.randint(1, N_DAYS - 4)
            width = py_rng.randint(1, 5)
            ax_t.axvspan(
                start, start + width,
                color=py_rng.choice(_BAND_COLORS),
                alpha=py_rng.uniform(0.25, 0.6), zorder=0,
            )

    # ── temperature line on primary (right or left) axis ──
    temp_color = py_rng.choice(_LINE_COLORS)
    lh_color = py_rng.choice([c for c in _LINE_COLORS if c != temp_color])
    marker = py_rng.choice(["o", "o", "s", "D", "."])
    lw = py_rng.uniform(1.2, 2.6)
    ms = py_rng.uniform(3, 7)

    # decide which physical side each axis sits on (random, like real apps)
    temp_on_right = py_rng.random() < 0.6

    def _plot_series(ax, raw, color):
        xs = days[~np.isnan(raw)]
        ys = raw[~np.isnan(raw)]
        if xs.size == 0:
            return
        style = py_rng.choice(["-", "-", "-o", "o"])
        if "o" in style and style != "o":
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)
            ax.plot(xs, ys, marker, color=color, markersize=ms, linestyle="none", zorder=4)
        elif style == "o":
            ax.plot(xs, ys, marker, color=color, markersize=ms, linestyle="none", zorder=4)
        else:
            ax.plot(xs, ys, color=color, linewidth=lw, marker=marker, markersize=ms, zorder=3)

    has_temp = present[0].sum() > 0
    has_lh = present[1].sum() > 0

    ax_l = ax_t.twinx()
    ax_l.set_facecolor("none")

    # temperature axis
    ax_temp_axis = ax_t
    ax_lh_axis = ax_l
    if has_temp:
        _plot_series(ax_temp_axis, temps_raw, temp_color)
    ax_temp_axis.set_ylim(temp_lo, temp_hi)
    if has_lh:
        _plot_series(ax_lh_axis, lh_raw, lh_color)
    ax_lh_axis.set_ylim(lh_lo, lh_hi)

    ax_t.set_xlim(0.5, N_DAYS + 0.5)

    # put temp ticks on chosen side
    if temp_on_right:
        ax_temp_axis.yaxis.tick_right()
        ax_lh_axis.yaxis.tick_left()
    else:
        ax_temp_axis.yaxis.tick_left()
        ax_lh_axis.yaxis.tick_right()

    # tick label styling
    fs = py_rng.uniform(6, 9)
    for ax in (ax_t, ax_l):
        ax.tick_params(labelsize=fs, length=py_rng.uniform(0, 3))

    # x tick labels (day numbers, possibly sparse)
    step = py_rng.choice([1, 2, 5])
    xticks = list(range(1, N_DAYS + 1, step))
    ax_t.set_xticks(xticks)
    ax_t.set_xticklabels(
        [str(day_label_start + (d - 1)) for d in xticks], fontsize=fs
    )

    if py_rng.random() < 0.3:
        ax_t.set_title(
            py_rng.choice(["Tages-Ansicht", "Daily view", "Mon cycle", "Cycle"]),
            fontsize=fs + 2, loc="left", color="#888",
        )

    for spine in ax_t.spines.values():
        spine.set_color(py_rng.choice(["#cccccc", "#dddddd", "#e0e0e0"]))

    img = _fig_to_pil(fig)

    # ── post-render image augmentation ──
    if py_rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(py_rng.uniform(0.3, 1.0)))
    # random rescale to the model canvas with a bit of jitter
    target = (IMG_W, IMG_H)
    img = img.resize(target, Image.BILINEAR)
    if py_rng.random() < 0.4:
        # jpeg-ish recompression artifacts
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=py_rng.randint(45, 90))
        b.seek(0)
        img = Image.open(b).convert("RGB")
    if py_rng.random() < 0.3:
        arr = np.asarray(img).astype(np.float32)
        arr += rng.normal(0, py_rng.uniform(2, 8), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    meta = {
        "temp_range": (temp_lo, temp_hi),
        "lh_range": (lh_lo, lh_hi),
        "temp_on_right": temp_on_right,
        "day_label_start": day_label_start,
        "has_temp": bool(has_temp),
        "has_lh": bool(has_lh),
    }
    return ChartSample(image=img, value=value, present=present, meta=meta)
