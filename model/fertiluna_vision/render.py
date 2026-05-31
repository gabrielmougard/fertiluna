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
    bbt_scale: int = 0   # index into constants.BBT_SCALES (0=celsius, 1=fahrenheit)


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
    # Generic charts use arbitrary axis ranges that don't match the fixed
    # C/F conventions, so their scale label is "unknown" (-1) and the scale
    # loss is masked for them. Only premom-style charts supervise the scale head.
    return ChartSample(image=img, value=value, present=present, meta=meta,
                       bbt_scale=-1)


# Premom-style renderer
# Reproduces the visual conventions of Premom app screenshots (see model docs):
#   * three y-axes, left->right: Ratio | Level | BBT
#   * blue line  = BBT (temperature)  -> always the RIGHTMOST axis
#   * orange line = "Ratio" (LH test Test/Control ratio) -> LEFTMOST axis
#                   -> this is the series the model predicts as `lh`
#   * purple line = "Level" (LH hormone level) -> MIDDLE axis, distractor only
#   * vertical fertile-window (violet) + period (pink) bands
#   * bottom rows: month/day, CD or ZT, DPO, and icon rows (Sex/CM/Symptoms/...)
#   * BBT axis in either Celsius (35.6-37.4) or Fahrenheit (95-99.5)
# Only BBT (-> value[0]) and the orange Ratio LH (-> value[1]) are labeled; the
# purple Level line is drawn purely as a visual distractor.
_PREMOM_BBT = "#95aeff"       # blue (BBT line)
_PREMOM_RATIO = "#ff9e8d"     # orange (LH ratio -> predicted lh)
_PREMOM_LEVEL = "#9e6fe3"     # purple (Level -> distractor)
_PREMOM_FERTILE = "#c9a8ec"   # violet band
_PREMOM_FERTILE_PEAK = "#9b6fd6"
_PREMOM_PERIOD = "#f3c6cf"    # pink band
_PREMOM_BG = "#ffffff"
_PREMOM_GRID = "#efeff4"
_PREMOM_TITLES = ["Tages-Ansicht", "Cycle View", "Day View", "Daily view"]
# English 3-letter month abbreviations (as in the reference screenshots).
_MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                "Sep", "Oct", "Nov", "Dec"]


def render_premom_chart(rng: np.random.Generator,
                        dpi_override: int | None = None) -> ChartSample:
    """Render a synthetic chart styled to look like a Premom app screenshot.

    dpi_override: if set, force this DPI (for high-res visual inspection). The
    image is still resized to the model canvas, so the returned labels are
    unchanged — only the pre-resize figure detail differs.
    """
    py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))

    temps_raw, lh_raw = _series_curves(rng)

    # ── partial-width windowing (matches real screenshots) ──────────────────
    # Real charts show a VARIABLE number of data days, positioned in varied
    # sub-regions of the plot: data may leave empty "future" space on the RIGHT
    # (a cycle in progress, real-screen-1), empty space on the LEFT (the visible
    # window starts mid-cycle), or be densely packed. The v1 renderer always
    # drew exactly N_DAYS of data left-to-right filling the whole box, so the
    # model learned a fixed day→x mapping and shifted on real charts.
    #
    # Fix: place `n_data` real data days into columns [left_pad, left_pad+n_data)
    # of an axis that spans `x_capacity` day-slots, with NaN elsewhere:
    #   - left_pad > 0  → empty space on the left,
    #   - x_capacity > left_pad+n_data → empty "future" space on the right.
    # The per-column label contract is unchanged: present columns are exactly the
    # occupied ones, which the browser collapses left-to-right into table days.
    n_data = int(py_rng.randint(14, N_DAYS))  # randint inclusive; 14..N_DAYS
    # left offset: usually 0, but a meaningful fraction start mid-axis.
    max_left = max(0, N_DAYS - n_data)
    if py_rng.random() < 0.45 and max_left > 0:
        left_pad = int(py_rng.randint(1, max_left))
    else:
        left_pad = 0
    # right empty "future" space beyond the data.
    used = left_pad + n_data
    if py_rng.random() < 0.5 and used < N_DAYS:
        x_capacity = int(py_rng.randint(used, N_DAYS))  # randint is inclusive
    else:
        x_capacity = used
    x_capacity = max(used, min(N_DAYS, x_capacity))

    # shift the curves into [left_pad, left_pad+n_data), NaN elsewhere
    def _window(arr):
        out = np.full(N_DAYS, np.nan, dtype=np.float64)
        src = arr[:n_data]
        out[left_pad:left_pad + len(src)] = src
        return out

    temps_raw = _window(temps_raw)
    lh_raw = _window(lh_raw)

    # The orange "Ratio" line in Premom is a SHARP LH-test ratio: a low baseline
    # (~0.2-0.4) with a tall narrow spike (1.3-1.9) at the surge, then a fast
    # drop. Reshape the generic LH into that spiky profile so it matches real
    # screenshots. This reshaped curve is what we LABEL as `lh`.
    ratio_lo, ratio_hi = 0.1, 1.9
    valid_lh = ~np.isnan(lh_raw)
    ratio_raw = np.full(N_DAYS, np.nan, dtype=np.float64)
    if valid_lh.any():
        pk = int(np.nanargmax(lh_raw))
        base = py_rng.uniform(0.18, 0.40)
        peak = py_rng.uniform(1.25, 1.85)
        falloff_up = py_rng.uniform(1.6, 2.6)    # steep rise
        falloff_dn = py_rng.uniform(1.1, 1.8)    # slightly slower drop
        for d in range(N_DAYS):
            if not valid_lh[d]:
                continue
            dist = d - pk
            if dist <= 0:
                h = peak * np.exp(-(abs(dist) ** 1.4) / falloff_up)
            else:
                h = peak * np.exp(-(abs(dist) ** 1.4) / falloff_dn)
            ratio_raw[d] = base + (peak - base) * (h / peak) + py_rng.uniform(-0.04, 0.06)
        ratio_raw = np.clip(ratio_raw, ratio_lo, ratio_hi)

    # The purple "Level" line is a SMOOTHER, BROADER hump (hormone level), offset
    # from the spike — purely a visual distractor (NOT labeled).
    level_lo, level_hi = 5.0, 95.0
    level_raw = np.full(N_DAYS, np.nan, dtype=np.float64)
    if valid_lh.any():
        pk = int(np.nanargmax(lh_raw))
        lbase = py_rng.uniform(8, 22)
        lpeak = py_rng.uniform(45, 90)
        lspread = py_rng.uniform(3.0, 6.0)
        for d in range(N_DAYS):
            if not valid_lh[d]:
                continue
            level_raw[d] = lbase + (lpeak - lbase) * np.exp(
                -((d - pk) ** 2) / (2 * lspread ** 2)
            ) + py_rng.uniform(-3, 3)
        level_raw = np.clip(level_raw, level_lo, level_hi)
    valid = ~np.isnan(level_raw)

    use_fahrenheit = py_rng.random() < 0.5
    # BBT axis range (the model normalizes within this range either way).
    if use_fahrenheit:
        # convert celsius working values to F for display only
        def c2f(c):
            return c * 9.0 / 5.0 + 32.0
        disp_temps = np.where(np.isnan(temps_raw), np.nan, c2f(temps_raw))
        bbt_lo, bbt_hi = 95.0, 99.5
        n_grid = 18   # 95.0..99.5 every 0.25 -> 18 intervals
    else:
        disp_temps = temps_raw
        bbt_lo, bbt_hi = 35.6, 37.4
        n_grid = 18   # 35.6..37.4 every 0.1 -> 18 intervals

    # Premom snaps every plotted point onto a horizontal gridline — the y-values
    # are quantized, never sitting between lines. The horizontal gridlines belong
    # to the BBT axis (n_grid evenly-spaced lines), so snapping in shared HEIGHT
    # fraction space quantizes all three series to those same lines.
    def _snap(raw, lo, hi):
        out = raw.copy()
        m = ~np.isnan(out)
        if m.any():
            frac = (out[m] - lo) / max(1e-9, hi - lo)
            frac = np.round(frac * n_grid) / n_grid
            out[m] = lo + np.clip(frac, 0.0, 1.0) * (hi - lo)
        return out

    disp_temps = _snap(disp_temps, bbt_lo, bbt_hi)
    ratio_raw = _snap(ratio_raw, ratio_lo, ratio_hi)
    level_raw = _snap(level_raw, level_lo, level_hi)
    valid = ~np.isnan(level_raw)

    # ── labels (model contract): value[0]=temp uses BBT axis; value[1]=lh uses
    #    the orange Ratio line + its Ratio axis. Labels reflect the SNAPPED values
    #    so the target matches exactly what is drawn. ──
    value = np.zeros((N_SERIES, N_DAYS), dtype=np.float32)
    present = np.zeros((N_SERIES, N_DAYS), dtype=np.float32)
    value[0], present[0] = _normalize_series(disp_temps, bbt_lo, bbt_hi)
    value[1], present[1] = _normalize_series(ratio_raw, ratio_lo, ratio_hi)

    has_temp = present[0].sum() > 0
    has_lh = present[1].sum() > 0
    show_level = py_rng.random() < 0.85

    # ── figure ──
    w_in = py_rng.uniform(7.0, 9.0)
    h_in = py_rng.uniform(3.4, 4.4)
    dpi = dpi_override if dpi_override is not None else py_rng.choice([80, 90, 100])
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    fig.patch.set_facecolor(_PREMOM_BG)

    # Leave room at the bottom for the date / CD / DPO / icon rows, and a wider
    # left margin so the two stacked left axes (Ratio | Level) sit OUTSIDE the
    # plot and the data never spills past the y-axis. The plot top is lowered a
    # bit so the legend headers (Ratio/Level/BBT + underlines) don't overlap it.
    plot_bottom = py_rng.uniform(0.26, 0.34)
    ax_ratio = fig.add_axes([0.10, plot_bottom, 0.80, 0.56])
    ax_ratio.set_facecolor(_PREMOM_BG)
    ax_level = ax_ratio.twinx()
    ax_bbt = ax_ratio.twinx()
    ax_level.set_facecolor("none")
    ax_bbt.set_facecolor("none")

    days = np.arange(1, N_DAYS + 1)
    # small margin on each side so day 1 / day N sit inside the axes, not on the
    # left spine (the references always have a gap before the first point).
    ax_ratio.set_xlim(0.0, x_capacity + 1.0)
    ax_ratio.set_ylim(ratio_lo, ratio_hi)
    ax_level.set_ylim(level_lo, level_hi)
    ax_bbt.set_ylim(bbt_lo, bbt_hi)

    # grid: vertical lines BETWEEN days (at half-integer x), so each day's data
    # point sits in the MIDDLE of two vertical lines (matches Premom). Horizontal
    # gridlines are drawn by the BBT axis below (Premom's grid follows the BBT
    # scale, with an unlabeled line halfway between each labeled tick).
    for d in range(0, x_capacity + 2):
        ax_ratio.axvline(d - 0.5, color=_PREMOM_GRID, linewidth=0.7, zorder=0)

    # ── bands ──
    # fertile window (violet) centered near ovulation; ovulation day darker.
    # Track which days fall in each colored band so the date numbers below get a
    # matching rounded pill background (as in the app).
    ov = int(np.clip(np.nanargmax(ratio_raw) if has_lh else N_DAYS // 2, 3, N_DAYS - 3))
    fertile_days: set[int] = set()
    peak_days: set[int] = set()
    period_days: set[int] = set()
    if py_rng.random() < 0.9:
        f0 = max(1, ov - py_rng.randint(3, 5))
        f1 = min(N_DAYS, ov + py_rng.randint(1, 3))
        ax_ratio.axvspan(f0 - 0.5, f1 + 0.5, color=_PREMOM_FERTILE, alpha=0.30, zorder=0)
        ax_ratio.axvspan(ov - 1.5, ov + 0.5, color=_PREMOM_FERTILE_PEAK, alpha=0.35, zorder=0)
        fertile_days = set(range(f0, f1 + 1))
        peak_days = {ov - 1, ov}
    # period band (pink) — early and/or late in the window. The late band sits
    # at the right edge of the VISIBLE axis (x_capacity), matching real charts
    # where a pink "predicted period / pregnancy" band fills the empty future.
    if py_rng.random() < 0.7:
        p0 = py_rng.randint(max(1, x_capacity - 8), max(2, x_capacity - 3))
        ax_ratio.axvspan(p0 - 0.5, x_capacity + 0.5, color=_PREMOM_PERIOD, alpha=0.45, zorder=0)
        period_days |= set(range(p0, x_capacity + 1))
    if py_rng.random() < 0.4:
        pe = py_rng.randint(2, 4)
        ax_ratio.axvspan(0.5, pe + 0.5, color=_PREMOM_PERIOD, alpha=0.45, zorder=0)
        period_days |= set(range(1, pe + 1))
    # peak days take precedence visually over the lighter fertile fill
    fertile_days -= peak_days

    # ── lines ──
    ms = py_rng.uniform(3, 4.5)
    lw = py_rng.uniform(1.1, 1.7)
    fs = py_rng.uniform(7, 9)
    fs_lh = fs

    # Some real charts (esp. the BBT line) are pale / thin / low-contrast — the
    # v1 model missed those entirely (real-screen-2). Randomly render a series
    # with reduced alpha + thinner stroke so the model learns faint lines too.
    line_alpha = py_rng.uniform(0.45, 0.7) if py_rng.random() < 0.25 else 1.0

    def _plot_open(ax, raw, color, z, alpha=None):
        xs = days[~np.isnan(raw)]
        ys = raw[~np.isnan(raw)]
        if xs.size == 0:
            return
        a = line_alpha if alpha is None else alpha
        ax.plot(xs, ys, color=color, linewidth=lw, zorder=z, alpha=a)
        ax.plot(xs, ys, "o", color=color, markersize=ms, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.4, zorder=z + 1, alpha=a)

    # orange Ratio (predicted lh) on the leftmost (ratio) axis
    if has_lh:
        _plot_open(ax_ratio, ratio_raw, _PREMOM_RATIO, 4)
        # the LH-peak: prominent filled orange circle with a white '+'
        pk = int(np.nanargmax(ratio_raw))
        ax_ratio.plot([pk + 1], [ratio_raw[pk]], "o", color=_PREMOM_RATIO,
                      markersize=ms + 8, markeredgecolor="white",
                      markeredgewidth=1.2, zorder=8)
        ax_ratio.plot([pk + 1], [ratio_raw[pk]], marker="P", color="white",
                      markersize=ms + 1, linestyle="none", zorder=9)
        # occasional "LH Peak" label badge
        if py_rng.random() < 0.4:
            ax_ratio.annotate(
                "LH Peak", xy=(pk + 1, ratio_raw[pk]),
                xytext=(pk + 1 + 1.2, min(ratio_hi - 0.1, ratio_raw[pk] + 0.12)),
                fontsize=fs_lh, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=_PREMOM_LEVEL, ec="none"),
                zorder=10,
            )
    # purple Level (distractor) on middle axis
    if show_level and valid.any():
        _plot_open(ax_level, level_raw, _PREMOM_LEVEL, 3)
    # blue BBT on the rightmost axis (+ horizontal coverline + optional 'B' marker)
    if has_temp:
        _plot_open(ax_bbt, disp_temps, _PREMOM_BBT, 5)
        if py_rng.random() < 0.7:
            cover = float(np.nanmedian(disp_temps))
            ax_bbt.axhline(cover, color=_PREMOM_BBT, linewidth=1.0, alpha=0.7, zorder=2)
        # "B" coverline-start marker on the first post-ovulation temp day
        if py_rng.random() < 0.55:
            post = [d for d in range(ov + 1, N_DAYS) if not np.isnan(disp_temps[d])]
            if post:
                bd = post[0]
                ax_bbt.plot([bd + 1], [disp_temps[bd]], "o", color=_PREMOM_BBT,
                            markersize=ms + 5, zorder=6)
                ax_bbt.text(bd + 1, disp_temps[bd], "B", fontsize=fs_lh - 1,
                            color="white", ha="center", va="center",
                            weight="bold", zorder=7)

    # ── axis ticks / headers ──
    # Two stacked y-axes OUTSIDE the plot on the left: Ratio (outermost) then
    # Level (just inside it), matching the app's "Ratio  Level" columns.
    ax_ratio.yaxis.tick_left()
    ax_ratio.spines["left"].set_position(("axes", -0.045))
    ax_ratio.set_yticks(np.arange(0.1, 2.0, 0.2))
    ax_ratio.set_yticklabels([f"{v:.1f}" for v in np.arange(0.1, 2.0, 0.2)])
    ax_ratio.tick_params(labelsize=fs, length=0, colors="#888", pad=1)
    # level: a second left axis just to the RIGHT of the ratio column
    ax_level.yaxis.set_label_position("left")
    ax_level.yaxis.tick_left()
    ax_level.spines["left"].set_position(("axes", 0.0))
    ax_level.spines["left"].set_color("none")
    ax_level.set_yticks(np.arange(5, 96, 10))
    ax_level.tick_params(labelsize=fs, length=0, colors="#888", pad=1)
    # bbt on the right, with Premom-style tick labels + grid.
    #   * labeled ticks: every 0.2 C (".2",".4",..,"36","37") or 0.5 F ("96",".5"..)
    #   * an UNLABELED gridline sits halfway between each labeled tick
    #   * extreme top/bottom labels carry >= / <= prefixes
    ax_bbt.yaxis.tick_right()
    if use_fahrenheit:
        major = np.round(np.arange(95.0, 99.51, 0.5), 2)
        minor = np.round(np.arange(95.25, 99.5, 0.5), 2)  # halfway lines
        def _lab(t):
            frac = round(t - int(t), 2)
            return str(int(round(t))) if frac == 0.0 else ".5"
    else:
        major = np.round(np.arange(35.6, 37.51, 0.2), 2)
        minor = np.round(np.arange(35.7, 37.5, 0.2), 2)   # halfway lines
        def _lab(t):
            tr = round(t, 1)
            frac = round(tr - int(tr), 1)
            return str(int(round(tr))) if frac == 0.0 else "." + str(int(round(frac * 10)))

    labels = []
    for i, t in enumerate(major):
        s = _lab(t)
        if i == 0:
            s = "\u2264" + s          # bottom: <=
        elif i == len(major) - 1:
            s = "\u2265" + s          # top: >=
        labels.append(s)
    ax_bbt.set_yticks(major)
    ax_bbt.set_yticklabels(labels)
    ax_bbt.set_yticks(minor, minor=True)
    # horizontal grid: solid-ish on labeled ticks, fainter on the halfway lines
    ax_bbt.grid(True, which="major", axis="y", color=_PREMOM_GRID, linewidth=0.9, zorder=0)
    ax_bbt.grid(True, which="minor", axis="y", color=_PREMOM_GRID, linewidth=0.7,
                alpha=0.7, zorder=0)
    ax_bbt.tick_params(which="both", labelsize=fs, length=0, colors="#888")

    for ax in (ax_ratio, ax_level, ax_bbt):
        for s in ax.spines.values():
            s.set_color("none")

    # axis legend headers: labels in BLACK with a short colored legend line just
    # below each (orange=Ratio, purple=Level, blue=BBT), like the real app.
    import matplotlib.lines as _mlines

    def _legend(x, text, color):
        fig.text(x, 0.965, text, fontsize=fs, color="#222", weight="bold", ha="left")
        # short colored underline beneath the label
        ln = _mlines.Line2D([x, x + 0.035], [0.945, 0.945], color=color,
                            linewidth=2.2, transform=fig.transFigure,
                            solid_capstyle="round")
        fig.add_artist(ln)

    _legend(0.02, "Ratio", _PREMOM_RATIO)
    _legend(0.065, "Level", _PREMOM_LEVEL)
    _legend(0.90, "BBT", _PREMOM_BBT)
    # faint title
    ax_ratio.set_title(py_rng.choice(_PREMOM_TITLES), fontsize=fs + 4,
                       loc="left", color="#cccccc", pad=2)

    # ── x ticks (day numbers, hidden — we draw our own bottom rows) ──
    ax_ratio.set_xticks([])

    # ── bottom rows: calendar / CD / DPO, centered in cells under each point ──
    # Calendar: real day-of-month numbers that roll over across months; on the
    #   1st of a new month the 3-letter month abbrev replaces that day's number.
    # CD (cycle day): starts mid-cycle and rolls over to 1 when a new cycle
    #   begins (a period start within the window), like the references.
    # DPO: blank until ovulation, then 1,2,3,... from the day after the peak.
    y = plot_bottom - 0.02
    row_h = (plot_bottom - 0.02) / 6.5

    def _xfrac(d):  # d is 1-based day index -> figure x-fraction of that cell
        return 0.10 + 0.80 * d / (x_capacity + 1)

    # calendar start
    start_month = py_rng.randint(0, 11)
    start_day = py_rng.randint(1, 26)
    _MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cur_month = start_month
    cur_day = start_day
    # cycle day start + a possible rollover point (new cycle) inside the window
    cd = py_rng.randint(18, 30)
    cd_rollover = py_rng.randint(N_DAYS // 2, N_DAYS) if py_rng.random() < 0.55 else None

    # how many days to skip between drawn labels (1 = every day; real app shows
    # every day)
    # First pass: collect each day's calendar + CD label and its band color.
    import matplotlib.patches as _mpatches

    cal_labels = {}
    cd_labels = {}
    for d in range(1, N_DAYS + 1):
        # calendar
        if cur_day == 1 and d != 1:
            cur_month = (cur_month + 1) % 12
            cal_labels[d] = _MONTHS_ABBR[cur_month]
        else:
            cal_labels[d] = str(cur_day)
        cur_day += 1
        if cur_day > _MDAYS[cur_month]:
            cur_day = 1
        if cd_rollover is not None and d == cd_rollover:
            cd = 1
        cd_labels[d] = str(cd)
        cd += 1

    def _day_color(d):
        # Fertile window (incl. ovulation/peak days) is ONE continuous light-violet
        # capsule; period days are a pink capsule. Day numbers inside any capsule
        # are white.
        if d in fertile_days or d in peak_days:
            return _PREMOM_FERTILE_PEAK, "white"
        if d in period_days:
            return "#e8849a", "white"
        return None, None

    # continuous rounded CAPSULES spanning contiguous runs of same-colored days
    # (drawn once per run, behind the numbers), like the references. The capsule
    # is taller than the text and its rounded ends extend past the end numbers,
    # so there is clear padding all around the day numbers inside it.
    cell_w = 0.80 / (x_capacity + 1)
    cap_h = row_h * 0.78          # taller than the glyphs -> vertical padding
    for row_y, default_col in ((y, "#555"), (y - row_h, "#9b6fd6")):
        d = 1
        while d <= x_capacity:
            fc, _ = _day_color(d)
            if fc is None:
                d += 1
                continue
            run_end = d
            while run_end + 1 <= x_capacity and _day_color(run_end + 1)[0] == fc:
                run_end += 1
            # extend the capsule a bit beyond the first/last number centers so the
            # end caps clear the digits (horizontal padding).
            x0 = _xfrac(d) - cell_w * 0.34
            x1 = _xfrac(run_end) + cell_w * 0.34
            cap = _mpatches.FancyBboxPatch(
                (x0, row_y - cap_h * 0.5), x1 - x0, cap_h,
                boxstyle="round,pad=0,rounding_size=" + str(cap_h * 0.5),
                transform=fig.transFigure, fc=fc, ec="none",
                mutation_aspect=1.0, zorder=4,
            )
            fig.add_artist(cap)
            d = run_end + 1

    # numbers on top of the capsules
    for d in range(1, x_capacity + 1):
        xf = _xfrac(d)
        fc, txt_col = _day_color(d)
        fig.text(xf, y, cal_labels[d], fontsize=fs - 1, va="center",
                 color=txt_col if fc else "#555", ha="center", zorder=5)
        fig.text(xf, y - row_h, cd_labels[d], fontsize=fs - 1, va="center",
                 color=txt_col if fc else "#9b6fd6", ha="center", zorder=5)
        # DPO only after ovulation
        if d - 1 > ov:
            fig.text(xf, y - row_h * 2, str(d - 1 - ov), fontsize=fs - 1,
                     color="#9b6fd6", ha="center", va="center")

    # row labels (left margin) — CD/DPO/Sex/CM in black, right-aligned just left
    # of the plot so they line up under the "Level" y-axis column.
    label_x = 0.095
    # the calendar row's left prefix is the STARTING month (the days before the
    # first month-rollover belong to start_month, which otherwise has no header).
    fig.text(label_x, y, _MONTHS_ABBR[start_month], fontsize=fs - 1, color="#222",
             ha="right", va="center", weight="bold")
    for i, (lbl, col) in enumerate([("CD", "#222"), ("DPO", "#222"),
                                    ("Sex", "#222"), ("CM", "#222")]):
        fig.text(label_x, y - row_h * (i + 1.0), lbl, fontsize=fs - 1, color=col,
                 ha="right", va="center")
    # a few sex hearts at random fertile-ish days
    for _ in range(py_rng.randint(0, 4)):
        d = py_rng.randint(max(1, ov - 5), min(N_DAYS, ov + 2))
        fig.text(_xfrac(d), y - row_h * 3, "\u2665", fontsize=fs, color="#e8825f",
                 ha="center", va="center")

    # ── floating UI chrome (buttons / labels / close X) the model must learn to
    #    ignore — present in most real Premom screenshots, on the right side. ──
    if py_rng.random() < 0.7:
        chrome = py_rng.sample(
            [("Cycle View", 0.82, 0.90), ("Setting", 0.84, 0.78),
             ("Ask an Expert", 0.80, 0.16), ("\u2715", 0.95, 0.20),
             ("Zyklus-Galerie", 0.80, 0.90)],
            k=py_rng.randint(1, 3),
        )
        for txt, cx, cy in chrome:
            fig.text(cx, cy, txt, fontsize=fs + 1, color="#777", ha="center",
                     va="center",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f8",
                               ec="#e2e2ea", lw=0.8), zorder=20)
        # an occasional zoom +/- control pair
        if py_rng.random() < 0.4:
            for cy in (0.50, 0.40):
                fig.text(0.95, cy, py_rng.choice(["+", "\u2212"]), fontsize=fs + 2,
                         color="#999", ha="center", va="center",
                         bbox=dict(boxstyle="circle,pad=0.3", fc="#f6f6fa",
                                   ec="#e2e2ea", lw=0.8), zorder=20)

    img = _fig_to_pil(fig)

    # ── light post-render augmentation (screenshots aren't pristine) ──
    if py_rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(py_rng.uniform(0.3, 0.8)))
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    if py_rng.random() < 0.5:
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=py_rng.randint(55, 92))
        b.seek(0)
        img = Image.open(b).convert("RGB")

    meta = {
        "style": "premom",
        "bbt_range": (bbt_lo, bbt_hi),
        "ratio_range": (ratio_lo, ratio_hi),
        "fahrenheit": use_fahrenheit,
        "has_temp": bool(has_temp),
        "has_lh": bool(has_lh),
        "show_level": bool(show_level),
    }
    return ChartSample(image=img, value=value, present=present, meta=meta,
                       bbt_scale=1 if use_fahrenheit else 0)
