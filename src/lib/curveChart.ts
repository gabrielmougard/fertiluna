/**
 * curveChart.ts — renders the animated BBT/LH curve as inline SVG.
 *
 * This is the visual payoff of Outil 1: a clean, animated chart with the
 * follicular/luteal phases colour-coded, the detected ovulation day marked,
 * a confidence band around it, and LH peaks annotated. Drawn as SVG (no chart
 * lib) so it stays tiny and fully themeable.
 */

import { CYCLE_MAX_DAYS } from "./constants";

export interface CurveChartOptions {
  temps: (number | null)[];
  lh: (number | null)[];
  ovulationDay: number | null; // 1-indexed
  width?: number;
  height?: number;
}

const PAD = { top: 24, right: 18, bottom: 34, left: 44 };

function tempRange(temps: (number | null)[]): [number, number] {
  const vals = temps.filter((v): v is number => v != null && Number.isFinite(v));
  if (vals.length === 0) return [36.0, 37.2];
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  // pad and round to nice 0.1 grid
  lo = Math.floor((lo - 0.1) * 10) / 10;
  hi = Math.ceil((hi + 0.1) * 10) / 10;
  if (hi - lo < 0.5) hi = lo + 0.5;
  return [lo, hi];
}

export function renderCurveChart(opts: CurveChartOptions): string {
  const width = opts.width ?? 720;
  const height = opts.height ?? 320;
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  const [tLo, tHi] = tempRange(opts.temps);
  const x = (day: number) => PAD.left + (day / (CYCLE_MAX_DAYS - 1)) * plotW;
  const y = (t: number) =>
    PAD.top + plotH - ((t - tLo) / (tHi - tLo)) * plotH;

  // y grid lines every 0.2°C
  const gridLines: string[] = [];
  const yLabels: string[] = [];
  for (let t = tLo; t <= tHi + 1e-9; t += 0.2) {
    const yy = y(t);
    gridLines.push(
      `<line x1="${PAD.left}" y1="${yy.toFixed(1)}" x2="${(width - PAD.right).toFixed(1)}" y2="${yy.toFixed(1)}" class="grid" />`,
    );
    yLabels.push(
      `<text x="${PAD.left - 8}" y="${(yy + 4).toFixed(1)}" class="ytick">${t.toFixed(1)}</text>`,
    );
  }

  // x ticks every 5 days
  const xTicks: string[] = [];
  for (let d = 0; d < CYCLE_MAX_DAYS; d += 5) {
    xTicks.push(
      `<text x="${x(d).toFixed(1)}" y="${(height - 12).toFixed(1)}" class="xtick">J${d + 1}</text>`,
    );
  }

  // ovulation band + line
  let ovMarkup = "";
  if (opts.ovulationDay && opts.ovulationDay >= 1) {
    const ovIdx = opts.ovulationDay - 1;
    const cx = x(ovIdx);
    // ±1 day confidence band
    const bandLo = x(Math.max(0, ovIdx - 1));
    const bandHi = x(Math.min(CYCLE_MAX_DAYS - 1, ovIdx + 1));
    ovMarkup = `
      <rect x="${bandLo.toFixed(1)}" y="${PAD.top}" width="${(bandHi - bandLo).toFixed(1)}" height="${plotH}" class="ov-band" />
      <line x1="${cx.toFixed(1)}" y1="${PAD.top}" x2="${cx.toFixed(1)}" y2="${(PAD.top + plotH).toFixed(1)}" class="ov-line" />
      <text x="${cx.toFixed(1)}" y="${(PAD.top - 8).toFixed(1)}" class="ov-label">Ovulation ~J${opts.ovulationDay}</text>
    `;
  }

  // temperature path (follicular vs luteal coloured) + points
  const segments: string[] = [];
  const points: string[] = [];
  let prev: { px: number; py: number } | null = null;
  const ovIdx = opts.ovulationDay ? opts.ovulationDay - 1 : -1;
  for (let d = 0; d < CYCLE_MAX_DAYS; d++) {
    const v = opts.temps[d];
    if (v == null || !Number.isFinite(v)) {
      prev = null;
      continue;
    }
    const px = x(d);
    const py = y(v);
    const phase = ovIdx >= 0 && d > ovIdx ? "luteal" : "follicular";
    if (prev) {
      segments.push(
        `<line x1="${prev.px.toFixed(1)}" y1="${prev.py.toFixed(1)}" x2="${px.toFixed(1)}" y2="${py.toFixed(1)}" class="temp-line ${phase}" />`,
      );
    }
    points.push(
      `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3.4" class="temp-dot ${phase}" />`,
    );
    prev = { px, py };
  }

  // LH markers (drawn near the bottom of the plot as little bars)
  const lhMarkup: string[] = [];
  const lhVals = opts.lh.filter(
    (v): v is number => v != null && Number.isFinite(v),
  );
  if (lhVals.length > 0) {
    const lhMax = Math.max(1, ...lhVals);
    for (let d = 0; d < CYCLE_MAX_DAYS; d++) {
      const v = opts.lh[d];
      if (v == null || !Number.isFinite(v)) continue;
      const barH = (v / lhMax) * (plotH * 0.32);
      const bx = x(d);
      const by = PAD.top + plotH - barH;
      const strong = v >= Math.max(1.3, lhMax * 0.8);
      lhMarkup.push(
        `<rect x="${(bx - 2).toFixed(1)}" y="${by.toFixed(1)}" width="4" height="${barH.toFixed(1)}" class="lh-bar ${strong ? "peak" : ""}" />`,
      );
    }
  }

  const totalSegLen = segments.length;

  return `
<svg viewBox="0 0 ${width} ${height}" class="curve-svg" role="img"
     aria-label="Courbe de température basale annotée">
  <style>
    .grid { stroke: #efe7f3; stroke-width: 1; }
    .ytick { fill: #9b8fab; font-size: 10px; text-anchor: end; }
    .xtick { fill: #9b8fab; font-size: 10px; text-anchor: middle; }
    .ov-band { fill: rgba(126,75,158,0.10); }
    .ov-line { stroke: var(--purple); stroke-width: 1.5; stroke-dasharray: 4 4; }
    .ov-label { fill: var(--purple); font-size: 11px; font-weight: 700; text-anchor: middle; }
    .temp-line { stroke-width: 2.4; fill: none; stroke-linecap: round;
                 stroke-dasharray: 220; stroke-dashoffset: 220;
                 animation: draw 0.9s ease forwards; }
    .temp-line.follicular { stroke: var(--rose); }
    .temp-line.luteal { stroke: var(--purple); }
    .temp-dot { opacity: 0; animation: pop 0.3s ease forwards; }
    .temp-dot.follicular { fill: var(--rose); }
    .temp-dot.luteal { fill: var(--purple); }
    .lh-bar { fill: #d9c7e8; }
    .lh-bar.peak { fill: #f5a623; }
    @keyframes draw { to { stroke-dashoffset: 0; } }
    @keyframes pop { to { opacity: 1; } }
    @media (prefers-reduced-motion: reduce) {
      .temp-line { animation: none; stroke-dashoffset: 0; }
      .temp-dot { animation: none; opacity: 1; }
    }
  </style>
  ${gridLines.join("\n")}
  ${ovMarkup}
  ${lhMarkup.join("\n")}
  ${segments
    .map(
      (s, i) =>
        s.replace(
          'class="temp-line',
          `style="animation-delay:${(i * 0.9) / Math.max(1, totalSegLen)}s" class="temp-line`,
        ),
    )
    .join("\n")}
  ${points
    .map(
      (p, i) =>
        p.replace(
          'class="temp-dot',
          `style="animation-delay:${0.2 + (i * 0.6) / Math.max(1, points.length)}s" class="temp-dot`,
        ),
    )
    .join("\n")}
  ${yLabels.join("\n")}
  ${xTicks.join("\n")}
</svg>`;
}
