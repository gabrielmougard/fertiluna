import { describe, it, expect } from "vitest";
import {
  type RGBA,
  type Calibration,
  extractCurvePixels,
  resampleToDays,
  mapAxis,
  unmapAxis,
  colorDistance,
  suggestCurveColor,
  rounderFor,
} from "./curveDigitizer";

// ── helpers: build a synthetic chart RGBA buffer ──────────────────
const W = 400;
const H = 300;
const CURVE: RGBA = { r: 214, g: 68, b: 127, a: 255 }; // FertiLuna rose
const GRID: RGBA = { r: 230, g: 230, b: 235, a: 255 };
const BG: RGBA = { r: 255, g: 255, b: 255, a: 255 };

// Plot region maps: x in [40, 360] → day [1, 29]; y in [40, 260] → temp [37.0, 36.2]
// (note: y grows downward, higher temp is higher up → smaller y)
const cal: Calibration = {
  dayAxis: { pixel1: 40, value1: 1, pixel2: 360, value2: 29 },
  valueAxis: { pixel1: 40, value1: 37.0, pixel2: 260, value2: 36.2 },
};

function makeChart(dayTemps: (number | null)[]): {
  data: Uint8ClampedArray;
  width: number;
  height: number;
} {
  const data = new Uint8ClampedArray(W * H * 4);
  // fill background
  for (let i = 0; i < W * H; i++) {
    data[i * 4] = BG.r;
    data[i * 4 + 1] = BG.g;
    data[i * 4 + 2] = BG.b;
    data[i * 4 + 3] = 255;
  }
  const set = (x: number, y: number, c: RGBA) => {
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    const i = (Math.round(y) * W + Math.round(x)) * 4;
    data[i] = c.r;
    data[i + 1] = c.g;
    data[i + 2] = c.b;
    data[i + 3] = 255;
  };
  // grid lines every 5 days + every 0.2 temp
  for (let d = 1; d <= 29; d += 5) {
    const px = unmapAxis(cal.dayAxis, d);
    for (let y = 40; y <= 260; y++) set(px, y, GRID);
  }
  for (let t = 36.2; t <= 37.0001; t += 0.2) {
    const py = unmapAxis(cal.valueAxis, t);
    for (let x = 40; x <= 360; x++) set(x, py, GRID);
  }
  // draw the curve as 3px-thick dots at each day
  for (let d = 1; d <= dayTemps.length; d++) {
    const v = dayTemps[d - 1];
    if (v == null) continue;
    const px = unmapAxis(cal.dayAxis, d);
    const py = unmapAxis(cal.valueAxis, v);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++) set(px + dx, py + dy, CURVE);
  }
  return { data, width: W, height: H };
}

describe("curve digitizer math", () => {
  it("axis mapping round-trips", () => {
    expect(mapAxis(cal.dayAxis, unmapAxis(cal.dayAxis, 14))).toBeCloseTo(14, 5);
    expect(mapAxis(cal.valueAxis, unmapAxis(cal.valueAxis, 36.6))).toBeCloseTo(
      36.6,
      5,
    );
  });

  it("color distance is symmetric and zero on match", () => {
    expect(colorDistance(CURVE, CURVE)).toBe(0);
    expect(colorDistance(CURVE, GRID)).toBeGreaterThan(50);
  });
});

describe("end-to-end synthetic chart extraction", () => {
  // a realistic ovulatory cycle
  const truth = [
    36.4, 36.42, 36.38, 36.41, 36.37, 36.4, 36.36, 36.39, 36.35, 36.34, 36.3,
    36.28, 36.55, 36.72, 36.78, 36.8, 36.79, 36.82, 36.81, 36.83, 36.8, 36.79,
    36.81, 36.78, 36.77, 36.62,
  ];

  it("recovers per-day temperatures within 0.05 °C", () => {
    const chart = makeChart(truth);
    const points = extractCurvePixels({
      data: chart.data,
      width: chart.width,
      height: chart.height,
      target: CURVE,
      tolerance: 40,
      region: { x0: 35, y0: 35, x1: 365, y1: 265 },
    });
    expect(points.length).toBeGreaterThan(20);

    const days = resampleToDays(points, cal, 29);
    let compared = 0;
    for (let d = 1; d <= truth.length; d++) {
      const got = days[d - 1].value;
      expect(got).not.toBeNull();
      if (got != null) {
        expect(
          Math.abs(got - truth[d - 1]),
          `day ${d}: got ${got}, truth ${truth[d - 1]}`,
        ).toBeLessThanOrEqual(0.05);
        compared++;
      }
    }
    expect(compared).toBe(truth.length);
  });

  it("auto-suggests the curve colour (saturated rose, not grey)", () => {
    const chart = makeChart(truth);
    const suggested = suggestCurveColor(chart.data, chart.width, {
      x0: 35,
      y0: 35,
      x1: 365,
      y1: 265,
    });
    // should be much closer to the rose curve than to the grey grid
    expect(colorDistance(suggested, CURVE)).toBeLessThan(
      colorDistance(suggested, GRID),
    );
  });

  it("marks days with no curve pixels as missing (null)", () => {
    const withGaps = [...truth];
    withGaps[5] = null as unknown as number; // day 6 missing
    withGaps[6] = null as unknown as number; // day 7 missing
    const chart = makeChart(withGaps);
    const points = extractCurvePixels({
      data: chart.data,
      width: chart.width,
      height: chart.height,
      target: CURVE,
      tolerance: 40,
      region: { x0: 35, y0: 35, x1: 365, y1: 265 },
    });
    const days = resampleToDays(points, cal, 29);
    expect(days[5].value).toBeNull();
    expect(days[6].value).toBeNull();
    expect(days[0].value).not.toBeNull();
  });
});

// ── dual-axis: two lines, two value scales, shared day axis ──────────
describe("dual-axis per-series extraction", () => {
  const BLUE: RGBA = { r: 86, g: 120, b: 214, a: 255 }; // BBT
  const ORANGE: RGBA = { r: 230, g: 110, b: 70, a: 255 }; // LH

  // BBT on the RIGHT axis: y[40..260] → [37.0..36.2]
  const bbtCal: Calibration = {
    dayAxis: { pixel1: 40, value1: 1, pixel2: 360, value2: 29 },
    valueAxis: { pixel1: 40, value1: 37.0, pixel2: 260, value2: 36.2 },
  };
  // LH on the LEFT axis with a DIFFERENT scale: y[60..280] → [3.0..0.0]
  const lhCal: Calibration = {
    dayAxis: { pixel1: 40, value1: 1, pixel2: 360, value2: 29 },
    valueAxis: { pixel1: 60, value1: 3.0, pixel2: 280, value2: 0.0 },
  };

  const bbt = [
    36.4, 36.4, 36.38, 36.4, 36.36, 36.35, 36.34, 36.3, 36.55, 36.75, 36.8,
    36.82, 36.81, 36.8, 36.79,
  ];
  const lh = [
    0.3, 0.4, 0.3, 0.5, 0.6, 0.8, 1.2, 2.6, 1.4, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2,
  ];

  function makeDualChart() {
    const data = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      data[i * 4] = BG.r;
      data[i * 4 + 1] = BG.g;
      data[i * 4 + 2] = BG.b;
      data[i * 4 + 3] = 255;
    }
    const set = (x: number, y: number, c: RGBA) => {
      if (x < 0 || x >= W || y < 0 || y >= H) return;
      const i = (Math.round(y) * W + Math.round(x)) * 4;
      data[i] = c.r;
      data[i + 1] = c.g;
      data[i + 2] = c.b;
      data[i + 3] = 255;
    };
    const draw = (vals: number[], c: RGBA, cal: Calibration) => {
      for (let d = 1; d <= vals.length; d++) {
        const px = unmapAxis(cal.dayAxis, d);
        const py = unmapAxis(cal.valueAxis, vals[d - 1]);
        for (let dx = -1; dx <= 1; dx++)
          for (let dy = -1; dy <= 1; dy++) set(px + dx, py + dy, c);
      }
    };
    draw(bbt, BLUE, bbtCal);
    draw(lh, ORANGE, lhCal);
    return { data, width: W, height: H };
  }

  it("recovers BBT (blue/right axis) independently of LH", () => {
    const chart = makeDualChart();
    const pts = extractCurvePixels({
      data: chart.data,
      width: W,
      height: H,
      target: BLUE,
      tolerance: 40,
      region: { x0: 35, y0: 0, x1: 365, y1: H - 1 },
    });
    const days = resampleToDays(pts, bbtCal, 15, rounderFor("temp"));
    for (let d = 1; d <= bbt.length; d++) {
      const got = days[d - 1].value;
      expect(got, `BBT day ${d}`).not.toBeNull();
      if (got != null) expect(Math.abs(got - bbt[d - 1])).toBeLessThanOrEqual(0.05);
    }
  });

  it("recovers LH (orange/left axis, different scale) independently of BBT", () => {
    const chart = makeDualChart();
    const pts = extractCurvePixels({
      data: chart.data,
      width: W,
      height: H,
      target: ORANGE,
      tolerance: 40,
      region: { x0: 35, y0: 0, x1: 365, y1: H - 1 },
    });
    const days = resampleToDays(pts, lhCal, 15, rounderFor("lh"));
    for (let d = 1; d <= lh.length; d++) {
      const got = days[d - 1].value;
      expect(got, `LH day ${d}`).not.toBeNull();
      if (got != null) expect(Math.abs(got - lh[d - 1])).toBeLessThanOrEqual(0.15);
    }
  });

  it("lh rounder quantizes to 0.1", () => {
    const r = rounderFor("lh");
    expect(r(2.63)).toBe(2.6);
    expect(rounderFor("temp")(36.567)).toBe(36.57);
  });
});
