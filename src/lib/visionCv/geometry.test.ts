import { describe, it, expect } from "vitest";
import { connectedComponents } from "./ops";
import { autocorrCellPx, gridFromMarkerSpacing } from "./dayAxis";
import { detectPlotRegion } from "./plotRegion";
import type { Mask, SeriesMasks } from "./types";

function blank(w: number, h: number): Mask {
  return { data: new Uint8Array(w * h), width: w, height: h };
}
function fillRect(m: Mask, x0: number, y0: number, x1: number, y1: number) {
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++) m.data[y * m.width + x] = 255;
}

describe("connectedComponents", () => {
  it("labels two separate squares with correct area + bbox", () => {
    const m = blank(40, 20);
    fillRect(m, 2, 2, 6, 6); // 4×4 = 16
    fillRect(m, 20, 10, 25, 15); // 5×5 = 25
    const { stats } = connectedComponents(m, 8);
    expect(stats.length).toBe(2);
    const areas = stats.map((s) => s.area).sort((a, b) => a - b);
    expect(areas).toEqual([16, 25]);
  });
  it("does not bridge across the row edge (no wrap-around)", () => {
    const m = blank(10, 3);
    m.data[0 * 10 + 9] = 255; // right edge row 0
    m.data[1 * 10 + 0] = 255; // left edge row 1 — must NOT join
    const { stats } = connectedComponents(m, 8);
    expect(stats.length).toBe(2);
  });
});

describe("autocorrCellPx", () => {
  it("recovers a periodic stripe pitch", () => {
    const W = 800, H = 200, pitch = 40;
    const blue = blank(W, H);
    for (let cx = 20; cx < W; cx += pitch) {
      // a 3px-wide vertical stripe at each multiple of `pitch`
      for (let x = cx - 1; x <= cx + 1; x++) fillRect(blue, x, 10, x + 1, H - 10);
    }
    const masks: SeriesMasks = {
      blue, orange: blank(W, H), purple: blank(W, H),
    };
    const px = autocorrCellPx(masks, { x0: 0, y0: 0, x1: W, y1: H, method: "t" });
    expect(px).not.toBeNull();
    expect(Math.abs((px as number) - pitch)).toBeLessThanOrEqual(2);
  });
});

describe("gridFromMarkerSpacing", () => {
  it("recovers the pitch and lands every marker on a cell center", () => {
    const plot = { x0: 0, y0: 0, x1: 800, y1: 200, method: "t" };
    const xs = [100, 140, 180, 220, 260]; // pitch 40
    const g = gridFromMarkerSpacing(xs, plot, 40);
    expect(Math.abs(g.cellPx - 40)).toBeLessThanOrEqual(1);
    // Perfectly-periodic markers fix the GRID but not its absolute origin
    // (ambiguous by whole cells); the meaningful property is that each marker
    // sits within half a cell of some cell center.
    for (const mx of xs) {
      const nearest = Math.min(...g.cells.map((c) => Math.abs(c - mx)));
      expect(nearest).toBeLessThanOrEqual(2);
    }
  });
});

describe("detectPlotRegion", () => {
  it("excludes a small isolated icon below the chart line", () => {
    const W = 800, H = 600;
    const blue = blank(W, H);
    // Realistic chart curve: a thick diagonal so the component's bbox is TALL
    // (a flat bar would be filtered as a cover-line, which is correct).
    for (let x = 50; x < 700; x++) {
      const y = 90 + Math.round(((x - 50) / 650) * 150); // 90 → 240
      for (let t = -3; t <= 3; t++) {
        const yy = y + t;
        if (yy >= 0 && yy < H) blue.data[yy * W + x] = 255;
      }
    }
    const orange = blank(W, H);
    // a tiny icon far below (like a table heart) — should be excluded
    fillRect(orange, 120, 520, 132, 532);
    const masks: SeriesMasks = { blue, orange, purple: blank(W, H) };
    const plot = detectPlotRegion(masks);
    expect(plot.method).toBe("ink");
    expect(plot.y1).toBeLessThan(300); // icon at y~520 excluded
  });
});
