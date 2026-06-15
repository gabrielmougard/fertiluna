/**
 * CV primitives the pipeline needs that cv2 provides in Python.
 *
 * `connectedComponents` is implemented in PURE TYPESCRIPT here — it's the key
 * primitive marker detection and plot-region rely on, and a stack flood-fill
 * is small and fast enough that we avoid pulling in OpenCV.js just for it.
 *
 * `morphologyClose` / `findContourHoles` are heavier; they're stubbed with a
 * clear interface so the porting order is explicit. When we wire OpenCV.js
 * (the planned backend for the remaining stages), these become thin adapters.
 */

import type { Mask } from "./types";

export interface ComponentStats {
  label: number;
  area: number;
  x0: number;
  y0: number;
  x1: number; // inclusive
  y1: number; // inclusive
  cx: number; // centroid
  cy: number;
}

/**
 * Label connected foreground (>0) runs. Returns the label image and per-label
 * stats (area, bounding box, centroid). Background is label 0. 8-connectivity
 * by default to match cv2's `connectivity=8`.
 */
export function connectedComponents(
  mask: Mask,
  connectivity: 4 | 8 = 8,
): { labels: Int32Array; stats: ComponentStats[] } {
  const { data, width, height } = mask;
  const labels = new Int32Array(width * height); // 0 = background/unvisited
  const stats: ComponentStats[] = [];
  const stack: number[] = [];
  const neigh8 = [-1, 1, -width, width, -width - 1, -width + 1, width - 1, width + 1];
  const neigh4 = [-1, 1, -width, width];
  const neigh = connectivity === 8 ? neigh8 : neigh4;
  let next = 1;

  for (let start = 0; start < data.length; start++) {
    if (data[start] === 0 || labels[start] !== 0) continue;
    const label = next++;
    let area = 0;
    let x0 = width, y0 = height, x1 = 0, y1 = 0;
    let sx = 0, sy = 0;
    stack.length = 0;
    stack.push(start);
    labels[start] = label;
    while (stack.length) {
      const idx = stack.pop()!;
      const x = idx % width;
      const y = (idx / width) | 0;
      area++;
      sx += x; sy += y;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      for (const dN of neigh) {
        const ni = idx + dN;
        if (ni < 0 || ni >= data.length) continue;
        // guard horizontal wrap-around at row edges
        const nx = ni % width;
        if (Math.abs(nx - x) > 1) continue;
        if (data[ni] !== 0 && labels[ni] === 0) {
          labels[ni] = label;
          stack.push(ni);
        }
      }
    }
    stats.push({
      label, area, x0, y0, x1, y1,
      cx: sx / area, cy: sy / area,
    });
  }
  return { labels, stats };
}

/** 4πA/P² circularity proxy from a component's area + bbox perimeter estimate. */
export function bboxFill(s: ComponentStats): number {
  const w = s.x1 - s.x0 + 1;
  const h = s.y1 - s.y0 + 1;
  return s.area / Math.max(1, w * h);
}

/**
 * Binary dilation with a (kw × kh) rectangular structuring element, separable
 * (horizontal then vertical pass). Pure TS — used to merge the digit glyphs of
 * one axis number into one box without bridging to the next column.
 */
export function dilate(mask: Mask, kw: number, kh: number): Mask {
  const { data, width, height } = mask;
  const hw = kw >> 1, hh = kh >> 1;
  const tmp = new Uint8Array(width * height);
  // horizontal
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let on = 0;
      for (let dx = -hw; dx <= hw; dx++) {
        const xx = x + dx;
        if (xx >= 0 && xx < width && data[row + xx]) { on = 255; break; }
      }
      tmp[row + x] = on;
    }
  }
  // vertical
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let on = 0;
      for (let dy = -hh; dy <= hh; dy++) {
        const yy = y + dy;
        if (yy >= 0 && yy < height && tmp[yy * width + x]) { on = 255; break; }
      }
      out[y * width + x] = on;
    }
  }
  return { data: out, width, height };
}
