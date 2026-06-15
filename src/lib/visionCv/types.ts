/**
 * Shared types for the in-browser CV digitizer. The `ChartResultCv` shape
 * mirrors the Python pipeline's ChartResult / CLI JSON so the two backends
 * are interchangeable behind the smart router.
 */

export type ChartStatus = "extracted" | "low_confidence" | "not_a_chart";

/** A single-channel binary mask (0/255) over a working-canvas-sized image. */
export interface Mask {
  data: Uint8Array; // length = width*height, row-major
  width: number;
  height: number;
}

export interface SeriesMasks {
  blue: Mask; // BBT ink
  orange: Mask; // LH "Ratio" ink (candidate LH line)
  purple: Mask; // LH "Level" ink (candidate LH line)
}

export interface PlotRegion {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  method: string;
}

/**
 * Geometry detected by the classical CV pipeline, surfaced so the UI can draw
 * the same overlay the Python debug view renders (plot region, per-day markers,
 * axis-label ticks, table cells). All coordinates are in WORKING-CANVAS pixels;
 * `work` carries those dimensions so the UI can scale them to its display
 * canvas. Only the on-device CV backend produces these — the cloud path omits
 * them.
 */
export interface CvMarkerDetection {
  series: "temp" | "lh";
  cx: number;
  cy: number;
  radius: number;
  /** 1-based day index after left-packing. */
  day: number;
  /** ring-coverage detection score. */
  score: number;
  /** de-normalized value in real units (°C/°F or LH ratio), or null. */
  valueReal: number | null;
}

export interface CvAxisLabelDetection {
  bbox: [number, number, number, number];
  value: number | null;
  text: string;
}

export interface CvAxisColumnDetection {
  kind: string; // ratio | level | bbt_f | bbt_c | unknown
  side: string; // left | right
  labels: CvAxisLabelDetection[];
}

export interface CvTableCellDetection {
  bbox: [number, number, number, number];
  text: string | null;
  kind: string;
}

export interface CvTableRowDetection {
  name: string;
  labelBbox: [number, number, number, number];
  cells: CvTableCellDetection[];
}

export interface CvDetections {
  /** working-canvas dimensions these coordinates live in. */
  work: { width: number; height: number };
  plot: { x0: number; y0: number; x1: number; y1: number; method: string };
  markers: CvMarkerDetection[];
  axisColumns: CvAxisColumnDetection[];
  tableRows: CvTableRowDetection[];
}

export interface ChartResultCv {
  /** (N_SERIES × N_DAYS) normalized [0,1] within each series' axis range. */
  value: number[][];
  /** (N_SERIES × N_DAYS) {0,1}. */
  present: number[][];
  /** (N_SERIES × N_DAYS) {0,1}; interpolated (synthesised) days, kept apart. */
  interpolated: number[][];
  scaleIdx: number; // 0=celsius, 1=fahrenheit
  scaleLabel: string;
  confidence: number; // [0,1]
  status: ChartStatus;
  visibleDays: number;
  truncated: boolean;
  /** Parsed bottom-of-screen table (calendar/CD/DPO/Sex/CM/Symptoms/hCG), or
   *  undefined when absent / OCR unavailable. Type-only import avoids a cycle. */
  table?: import("./tableExtract").TableData;
  /** Detected geometry for the overlay (on-device CV backend only). */
  detections?: CvDetections;
}
