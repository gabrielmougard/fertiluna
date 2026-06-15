/**
 * autoExtractUI.ts — controller for the "Extraction auto (IA)" mode.
 *
 * Flow (zero numeric input from the user):
 *   1. user uploads a chart screenshot,
 *   2. the on-device CV pipeline reads the chart — per-day BBT/LH values, the
 *      BBT axis unit (°C vs °F) and ranges, plus the detection geometry,
 *   3. the detected boxes (plot region, per-day markers, axis labels, table
 *      cells) are drawn over the image, with hover tooltips for inspection,
 *   4. "Importer" emits `fertiluna:digitized` with de-normalized { temps, lh }.
 *
 * Everything is on-device. If the model is unavailable or fails, the host
 * component falls back to the manual calibration digitizer.
 */

import {
  denormalizeSeries,
  type VisionPrediction,
  type VisionSeriesKind,
} from "./visionInference";
import { assessVisionQuality } from "./visionConfidence";
import { t, tf, toLocale } from "./i18n";
import { getConsent, setConsent, getRouterConfig } from "./consent";
import {
  chooseBackend, digitizeCloud, digitizeOnDevice,
} from "./router/backendRouter";
import { ensurePaddleOcrLoaded } from "./paddleOcr";
import type { DigitizeResult } from "./router/types";
import type { CvDetections } from "./visionCv/types";
import { BBT_SCALES, LH_RANGE } from "./visionCv/constants";

/** Map a router DigitizeResult into the VisionPrediction shape the auto panel
 *  already renders, so the cloud path reuses the same preview + import flow. */
function digitizeResultToPrediction(r: DigitizeResult): VisionPrediction {
  const mk = (s: number): VisionPrediction["temp"] => ({
    kind: s === 0 ? "temp" : "lh",
    normalized: r.value[s].map((v, i) => (r.present[s][i] > 0.5 ? v : null)),
    confidence: r.present[s].slice(),
    presentCount: r.present[s].reduce((a, b) => a + b, 0),
  });
  const sc = BBT_SCALES[r.scaleIdx];
  return {
    temp: mk(0),
    lh: mk(1),
    bbtScale: { index: r.scaleIdx, label: sc.label, min: sc.min, max: sc.max },
    lhRange: { min: LH_RANGE.min, max: LH_RANGE.max },
  };
}

export interface AutoExtractElements {
  root: HTMLElement;
  fileInput: HTMLInputElement;
  canvas: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
  status: HTMLElement;
  /** optional per-series include toggles (auto-checked from detection) */
  tempEnable?: HTMLInputElement;
  lhEnable?: HTMLInputElement;
  /** optional manual axis-range overrides for the auto (IA) path — mirror of
   *  the Python --temp-min/--temp-max / --lh-min/--lh-max overrides. Passed by
   *  CurveDigitizer; consuming them in the controller is a follow-up. */
  tempMin?: HTMLInputElement;
  tempMax?: HTMLInputElement;
  lhMin?: HTMLInputElement;
  lhMax?: HTMLInputElement;
  /** shows the auto-detected scale, e.g. "Échelle : °C (35.6–37.4)" */
  scaleInfo?: HTMLElement;
  /** opt-in checkbox for the cloud (LLM) analysis path; present only when the
   *  cloud backend is enabled. Persists via consent.ts and gates the router. */
  cloudConsent?: HTMLInputElement;
  importBtn: HTMLButtonElement;
  /** called once the chosen image has been drawn to the canvas (before/while
   *  reading), so the host can collapse the dropzone and show the stage. */
  onImageLoaded?: () => void;
  /** called when the model can't be used, so the host can offer manual mode. */
  onUnavailable?: (reason: string) => void;
}

export function initAutoExtract(el: AutoExtractElements): void {
  const ctx = el.canvas.getContext("2d", { willReadFrequently: true })!;
  const octx = el.overlay.getContext("2d")!;
  // UI locale, mirrored from the <html lang> the server set.
  const locale = toLocale(document.documentElement.lang);
  // the .dz-canvas-wrap that holds both canvases (overlay box rendering)
  const stage = el.canvas.closest<HTMLElement>(".dz-canvas-wrap");
  // modern loader overlay shown while the CV pipeline computes detections
  const loader = stage?.querySelector<HTMLElement>(".dz-loader") ?? null;
  const loaderLabel = loader?.querySelector<HTMLElement>(".dz-loader-label") ?? null;
  function showLoader(msg: string) {
    if (loaderLabel) loaderLabel.textContent = msg;
    if (loader) loader.hidden = false;
  }
  function hideLoader() {
    if (loader) loader.hidden = true;
  }
  // The on-device CV pipeline is mostly synchronous CPU work, so without a
  // forced paint the loader would be toggled on and off within a single frame
  // and never actually render. Awaiting two rAFs lets the browser paint it
  // before the heavy compute blocks the main thread.
  const nextPaint = () =>
    new Promise<void>((res) =>
      requestAnimationFrame(() => requestAnimationFrame(() => res())),
    );

  let prediction: VisionPrediction | null = null;
  let detections: CvDetections | null = null;

  function setStatus(s: string) {
    el.status.textContent = s;
  }

  // Floating tooltip for hovering detected boxes. Appended to the (non-clipped)
  // .dz-stage so it isn't cut off by the canvas wrap's overflow:hidden.
  const stageOuter = stage?.parentElement ?? null;
  const tip = document.createElement("div");
  tip.className = "dz-tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  stageOuter?.appendChild(tip);

  // Warm the on-device OCR (axis + table reading) lazily, on first sign the
  // user intends to import an image — hovering/focusing the dropzone or opening
  // the file picker. The OCR model is ~8 MB, so eager warm-on-load used to put
  // that download on the critical path of EVERY tool-page visit (including the
  // many mobile users who just type values or read the page), wrecking mobile
  // Lighthouse TBT. Warming on intent keeps it snappy when actually used while
  // keeping initial load light. A warm-up failure is non-fatal (CV runs without
  // OCR), and ensurePaddleOcrLoaded() is idempotent so the import path can call
  // it again safely.
  let ocrWarmArmed = false;
  const warmOcr = () => {
    if (ocrWarmArmed) return;
    ocrWarmArmed = true;
    ensurePaddleOcrLoaded().then(
      () => setStatus(""),
      (e) => console.warn("[auto] OCR warm-up failed (non-fatal):", e),
    );
  };
  const dropzone = el.fileInput.closest<HTMLElement>("label") ?? el.root;
  ["pointerenter", "focusin"].forEach((ev) =>
    dropzone.addEventListener(ev, warmOcr, { once: true }),
  );
  el.fileInput.addEventListener("click", warmOcr, { once: true });

  // Consent checkbox: reflect the persisted choice, persist on change.
  if (el.cloudConsent) {
    el.cloudConsent.checked = getConsent();
    el.cloudConsent.addEventListener("change", () => {
      setConsent(el.cloudConsent!.checked);
    });
  }

  function tempIncluded(): boolean {
    return el.tempEnable ? el.tempEnable.checked : true;
  }
  function lhIncluded(): boolean {
    return el.lhEnable ? el.lhEnable.checked : true;
  }

  el.fileInput.addEventListener("change", () => {
    const file = el.fileInput.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const im = new Image();
    im.onload = async () => {
      const maxW = 900;
      const scale = im.width > maxW ? maxW / im.width : 1;
      const w = Math.round(im.width * scale);
      const h = Math.round(im.height * scale);
      el.canvas.width = w;
      el.canvas.height = h;
      el.overlay.width = w;
      el.overlay.height = h;
      ctx.drawImage(im, 0, 0, w, h);
      octx.clearRect(0, 0, w, h);
      URL.revokeObjectURL(url);
      hideTip();
      el.root.classList.add("has-image");
      el.onImageLoaded?.();
      setStatus(t("dz.analyzing", locale));
      showLoader(t("dz.analyzing", locale));
      await nextPaint(); // ensure the loader actually renders before compute

      // Hybrid router: with consent + cloud enabled, a share of uploads go to
      // the cloud LLM (enhanced); everything else stays on-device (CV pipeline).
      // Cloud failures fall back transparently to the on-device pipeline.
      const cfg = getRouterConfig();
      const useCloud = chooseBackend(cfg) === "cloud";
      let result: DigitizeResult;
      try {
        if (useCloud) {
          setStatus(t("dz.analyzingCloud", locale));
          showLoader(t("dz.analyzingCloud", locale));
          try {
            result = await digitizeCloud(im, locale);
          } catch (cloudErr) {
            console.warn("[auto] cloud failed, on-device fallback:", cloudErr);
            showLoader(t("dz.analyzingDevice", locale));
            await nextPaint();
            result = await digitizeOnDevice(im);
          }
        } else {
          result = await digitizeOnDevice(im);
        }
      } catch (e) {
        console.error(e);
        hideLoader();
        el.onUnavailable?.(t("dz.failed", locale));
        return;
      }
      hideLoader();
      prediction = digitizeResultToPrediction(result);
      // Only the on-device CV pipeline reports box geometry; cloud omits it.
      detections = result.detections ?? null;
      applyDetection();
      drawScene();
      updateImportEnabled();
    };
    im.onerror = () => setStatus(t("dz.unreadable", locale));
    im.src = url;
  });

  function applyDetection() {
    if (!prediction) return;
    const tHas = prediction.temp.presentCount >= 3;
    const lHas = prediction.lh.presentCount >= 3;
    if (el.tempEnable) el.tempEnable.checked = tHas;
    if (el.lhEnable) el.lhEnable.checked = lHas;

    // show the auto-detected BBT scale
    if (el.scaleInfo) {
      const s = prediction.bbtScale;
      const unit = s.label === "fahrenheit" ? "°F" : "°C";
      el.scaleInfo.textContent = tHas
        ? tf("dz.scaleDetected", locale, { unit, min: s.min, max: s.max })
        : "";
    }

    // Confidence-driven failure UX: decide whether to trust the extraction,
    // ask the user to verify, or reject the image outright.
    const q = assessVisionQuality(prediction);
    if (q.status === "not_a_chart") {
      el.onUnavailable?.(t("dz.noCurve", locale));
      setStatus(t("dz.noCurveStatus", locale));
      return;
    }

    const parts: string[] = [];
    if (tHas) {
      const unit = prediction.bbtScale.label === "fahrenheit" ? "°F" : "°C";
      parts.push(
        tf("dz.detectedTemp", locale, { unit, n: prediction.temp.presentCount }),
      );
    }
    if (lHas) {
      parts.push(tf("dz.detectedLh", locale, { n: prediction.lh.presentCount }));
    }
    if (!parts.length) {
      setStatus(t("dz.noNet", locale));
      return;
    }
    const detected = tf("dz.detected", locale, { parts: parts.join(" + ") });
    if (q.status === "low_confidence") {
      // Result shown, but flagged — the user must check it carefully.
      setStatus(
        tf("dz.lowConfidence", locale, {
          pct: Math.round(q.confidence * 100),
          detected,
        }),
      );
    } else {
      setStatus(tf("dz.verifyImport", locale, { detected }));
    }
  }

  function currentValues(): {
    temps: (number | null)[] | null;
    lh: (number | null)[] | null;
  } {
    if (!prediction) return { temps: null, lh: null };
    let temps: (number | null)[] | null = null;
    let lh: (number | null)[] | null = null;
    if (tempIncluded() && prediction.temp.presentCount > 0) {
      temps = denormalizeSeries(
        prediction.temp,
        prediction.bbtScale.min,
        prediction.bbtScale.max,
      );
    }
    if (lhIncluded() && prediction.lh.presentCount > 0) {
      lh = denormalizeSeries(
        prediction.lh,
        prediction.lhRange.min,
        prediction.lhRange.max,
      );
    }
    return { temps, lh };
  }

  // ── Curve fallback (cloud path only: it returns no box geometry) ──────────
  // Draws the recovered series as a static polyline over the screenshot — used
  // only when `detections` is null (i.e. the cloud LLM backend).
  function drawCurveFallback() {
    octx.clearRect(0, 0, el.overlay.width, el.overlay.height);
    if (!prediction) return;
    const W = el.overlay.width, H = el.overlay.height;
    const series: [VisionSeriesKind, string, boolean][] = [
      ["temp", "rgba(86,120,214,0.95)", tempIncluded()],
      ["lh", "rgba(214,68,127,0.95)", lhIncluded()],
    ];
    for (const [kind, color, enabled] of series) {
      if (!enabled) continue;
      const p = prediction[kind];
      const days = p.normalized.length;
      const dayX = (d: number) => ((d + 0.5) / days) * W;
      const valY = (n: number) => (1 - n) * H;
      octx.fillStyle = color;
      octx.strokeStyle = color;
      octx.lineWidth = 2.6;
      octx.lineJoin = "round";
      octx.lineCap = "round";
      let started = false;
      octx.beginPath();
      for (let d = 0; d < days; d++) {
        const n = p.normalized[d];
        if (n == null) { started = false; continue; }
        const x = dayX(d), y = valY(n);
        if (!started) { octx.moveTo(x, y); started = true; }
        else octx.lineTo(x, y);
      }
      octx.stroke();
      for (let d = 0; d < days; d++) {
        const n = p.normalized[d];
        if (n == null) continue;
        octx.beginPath();
        octx.arc(dayX(d), valY(n), 3, 0, Math.PI * 2);
        octx.fill();
      }
    }
  }

  // ── Detection overlay (on-device CV pipeline) ─────────────────────────────
  type Hit = (
    | { box: { x: number; y: number; w: number; h: number } }
    | { circle: { cx: number; cy: number; r: number } }
  ) & { title: string };
  // Hover hit-targets, in overlay-canvas pixels, pushed back-to-front so the
  // topmost (markers) wins when several overlap.
  let hits: Hit[] = [];
  let hoverIdx = -1;

  const SERIES_COLOR: Record<"temp" | "lh", string> = {
    temp: "#5678d6",
    lh: "rgba(214,68,127,0.95)",
  };
  const PLOT_COLOR = "rgba(46,158,107,0.95)";
  function axisColor(kind: string): string {
    if (kind === "ratio") return "rgba(255,140,40,0.95)";
    if (kind === "level") return "rgba(160,90,200,0.95)";
    if (kind.startsWith("bbt")) return "rgba(86,120,214,0.95)";
    return "rgba(150,150,150,0.9)";
  }
  const TABLE_ROW_COLOR: Record<string, string> = {
    calendar: "rgba(180,170,50,0.95)", CD: "rgba(200,100,200,0.95)",
    DPO: "rgba(50,190,190,0.95)", Sex: "rgba(70,110,230,0.95)",
    CM: "rgba(200,170,50,0.95)", Symptoms: "rgba(200,130,60,0.95)",
    hCG: "rgba(140,90,210,0.95)",
  };
  function tableColor(name: string): string {
    return TABLE_ROW_COLOR[name] ?? "rgba(130,130,130,0.9)";
  }
  function unit(): string {
    return prediction?.bbtScale.label === "fahrenheit" ? "°F" : "°C";
  }

  /** Render the detection overlay (or the curve fallback when there's no box
   *  geometry), rebuilding the hover hit-targets. A non-negative `highlight`
   *  (a hit index) draws that target emphasised. */
  function drawScene(highlight = -1) {
    if (!detections) { hits = []; drawCurveFallback(); return; }
    const W = el.overlay.width, H = el.overlay.height;
    const sx = W / detections.work.width;
    const sy = H / detections.work.height;
    octx.clearRect(0, 0, W, H);
    hits = [];

    const strokeBox = (
      x: number, y: number, w: number, h: number,
      color: string, title: string, lw = 1.5,
    ) => {
      const on = hits.length === highlight;
      if (on) {
        octx.fillStyle = color.replace(/[\d.]+\)$/, "0.18)");
        octx.fillRect(x, y, w, h);
      }
      octx.strokeStyle = color;
      octx.lineWidth = on ? lw + 1.5 : lw;
      octx.strokeRect(x, y, w, h);
      hits.push({ box: { x, y, w, h }, title });
    };

    // plot region (drawn first, lowest hover priority)
    const p = detections.plot;
    strokeBox(p.x0 * sx, p.y0 * sy, (p.x1 - p.x0) * sx, (p.y1 - p.y0) * sy,
      PLOT_COLOR, tf("dz.tipPlot", locale, { method: p.method }), 2);

    // y-axis label boxes
    for (const col of detections.axisColumns) {
      const color = axisColor(col.kind);
      for (const lb of col.labels) {
        const [x0, y0, x1, y1] = lb.bbox;
        const v = lb.value != null ? lb.value : lb.text;
        strokeBox(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy, color,
          tf("dz.tipAxis", locale, { kind: col.kind, v }));
      }
    }

    // bottom-table cells
    for (const row of detections.tableRows) {
      const color = tableColor(row.name);
      const [lx0, ly0, lx1, ly1] = row.labelBbox;
      if (lx1 > lx0) {
        strokeBox(lx0 * sx, ly0 * sy, (lx1 - lx0) * sx, (ly1 - ly0) * sy,
          color, tf("dz.tipRow", locale, { name: row.name }));
      }
      for (const c of row.cells) {
        const [x0, y0, x1, y1] = c.bbox;
        if (x1 - x0 < 2) continue;
        const label = c.text
          ? `${row.name} · ${c.text}`
          : tf("dz.tipCellEmpty", locale, { name: row.name });
        strokeBox(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy, color, label);
      }
    }

    // per-day markers (drawn last, highest hover priority)
    for (const m of detections.markers) {
      if (m.series === "temp" && !tempIncluded()) continue;
      if (m.series === "lh" && !lhIncluded()) continue;
      const color = SERIES_COLOR[m.series];
      const cx = m.cx * sx, cy = m.cy * sy;
      const r = Math.max(6, m.radius * sx);
      const on = hits.length === highlight;
      octx.strokeStyle = color;
      octx.lineWidth = on ? 3 : 2;
      octx.beginPath();
      octx.arc(cx, cy, r, 0, Math.PI * 2);
      octx.stroke();
      octx.beginPath();
      octx.moveTo(cx - 4, cy); octx.lineTo(cx + 4, cy);
      octx.moveTo(cx, cy - 4); octx.lineTo(cx, cy + 4);
      octx.stroke();
      const who = m.series === "temp" ? "BBT" : "LH";
      const val = m.valueReal == null ? ""
        : m.series === "temp"
          ? ` · ${m.valueReal.toFixed(2)} ${unit()}`
          : ` · ${m.valueReal.toFixed(1)}`;
      hits.push({
        circle: { cx, cy, r: r + 4 },
        title: `${who} · ${tf("dz.tipDay", locale, { day: m.day })}${val}`,
      });
    }
  }

  function hitAt(px: number, py: number): number {
    for (let i = hits.length - 1; i >= 0; i--) {
      const h = hits[i];
      if ("circle" in h) {
        const { cx, cy, r } = h.circle;
        if ((px - cx) ** 2 + (py - cy) ** 2 <= r * r) return i;
      } else {
        const { x, y, w, h: bh } = h.box;
        if (px >= x && px <= x + w && py >= y && py <= y + bh) return i;
      }
    }
    return -1;
  }

  function hideTip() {
    tip.hidden = true;
    el.overlay.style.cursor = "";
  }

  el.overlay.addEventListener("mousemove", (e) => {
    if (!detections || !hits.length) return;
    const rect = el.overlay.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * el.overlay.width;
    const py = ((e.clientY - rect.top) / rect.height) * el.overlay.height;
    const idx = hitAt(px, py);
    if (idx !== hoverIdx) {
      hoverIdx = idx;
      drawScene(idx);
    }
    if (idx < 0) { hideTip(); return; }
    el.overlay.style.cursor = "help";
    tip.textContent = hits[idx].title;
    tip.hidden = false;
    if (stageOuter) {
      const sr = stageOuter.getBoundingClientRect();
      tip.style.left = `${e.clientX - sr.left}px`;
      tip.style.top = `${e.clientY - sr.top}px`;
    }
  });
  el.overlay.addEventListener("mouseleave", () => {
    hoverIdx = -1;
    hideTip();
    if (detections) drawScene();
  });

  function updateImportEnabled() {
    const { temps, lh } = currentValues();
    const any =
      (temps && temps.some((v) => v != null)) ||
      (lh && lh.some((v) => v != null));
    el.importBtn.disabled = !any;
  }

  for (const input of [el.tempEnable, el.lhEnable]) {
    if (!input) continue;
    input.addEventListener("change", () => {
      drawScene();
      updateImportEnabled();
    });
  }

  el.importBtn.addEventListener("click", () => {
    const { temps, lh } = currentValues();
    if (!temps && !lh) return;
    el.root.dispatchEvent(
      new CustomEvent("fertiluna:digitized", {
        bubbles: true,
        detail: { temps, lh },
      }),
    );
  });
}
