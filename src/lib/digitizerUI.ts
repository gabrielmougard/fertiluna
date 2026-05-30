/**
 * digitizerUI.ts — interactive controller for the screenshot digitizer.
 *
 * Per-series, dual-axis workflow (handles real charts like Inito/Premom that
 * plot BBT and an LH/hormone line on TWO different Y-axes):
 *
 *   1. Calibrate the DAY axis once (two clicks on known cycle-day gridlines).
 *   2. For EACH series you want (BBT temperature, and/or LH):
 *        a. select the series tab,
 *        b. calibrate THAT series' value axis (two clicks on its own axis —
 *           e.g. the right axis for BBT, the left axis for LH),
 *        c. click the series' coloured line to pick its colour,
 *        d. the points are extracted and previewed; tune tolerance if needed.
 *   3. Import → fills both the temperature and LH columns of the table.
 *
 * Everything runs locally; the image never leaves the browser. A magnifier
 * loupe assists precise clicking on busy charts.
 */

import {
  type Calibration,
  type RGBA,
  type DayValue,
  type SeriesKind,
  extractCurvePixels,
  resampleToDays,
  suggestCurveColor,
  guessPlotRegion,
  pixelAt,
  unmapAxis,
  rounderFor,
} from "./curveDigitizer";
import { CYCLE_MAX_DAYS } from "./constants";

interface RefPoint {
  pixel: number;
  value: number;
}

interface SeriesState {
  kind: SeriesKind;
  valueRefs: RefPoint[]; // 0..2 (pixel = Y)
  color: RGBA | null;
  tolerance: number;
  days: DayValue[];
}

type Phase =
  | "idle"
  | "day1"
  | "day2"
  | "value1"
  | "value2"
  | "color"
  | "ready";

const SERIES_META: Record<
  SeriesKind,
  { label: string; axisSide: string; unit: string; ex1: number; ex2: number }
> = {
  temp: {
    label: "Température (BBT)",
    axisSide: "souvent l'axe de DROITE (°C)",
    unit: "°C",
    ex1: 36.0,
    ex2: 37.0,
  },
  lh: {
    label: "LH / hormone",
    axisSide: "souvent l'axe de GAUCHE",
    unit: "",
    ex1: 0.1,
    ex2: 1.0,
  },
};

function phasePrompt(phase: Phase, kind: SeriesKind): string {
  const m = SERIES_META[kind];
  switch (phase) {
    case "idle":
      return "Importez une capture d'écran de votre courbe pour commencer.";
    case "day1":
      return "Axe des jours (1/2) — cliquez sur un repère de jour connu (ligne ZT/jour), puis indiquez son numéro.";
    case "day2":
      return "Axe des jours (2/2) — cliquez sur un autre repère de jour, plus à droite.";
    case "value1":
      return `« ${m.label} » — valeur (1/2) : cliquez sur une graduation de son axe (${m.axisSide}), puis indiquez sa valeur.`;
    case "value2":
      return `« ${m.label} » — valeur (2/2) : cliquez sur une autre graduation du même axe.`;
    case "color":
      return `« ${m.label} » — cliquez directement sur sa courbe pour sélectionner sa couleur.`;
    case "ready":
      return `« ${m.label} » détectée. Vérifiez l'aperçu, ajustez la tolérance, ou passez à l'autre série / importez.`;
  }
}

export interface DigitizerElements {
  root: HTMLElement;
  fileInput: HTMLInputElement;
  canvas: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
  loupe: HTMLCanvasElement;
  prompt: HTMLElement;
  toleranceInput: HTMLInputElement;
  toleranceLabel: HTMLElement;
  redoBtn: HTMLButtonElement;
  importBtn: HTMLButtonElement;
  status: HTMLElement;
  seriesTabs: HTMLElement; // container with [data-series] buttons
}

export function initDigitizer(el: DigitizerElements): void {
  const ctx = el.canvas.getContext("2d", { willReadFrequently: true })!;
  const octx = el.overlay.getContext("2d")!;
  const lctx = el.loupe.getContext("2d")!;

  let imgData: ImageData | null = null;
  let phase: Phase = "idle";
  let dayRefs: RefPoint[] = [];
  let active: SeriesKind = "temp";

  const series: Record<SeriesKind, SeriesState> = {
    temp: { kind: "temp", valueRefs: [], color: null, tolerance: 45, days: [] },
    lh: { kind: "lh", valueRefs: [], color: null, tolerance: 45, days: [] },
  };

  function cur(): SeriesState {
    return series[active];
  }

  function setPrompt() {
    el.prompt.textContent = phasePrompt(phase, active);
  }
  function setStatus(s: string) {
    el.status.textContent = s;
  }

  function calibrationFor(s: SeriesState): Calibration | null {
    if (dayRefs.length < 2 || s.valueRefs.length < 2) return null;
    return {
      dayAxis: {
        pixel1: dayRefs[0].pixel,
        value1: dayRefs[0].value,
        pixel2: dayRefs[1].pixel,
        value2: dayRefs[1].value,
      },
      valueAxis: {
        pixel1: s.valueRefs[0].pixel,
        value1: s.valueRefs[0].value,
        pixel2: s.valueRefs[1].pixel,
        value2: s.valueRefs[1].value,
      },
    };
  }

  // ── tabs ──
  function refreshTabs() {
    el.seriesTabs
      .querySelectorAll<HTMLButtonElement>("[data-series]")
      .forEach((b) => {
        const k = b.dataset.series as SeriesKind;
        b.classList.toggle("active", k === active);
        const done = series[k].days.some((d) => d.value != null);
        b.classList.toggle("done", done);
      });
  }

  // determine the phase to resume at when (re)selecting a series
  function phaseForActive(): Phase {
    if (dayRefs.length < 1) return "day1";
    if (dayRefs.length < 2) return "day2";
    const s = cur();
    if (s.valueRefs.length < 1) return "value1";
    if (s.valueRefs.length < 2) return "value2";
    if (!s.color) return "color";
    return "ready";
  }

  el.seriesTabs.addEventListener("click", (ev) => {
    const btn = (ev.target as HTMLElement).closest<HTMLButtonElement>(
      "[data-series]",
    );
    if (!btn || !imgData) return;
    active = btn.dataset.series as SeriesKind;
    el.toleranceInput.value = String(cur().tolerance);
    el.toleranceLabel.textContent = String(cur().tolerance);
    phase = phaseForActive();
    setPrompt();
    refreshTabs();
    drawOverlay();
  });

  // ── image load ──
  el.fileInput.addEventListener("change", () => {
    const file = el.fileInput.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const maxW = 1100;
      const scale = img.width > maxW ? maxW / img.width : 1;
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      el.canvas.width = w;
      el.canvas.height = h;
      el.overlay.width = w;
      el.overlay.height = h;
      ctx.drawImage(img, 0, 0, w, h);
      imgData = ctx.getImageData(0, 0, w, h);
      URL.revokeObjectURL(url);

      dayRefs = [];
      series.temp = {
        kind: "temp",
        valueRefs: [],
        color: null,
        tolerance: 45,
        days: [],
      };
      series.lh = {
        kind: "lh",
        valueRefs: [],
        color: null,
        tolerance: 45,
        days: [],
      };
      active = "temp";
      phase = "day1";
      el.root.classList.add("has-image");
      el.importBtn.disabled = true;
      setPrompt();
      setStatus("");
      refreshTabs();
      drawOverlay();
    };
    img.src = url;
  });

  function toSourcePx(ev: MouseEvent): { x: number; y: number } {
    const rect = el.canvas.getBoundingClientRect();
    const scale = el.canvas.width / rect.width;
    return {
      x: Math.round((ev.clientX - rect.left) * scale),
      y: Math.round((ev.clientY - rect.top) * scale),
    };
  }

  function askNumber(message: string, fallback: number): number | null {
    const raw = window.prompt(message, String(fallback));
    if (raw == null) return null;
    const v = Number(raw.replace(",", "."));
    return Number.isFinite(v) ? v : null;
  }

  el.overlay.addEventListener("click", (ev) => {
    if (!imgData || phase === "idle") return;
    const { x, y } = toSourcePx(ev);
    const s = cur();
    const m = SERIES_META[active];

    if (phase === "day1") {
      const v = askNumber("Numéro de ce jour (ex. 7) :", 1);
      if (v == null) return;
      dayRefs[0] = { pixel: x, value: v };
      phase = "day2";
    } else if (phase === "day2") {
      const v = askNumber(
        "Numéro de ce jour (plus à droite, ex. 24) :",
        Math.min(CYCLE_MAX_DAYS, dayRefs[0].value + 15),
      );
      if (v == null) return;
      dayRefs[1] = { pixel: x, value: v };
      phase = "value1";
    } else if (phase === "value1") {
      const v = askNumber(
        `${m.label} — valeur de cette graduation (${m.unit}, ex. ${m.ex1}) :`,
        m.ex1,
      );
      if (v == null) return;
      s.valueRefs[0] = { pixel: y, value: v };
      phase = "value2";
    } else if (phase === "value2") {
      const v = askNumber(
        `${m.label} — valeur de l'autre graduation (${m.unit}, ex. ${m.ex2}) :`,
        m.ex2,
      );
      if (v == null) return;
      s.valueRefs[1] = { pixel: y, value: v };
      phase = "color";
      // auto-suggest a colour as a starting point
      const region = guessPlotRegion(
        imgData.data,
        imgData.width,
        imgData.height,
      );
      s.color = suggestCurveColor(imgData.data, imgData.width, region);
    } else if (phase === "color") {
      s.color = pixelAt(imgData.data, imgData.width, x, y);
      phase = "ready";
    } else if (phase === "ready") {
      // re-pick colour on click
      s.color = pixelAt(imgData.data, imgData.width, x, y);
    }

    runExtraction();
    setPrompt();
    refreshTabs();
    updateImportEnabled();
    drawOverlay();
  });

  // ── magnifier loupe ──
  el.overlay.addEventListener("mousemove", (ev) => {
    if (!imgData) return;
    const { x, y } = toSourcePx(ev);
    drawLoupe(ev, x, y);
  });
  el.overlay.addEventListener("mouseleave", () => {
    el.loupe.style.display = "none";
  });

  function drawLoupe(ev: MouseEvent, sx: number, sy: number) {
    const Z = 6; // zoom factor
    const R = 52; // loupe radius (display px)
    const src = R / Z;
    el.loupe.width = R * 2;
    el.loupe.height = R * 2;
    lctx.imageSmoothingEnabled = false;
    lctx.clearRect(0, 0, R * 2, R * 2);
    lctx.drawImage(
      el.canvas,
      sx - src,
      sy - src,
      src * 2,
      src * 2,
      0,
      0,
      R * 2,
      R * 2,
    );
    // crosshair
    lctx.strokeStyle = "rgba(126,75,158,0.9)";
    lctx.lineWidth = 1;
    lctx.beginPath();
    lctx.moveTo(R, 0);
    lctx.lineTo(R, R * 2);
    lctx.moveTo(0, R);
    lctx.lineTo(R * 2, R);
    lctx.stroke();
    const wrapRect = el.canvas.getBoundingClientRect();
    el.loupe.style.display = "block";
    el.loupe.style.left = `${ev.clientX - wrapRect.left + 14}px`;
    el.loupe.style.top = `${ev.clientY - wrapRect.top + 14}px`;
  }

  el.toleranceInput.addEventListener("input", () => {
    cur().tolerance = Number(el.toleranceInput.value);
    el.toleranceLabel.textContent = String(cur().tolerance);
    runExtraction();
    drawOverlay();
  });

  el.redoBtn.addEventListener("click", () => {
    if (!imgData) return;
    // recalibrate only the active series (keep the shared day axis)
    cur().valueRefs = [];
    cur().color = null;
    cur().days = [];
    phase = phaseForActive();
    setStatus("");
    refreshTabs();
    updateImportEnabled();
    setPrompt();
    drawOverlay();
  });

  function runExtraction() {
    const s = cur();
    const cal = calibrationFor(s);
    if (!imgData || !cal || !s.color || phase !== "ready") {
      s.days = [];
      return;
    }
    const region = {
      x0: Math.min(dayRefs[0].pixel, dayRefs[1].pixel) - 5,
      x1: Math.max(dayRefs[0].pixel, dayRefs[1].pixel) + 5,
      y0: 0,
      y1: imgData.height - 1,
    };
    const points = extractCurvePixels({
      data: imgData.data,
      width: imgData.width,
      height: imgData.height,
      target: s.color,
      tolerance: s.tolerance,
      region,
    });
    s.days = resampleToDays(points, cal, CYCLE_MAX_DAYS, rounderFor(s.kind));
    const found = s.days.filter((d) => d.value != null).length;
    setStatus(
      found > 0
        ? `${found} jours détectés pour « ${SERIES_META[s.kind].label} ».`
        : "Aucun point détecté — cliquez précisément sur la courbe (utilisez la loupe) ou augmentez la tolérance.",
    );
  }

  function updateImportEnabled() {
    const any =
      series.temp.days.some((d) => d.value != null) ||
      series.lh.days.some((d) => d.value != null);
    el.importBtn.disabled = !any;
  }

  function drawOverlay() {
    octx.clearRect(0, 0, el.overlay.width, el.overlay.height);

    // day reference lines (shared)
    octx.lineWidth = 1.5;
    octx.strokeStyle = "rgba(126,75,158,0.9)";
    octx.fillStyle = "rgba(126,75,158,0.95)";
    octx.font = "12px sans-serif";
    for (const r of dayRefs) {
      octx.beginPath();
      octx.moveTo(r.pixel, 0);
      octx.lineTo(r.pixel, el.overlay.height);
      octx.stroke();
      octx.fillText(`J${r.value}`, r.pixel + 3, 14);
    }

    // value reference lines for the ACTIVE series
    const s = cur();
    const axisColor = active === "temp" ? "rgba(86,120,214,0.95)" : "rgba(214,68,127,0.95)";
    octx.strokeStyle = axisColor;
    octx.fillStyle = axisColor;
    for (const r of s.valueRefs) {
      octx.beginPath();
      octx.moveTo(0, r.pixel);
      octx.lineTo(el.overlay.width, r.pixel);
      octx.stroke();
      octx.fillText(`${r.value}${SERIES_META[active].unit}`, 3, r.pixel - 3);
    }

    // extracted points for BOTH series (so progress is visible)
    for (const kind of ["temp", "lh"] as SeriesKind[]) {
      const ss = series[kind];
      const cal = calibrationFor(ss);
      if (!cal || !ss.days.length) continue;
      octx.fillStyle =
        kind === "temp" ? "rgba(86,120,214,0.95)" : "rgba(214,68,127,0.95)";
      for (const dv of ss.days) {
        if (dv.value == null) continue;
        const px = unmapAxis(cal.dayAxis, dv.day);
        const py = unmapAxis(cal.valueAxis, dv.value);
        octx.beginPath();
        octx.arc(px, py, 3, 0, Math.PI * 2);
        octx.fill();
      }
    }

    // colour swatch for active series
    if (s.color) {
      octx.fillStyle = `rgb(${s.color.r},${s.color.g},${s.color.b})`;
      octx.fillRect(el.overlay.width - 26, 6, 20, 20);
      octx.strokeStyle = "#fff";
      octx.lineWidth = 2;
      octx.strokeRect(el.overlay.width - 26, 6, 20, 20);
    }
  }

  el.importBtn.addEventListener("click", () => {
    const temps = series.temp.days.length
      ? series.temp.days.map((d) => d.value)
      : null;
    const lh = series.lh.days.length
      ? series.lh.days.map((d) => d.value)
      : null;
    if (!temps && !lh) return;
    el.root.dispatchEvent(
      new CustomEvent("fertiluna:digitized", {
        bubbles: true,
        detail: { temps, lh },
      }),
    );
  });

  setPrompt();
}
