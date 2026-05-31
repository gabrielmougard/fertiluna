/**
 * autoExtractUI.ts — controller for the "Extraction auto (IA)" mode.
 *
 * Flow (zero numeric input from the user):
 *   1. user uploads a chart screenshot,
 *   2. the chart-vision model predicts the normalized BBT/LH curve shapes AND
 *      auto-detects the BBT axis unit (°C vs °F) and ranges,
 *   3. a live preview overlays the recovered curves on the image,
 *   4. "Importer" emits `fertiluna:digitized` with de-normalized { temps, lh }.
 *
 * Everything is on-device. If the model is unavailable or fails, the host
 * component falls back to the manual calibration digitizer.
 */

import { CYCLE_MAX_DAYS } from "./constants";
import {
  predictChart,
  denormalizeSeries,
  ensureVisionModelLoaded,
  type VisionPrediction,
  type VisionSeriesKind,
  type VisionLoadProgress,
} from "./visionInference";

export interface AutoExtractElements {
  root: HTMLElement;
  fileInput: HTMLInputElement;
  canvas: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
  status: HTMLElement;
  /** optional per-series include toggles (auto-checked from detection) */
  tempEnable?: HTMLInputElement;
  lhEnable?: HTMLInputElement;
  /** shows the auto-detected scale, e.g. "Échelle : °C (35.6–37.4)" */
  scaleInfo?: HTMLElement;
  importBtn: HTMLButtonElement;
  /** called when the model can't be used, so the host can offer manual mode. */
  onUnavailable?: (reason: string) => void;
}

export function initAutoExtract(el: AutoExtractElements): void {
  const ctx = el.canvas.getContext("2d", { willReadFrequently: true })!;
  const octx = el.overlay.getContext("2d")!;

  let prediction: VisionPrediction | null = null;

  function setStatus(s: string) {
    el.status.textContent = s;
  }

  function reportLoad(p: VisionLoadProgress) {
    if (p.phase === "ready") return;
    if (p.file && p.file.total > 0) {
      const pct = Math.round((p.file.loaded / p.file.total) * 100);
      setStatus(`Chargement du modèle IA… ${pct}%`);
    } else {
      setStatus("Préparation du modèle IA…");
    }
  }

  // Warm the model in the background.
  (window.requestIdleCallback ?? ((cb: () => void) => setTimeout(cb, 1200)))(
    () => {
      ensureVisionModelLoaded(reportLoad).then(
        () => setStatus(""),
        (e) => {
          console.error(e);
          el.onUnavailable?.("Le modèle IA n'a pas pu être chargé.");
        },
      );
    },
  );

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
      URL.revokeObjectURL(url);
      el.root.classList.add("has-image");

      setStatus("Analyse de l'image par le modèle IA…");
      try {
        prediction = await predictChart(im, reportLoad);
      } catch (e) {
        console.error(e);
        el.onUnavailable?.("L'analyse IA a échoué sur cette image.");
        return;
      }
      applyDetection();
      drawOverlay();
      updateImportEnabled();
    };
    im.onerror = () => setStatus("Image illisible.");
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
        ? `Échelle détectée : ${unit} (${s.min}–${s.max})`
        : "";
    }

    const parts: string[] = [];
    if (tHas) {
      const unit = prediction.bbtScale.label === "fahrenheit" ? "°F" : "°C";
      parts.push(`température en ${unit} (${prediction.temp.presentCount} j)`);
    }
    if (lHas) parts.push(`LH (${prediction.lh.presentCount} j)`);
    setStatus(
      parts.length
        ? `Détecté : ${parts.join(" + ")}. Vérifiez l'aperçu, puis importez. Vous corrigerez les valeurs dans le tableau si besoin.`
        : "Aucune courbe nette détectée. Essayez une image plus contrastée ou la calibration manuelle.",
    );
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

  function drawOverlay() {
    octx.clearRect(0, 0, el.overlay.width, el.overlay.height);
    if (!prediction) return;
    const W = el.overlay.width;
    const H = el.overlay.height;
    const dayX = (d: number) => ((d + 0.5) / CYCLE_MAX_DAYS) * W;
    const valY = (n: number) => (1 - n) * H;

    const series: [VisionSeriesKind, string, boolean][] = [
      ["temp", "rgba(86,120,214,0.95)", tempIncluded()],
      ["lh", "rgba(214,68,127,0.95)", lhIncluded()],
    ];
    for (const [kind, color, enabled] of series) {
      if (!enabled) continue;
      const p = prediction[kind];
      octx.fillStyle = color;
      octx.strokeStyle = color;
      octx.lineWidth = 2;
      let started = false;
      octx.beginPath();
      for (let d = 0; d < CYCLE_MAX_DAYS; d++) {
        const n = p.normalized[d];
        if (n == null) {
          started = false;
          continue;
        }
        const x = dayX(d);
        const y = valY(n);
        if (!started) {
          octx.moveTo(x, y);
          started = true;
        } else {
          octx.lineTo(x, y);
        }
      }
      octx.stroke();
      for (let d = 0; d < CYCLE_MAX_DAYS; d++) {
        const n = p.normalized[d];
        if (n == null) continue;
        octx.beginPath();
        octx.arc(dayX(d), valY(n), 3, 0, Math.PI * 2);
        octx.fill();
      }
    }
  }

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
      drawOverlay();
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
