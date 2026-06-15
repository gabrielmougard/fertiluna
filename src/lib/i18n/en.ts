/** English catalog. Keys MUST match fr.ts (type-checked via Catalog). */
import type { Catalog } from "./index";

export const en: Catalog = {
  "consent.title": "AI-enhanced analysis",
  "consent.body":
    "For a more accurate reading, your image can be analyzed by an external "
    + "AI service. In that case the image leaves your device for "
    + "the duration of the analysis. Without your consent, analysis stays "
    + "100 % on your device.",
  "consent.accept": "Allow enhanced AI analysis",
  "consent.decline": "Keep it 100 % on my device",
  "consent.medical":
    "FertiLuna is not a medical device. Results are indicative and do "
    + "not replace the advice of a healthcare professional.",
  "status.extracted": "Analysis succeeded. Check the preview before importing.",
  "status.low_confidence":
    "Uncertain result: check each value carefully, or use manual calibration.",
  "status.not_a_chart":
    "No cycle chart detected. Try another image or manual entry.",
  "backend.on_device": "On-device analysis",
  "backend.cloud": "AI analysis (cloud)",

  // ── shared chrome (header / footer) ──
  "brand.domain": "FertiLuna",
  "nav.tool": "Analyze my chart",
  "nav.tools": "Tools",
  "nav.about": "About",
  "footer.medical":
    "<strong>FertiLuna</strong> is not a medical device. Results are "
    + "indicative and do not replace the advice of a healthcare professional.",
  "footer.privacy":
    "No data is collected or sent to a server. The entire analysis runs in "
    + "your browser.",
  "footer.sourceLabel": "Source code",
  "footer.sourceLead": "Open source project:",

  // ── tool page (outils/analyse-courbe) ──
  "tool.metaTitle": "Temperature chart analysis · FertiLuna",
  "tool.metaDescription":
    "Enter your basal temperatures and LH tests day by day. FertiLuna "
    + "detects ovulation, assesses your follicular and luteal phases and "
    + "interprets your cycle, no sign-up, entirely in your browser.",
  "tool.eyebrow": "Tool 1 · Chart analysis",
  "tool.h1": "Analyze your chart",
  "tool.intro":
    "Enter your basal temperatures (taken on waking, before getting up) and, "
    + "if you have them, your ovulation (LH) tests. Everything runs "
    + "<strong>entirely in your browser</strong>: no data is sent to a server.",
  "tool.useTemp": "Temperature",
  "tool.useLh": "LH tests",
  "tool.colDay": "Day",
  "tool.colTemp": "Temp. (°C)",
  "tool.colLh": "LH (0–3)",
  "tool.analyze": "Analyze my cycle",
  "tool.analyzing": "Analyzing…",
  "tool.demo": "Fill an example",
  "tool.clear": "Clear",
  "tool.needData": "Enter at least a few days of data to run the analysis.",
  "tool.error": "Something went wrong during the analysis. Please try again.",
  "tool.modelReady":
    "Model ready and cached ({mb} MB). Your next analyses will be instant and "
    + "offline.",
  "tool.modelLoading": "Loading the model… {pct}%",
  "tool.modelPreparing": "Preparing the model…",
  "tool.imported":
    "{n} values imported from the image. Check them, then run the analysis.",
  "tool.doctorTitle": "To ask your doctor",
  "tool.glossaryTitle": "Glossary: understand the terms",
  "tool.legendFollicular": "Follicular phase",
  "tool.legendLuteal": "Luteal phase",
  "tool.legendLhPeak": "LH peak",
  "tool.confidence": "Confidence {pct}%",
  "tool.confidenceLow": "Limited confidence ({pct}%)",
  "tool.resultDisclaimer":
    "This analysis is indicative. If in doubt or struggling to conceive, talk "
    + "to your doctor or midwife.",

  // ── image import / digitizer (CurveDigitizer + controllers) ──
  "dz.sectionTitle": "Import a screenshot",
  "dz.sectionDesc":
    "A chart in another app (Inito, Premom, Clearblue, Flo…) or on paper? Drop "
    + "a screenshot and FertiLuna reads the curves for you. "
    + "<strong>The image stays on your device.</strong>",
  "dz.dropTitle": "Drop your screenshot here",
  "dz.dropSub": "or click to choose an image",
  "dz.dropFormats": "PNG, JPG · the image never leaves your device",
  "dz.tempLabel": "Temperature (BBT)",
  "dz.lhLabel": "LH / hormone",
  "dz.import": "Import into the table",
  "dz.reset": "Change image",
  "dz.advanced": "Manual adjustment",
  "dz.advancedHint": "(if the automatic reading gets it wrong)",
  "dz.manualUpload": "Choose an image",
  "dz.seriesLabel": "Series:",
  "dz.step1":
    "Calibrate the <strong>day</strong> axis (2 clicks on the ZT/day line).",
  "dz.step2":
    "Calibrate the <strong>value</strong> axis of the chosen series. Note that "
    + "BBT and LH are often on <strong>opposite</strong> axes.",
  "dz.step3":
    "Click the series' <strong>curve</strong> to pick up its color.",
  "dz.step4": "Switch series for the second curve, or import.",
  "dz.tol": "Color tolerance",
  "dz.redo": "Recalibrate this series",
  // auto-extract status / loader
  "dz.analyzing": "Analyzing the image…",
  "dz.analyzingCloud": "Enhanced AI analysis…",
  "dz.analyzingDevice": "On-device analysis…",
  "dz.failed": "Automatic reading failed on this image.",
  "dz.unreadable": "Unreadable image.",
  "dz.unavailable": "Automatic reading could not be prepared.",
  "dz.tryManual": " Try the manual adjustment below.",
  "dz.noCurve": "No cycle chart detected in this image.",
  "dz.noCurveStatus":
    "No usable chart detected. Make sure the image is a cycle chart, or use "
    + "manual adjustment.",
  "dz.noNet":
    "No clear curve detected. Try a higher-contrast image or manual "
    + "calibration.",
  "dz.scaleDetected": "Detected scale: {unit} ({min}–{max})",
  "dz.detectedTemp": "temperature in {unit} ({n} d)",
  "dz.detectedLh": "LH ({n} d)",
  "dz.detected": "Detected: {parts}.",
  "dz.verifyImport":
    "{detected} Check the preview, then import. You can fix the values in the "
    + "table if needed.",
  "dz.lowConfidence":
    "Limited confidence ({pct} %) — check each point carefully in the table, "
    + "or use manual adjustment. {detected}",
  // overlay hover tooltips
  "dz.tipDay": "day {day}",
  "dz.tipPlot": "Plot area ({method})",
  "dz.tipAxis": "Axis {kind} · {v}",
  "dz.tipRow": "Row « {name} »",
  "dz.tipCellEmpty": "{name} (empty)",
  // manual digitizer (prompts)
  "dzm.idle": "Import a screenshot of your chart to begin.",
  "dzm.day1":
    "Day axis (1/2): click a known day marker (ZT/day line), then enter its "
    + "number.",
  "dzm.day2": "Day axis (2/2): click another day marker, further to the right.",
  "dzm.value1":
    "« {label} », value (1/2): click a tick on its axis ({side}), then enter "
    + "its value.",
  "dzm.value2": "« {label} », value (2/2): click another tick on the same axis.",
  "dzm.color": "« {label} »: click directly on its curve to pick up its color.",
  "dzm.ready":
    "« {label} » detected. Check the preview, adjust the tolerance, or switch "
    + "to the other series / import.",
  "dzm.askValue1": "{label}, value of this tick ({unit}, e.g. {ex}):",
  "dzm.askValue2": "{label}, value of the other tick ({unit}, e.g. {ex}):",
  "dzm.noPoint":
    "No point detected: click precisely on the curve (use the loupe) or raise "
    + "the tolerance.",
  "dzm.daysDetected": "{n} days detected for « {label} ».",
  "dzm.sideTemp": "often the RIGHT axis (°C)",
  "dzm.sideLh": "often the LEFT axis",
  "dzm.askDay1": "Number of this day (e.g. 7):",
  "dzm.askDay2": "Number of this day (further right, e.g. 24):",

  // ── run history (on-device, IndexedDB) ──
  "history.title": "My analyses",
  "history.subtitle":
    "Your past analyses are saved on this device only. Nothing is sent online.",
  "history.empty":
    "No saved analyses yet. Your next analyses will show up here.",
  "history.open": "Reopen",
  "history.delete": "Delete",
  "history.clearAll": "Clear all",
  "history.clearConfirm": "Clear all analyses saved on this device?",
  "history.saved": "Analysis saved on this device.",
  "history.restored": "Analysis reloaded. Re-run to recompute if needed.",
  "history.count": "{n} saved",
  "history.withImage": "with image",
};
