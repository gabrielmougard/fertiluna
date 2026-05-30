/**
 * Shared constants — the TypeScript mirror of model/fertiluna/constants.py.
 *
 * These MUST stay in sync with the Python side. The parity test
 * (src/lib/features.test.ts) loads the fixtures emitted by the Python test
 * suite and asserts that the TS feature extractor produces identical output.
 */

export const CYCLE_MAX_DAYS = 35;

export const LABELS = [
  "ovulation_confirmee",
  "ovulation_douteuse",
  "anovulation",
  "phase_luteale_courte",
  "donnees_insuffisantes",
] as const;

export type CycleLabel = (typeof LABELS)[number];

export const CONFIDENCE_THRESHOLD = 0.6;

export const FEATURE_NAMES = [
  "n_temp_observed",
  "n_lh_observed",
  "missing_rate_temp",
  "missing_rate_lh",
  "temp_mean_overall",
  "temp_std_overall",
  "temp_min",
  "temp_max",
  "temp_range",
  "estimated_ovulation_day",
  "follicular_mean",
  "follicular_std",
  "luteal_mean",
  "luteal_std",
  "thermal_rise_amplitude",
  "rise_steepness_max",
  "plateau_days_above_baseline",
  "n_consecutive_high_days",
  "post_rise_dip_count",
  "lh_peak_day",
  "lh_peak_value",
  "lh_peak_count",
  "days_lh_peak_to_thermal_rise",
  "follicular_length",
  "luteal_length",
  "spline_slope_pre_ovulation",
  "spline_slope_post_ovulation",
  "fraction_days_below_mean",
  "longest_run_above_mean",
  "longest_run_below_mean",
] as const;

export const N_FEATURES = FEATURE_NAMES.length;

/** Model versioning — bump when re-exporting the ONNX artifacts. */
export const MODEL_VERSION = "v1";

/**
 * Human-facing copy for each label. Mirrors the "Exemples de résultats produits"
 * table in the roadmap (Outil 1). `title` is the short verdict; `summary` is the
 * pedagogical 1-2 sentence explanation; `doctorQuestion` is the optional "question
 * à poser à votre médecin".
 */
export interface LabelCopy {
  title: string;
  tone: "positive" | "neutral" | "caution" | "info";
  summary: string;
  doctorQuestion?: string;
}

export const LABEL_COPY: Record<CycleLabel, LabelCopy> = {
  ovulation_confirmee: {
    title: "Ovulation confirmée",
    tone: "positive",
    summary:
      "La montée thermique est nette et stable sur au moins 3 jours, et votre phase lutéale est de bonne durée. C'est le signe d'un cycle ovulatoire de bonne qualité.",
  },
  ovulation_douteuse: {
    title: "Montée thermique insuffisante",
    tone: "caution",
    summary:
      "Une élévation de température a été détectée mais elle ne remplit pas les critères de confirmation sur 3 jours. Cela peut indiquer une ovulation tardive ou de faible qualité.",
    doctorQuestion:
      "Demandez un dosage de progestérone en milieu de phase lutéale (J21) pour confirmer la qualité de l'ovulation.",
  },
  anovulation: {
    title: "Anovulation suspectée",
    tone: "caution",
    summary:
      "Aucune ovulation n'a pu être identifiée sur ce cycle. La courbe reste plate ou trop irrégulière. Il peut s'agir d'un cycle exceptionnel — mais si cela se répète, parlez-en à votre médecin.",
    doctorQuestion:
      "Si l'absence d'ovulation se répète, demandez un bilan hormonal (FSH, LH, estradiol) en début de cycle.",
  },
  phase_luteale_courte: {
    title: "Phase lutéale courte",
    tone: "caution",
    summary:
      "Votre ovulation est confirmée, mais votre phase lutéale semble inférieure à 10 jours. Une phase lutéale courte peut rendre l'implantation plus difficile.",
    doctorQuestion:
      "Demandez à votre médecin un dosage de progestérone en phase lutéale, et discutez d'un éventuel soutien.",
  },
  donnees_insuffisantes: {
    title: "Données insuffisantes",
    tone: "info",
    summary:
      "Il n'y a pas assez de mesures fiables sur ce cycle pour produire une analyse robuste. Continuez à relever votre température chaque matin au réveil, à la même heure, pendant un cycle complet.",
  },
};
