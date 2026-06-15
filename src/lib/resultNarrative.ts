/**
 * resultNarrative.ts — turns a CycleAnalysis into the rich, scrollable result
 * content that keeps users on the page (and creates natural slots for inline
 * ads between sections).
 *
 * Produces (in the requested locale):
 *   - the verdict (title/tone/summary from LABEL_COPY)
 *   - a phase-by-phase narrative ("Jour 1-13 : phase folliculaire stable…")
 *   - derived stat chips
 *   - prioritised doctor questions
 *   - glossary terms referenced in the copy
 */

import { LABEL_COPY, LABEL_COPY_EN, type CycleLabel } from "./constants";
import type { CycleAnalysis } from "./inference";
import { DEFAULT_LOCALE, type Locale } from "./i18n";

export interface StatChip {
  label: string;
  value: string;
  hint?: string;
}

export interface NarrativeSection {
  heading: string;
  body: string;
}

export interface GlossaryTerm {
  term: string;
  definition: string;
}

export interface ResultContent {
  title: string;
  tone: "positive" | "neutral" | "caution" | "info";
  summary: string;
  confidencePct: number;
  confident: boolean;
  chips: StatChip[];
  narrative: NarrativeSection[];
  doctorQuestions: string[];
  oodNote: string | null;
  glossary: GlossaryTerm[];
}

/** Locale-specific copy + builders. FR mirrors the original strings verbatim. */
interface NarrativeStrings {
  labels: Record<CycleLabel, { title: string; tone: ResultContent["tone"]; summary: string; doctorQuestion?: string }>;
  glossary: Record<string, string>;
  fmtAmpl: (a: number) => string;
  day: (n: number) => string;
  dayUnit: string;
  chipOvulation: string;
  chipOvulationHint: string;
  chipFollicular: string;
  chipLuteal: string;
  chipLutealHint: string;
  chipRiseAmpl: string;
  chipLhPeak: string;
  chipLhPeakMulti: (n: number) => string;
  follicularHeading: (ov: number) => string;
  follicularBody: (len: number) => string;
  ovulationHeading: (ov: number) => string;
  ovulationBodyClear: (ampl: string) => string;
  ovulationBodyFaint: (present: boolean) => string;
  lutealHeading: (from: number) => string;
  lutealBody: (good: boolean, len: number) => string;
  noRiseHeading: string;
  noRiseBody: string;
  shortLutealDoctorQ: string;
  oodNote: string;
}

const FR: NarrativeStrings = {
  labels: LABEL_COPY,
  glossary: {
    "phase folliculaire":
      "Première partie du cycle, du premier jour des règles jusqu'à l'ovulation. La température y est basse et relativement stable.",
    "phase lutéale":
      "Seconde partie du cycle, de l'ovulation jusqu'aux règles suivantes. Le corps jaune sécrète de la progestérone, ce qui élève la température.",
    ovulation:
      "Libération de l'ovule par l'ovaire. Sur une courbe de température, elle est suivie d'une montée thermique de 0,2 à 0,5 °C.",
    "montée thermique":
      "Élévation durable de la température basale (≥ 3 jours) qui suit l'ovulation, due à la progestérone.",
    "pic LH":
      "Pic de l'hormone lutéinisante, détecté par les tests d'ovulation. Il précède l'ovulation de 12 à 36 heures.",
    progestérone:
      "Hormone produite après l'ovulation. Un dosage en phase lutéale (vers J21) permet de confirmer l'ovulation.",
  },
  fmtAmpl: (a) => `${a.toFixed(2).replace(".", ",")} °C`,
  day: (n) => `J${n}`,
  dayUnit: "j",
  chipOvulation: "Ovulation estimée",
  chipOvulationHint: "Dernier jour avant la montée thermique (règle 3 sur 6).",
  chipFollicular: "Phase folliculaire",
  chipLuteal: "Phase lutéale",
  chipLutealHint: "≥ 10 jours est considéré comme une bonne durée.",
  chipRiseAmpl: "Amplitude de montée",
  chipLhPeak: "Pic LH",
  chipLhPeakMulti: (n) => `${n} pics détectés`,
  follicularHeading: (ov) => `Jours 1 à ${ov} : phase folliculaire`,
  follicularBody: (len) =>
    `Avant l'ovulation, votre température reste dans sa zone basse. C'est normal : cette première moitié du cycle, la phase folliculaire, dure ${len} jours sur ce cycle. Une phase folliculaire qui varie d'un cycle à l'autre est courante et n'est pas inquiétante en soi.`,
  ovulationHeading: (ov) => `Autour de J${ov} : l'ovulation`,
  ovulationBodyClear: (ampl) =>
    `On observe une montée thermique nette de ${ampl}. C'est le signe que l'ovulation a probablement eu lieu : le corps jaune produit de la progestérone, qui élève la température.`,
  ovulationBodyFaint: (present) =>
    `La transition vers la phase lutéale est ${present ? "présente mais discrète" : "difficile à identifier"}. Une montée peu marquée mérite d'être confirmée par un dosage hormonal.`,
  lutealHeading: (from) => `Jours ${from} à fin : phase lutéale`,
  lutealBody: (good, len) =>
    `Après l'ovulation, la température reste en plateau ${good ? "sur une bonne durée" : "sur une durée courte"} (${len} jours observés). Une phase lutéale d'au moins 10 jours laisse le temps à une éventuelle implantation.`,
  noRiseHeading: "Lecture de la courbe",
  noRiseBody:
    "Aucune montée thermique claire n'a pu être isolée sur ce cycle. Cela peut venir d'un cycle sans ovulation, de mesures trop espacées, ou de relevés pris à des heures variables. Mesurer chaque matin à la même heure, avant de se lever, améliore beaucoup la lisibilité.",
  shortLutealDoctorQ:
    "Votre phase lutéale est un peu juste : un dosage de progestérone en phase lutéale peut être utile.",
  oodNote:
    "Votre courbe est assez atypique par rapport aux profils habituels. L'analyse reste indicative : n'hésitez pas à en discuter avec un professionnel de santé.",
};

const EN: NarrativeStrings = {
  labels: LABEL_COPY_EN,
  glossary: {
    "follicular phase":
      "The first part of the cycle, from the first day of your period to ovulation. The temperature is low and relatively stable.",
    "luteal phase":
      "The second part of the cycle, from ovulation to the next period. The corpus luteum secretes progesterone, which raises the temperature.",
    ovulation:
      "Release of the egg by the ovary. On a temperature chart it is followed by a thermal rise of 0.2 to 0.5 °C.",
    "thermal rise":
      "A sustained rise in basal temperature (≥ 3 days) following ovulation, caused by progesterone.",
    "lh peak":
      "Peak of luteinizing hormone, detected by ovulation tests. It precedes ovulation by 12 to 36 hours.",
    progesterone:
      "A hormone produced after ovulation. A luteal-phase test (around day 21) confirms ovulation.",
  },
  fmtAmpl: (a) => `${a.toFixed(2)} °C`,
  day: (n) => `D${n}`,
  dayUnit: "d",
  chipOvulation: "Estimated ovulation",
  chipOvulationHint: "Last day before the thermal rise (3-over-6 rule).",
  chipFollicular: "Follicular phase",
  chipLuteal: "Luteal phase",
  chipLutealHint: "≥ 10 days is considered a good length.",
  chipRiseAmpl: "Rise amplitude",
  chipLhPeak: "LH peak",
  chipLhPeakMulti: (n) => `${n} peaks detected`,
  follicularHeading: (ov) => `Days 1 to ${ov}: follicular phase`,
  follicularBody: (len) =>
    `Before ovulation, your temperature stays in its low range. That is normal: this first half of the cycle, the follicular phase, lasts ${len} days this cycle. A follicular phase that varies from one cycle to the next is common and not concerning in itself.`,
  ovulationHeading: (ov) => `Around D${ov}: ovulation`,
  ovulationBodyClear: (ampl) =>
    `A clear thermal rise of ${ampl} is visible. This suggests ovulation has probably occurred: the corpus luteum produces progesterone, which raises the temperature.`,
  ovulationBodyFaint: (present) =>
    `The transition to the luteal phase is ${present ? "present but subtle" : "hard to identify"}. A faint rise is worth confirming with a hormone test.`,
  lutealHeading: (from) => `Days ${from} to end: luteal phase`,
  lutealBody: (good, len) =>
    `After ovulation, the temperature stays on a plateau ${good ? "for a good duration" : "for a short duration"} (${len} days observed). A luteal phase of at least 10 days allows time for a possible implantation.`,
  noRiseHeading: "Reading the chart",
  noRiseBody:
    "No clear thermal rise could be isolated in this cycle. This may be due to a cycle without ovulation, measurements that are too sparse, or readings taken at variable times. Measuring every morning at the same time, before getting up, greatly improves readability.",
  shortLutealDoctorQ:
    "Your luteal phase is a little short: a luteal-phase progesterone test may be useful.",
  oodNote:
    "Your curve is fairly atypical compared to usual profiles. The analysis remains indicative: do not hesitate to discuss it with a healthcare professional.",
};

const STRINGS: Record<Locale, NarrativeStrings> = { fr: FR, en: EN };

export function buildResultContent(
  analysis: CycleAnalysis,
  locale: Locale = DEFAULT_LOCALE,
): ResultContent {
  const L = STRINGS[locale] ?? FR;
  const copy = L.labels[analysis.label];
  const d = analysis.derived;
  const confidencePct = Math.round(analysis.confidence * 100);

  // ── stat chips ──
  const chips: StatChip[] = [];
  if (d.estimatedOvulationDay) {
    chips.push({
      label: L.chipOvulation,
      value: L.day(d.estimatedOvulationDay),
      hint: L.chipOvulationHint,
    });
  }
  if (d.follicularLength) {
    chips.push({
      label: L.chipFollicular,
      value: `${d.follicularLength} ${L.dayUnit}`,
    });
  }
  if (d.lutealLength) {
    chips.push({
      label: L.chipLuteal,
      value: `${d.lutealLength} ${L.dayUnit}`,
      hint: L.chipLutealHint,
    });
  }
  if (d.thermalRiseAmplitude > 0) {
    chips.push({
      label: L.chipRiseAmpl,
      value: L.fmtAmpl(d.thermalRiseAmplitude),
    });
  }
  if (d.lhPeakDay) {
    chips.push({
      label: L.chipLhPeak,
      value: L.day(d.lhPeakDay),
      hint: d.lhPeakCount > 1 ? L.chipLhPeakMulti(d.lhPeakCount) : undefined,
    });
  }

  // ── narrative sections ──
  const narrative: NarrativeSection[] = [];
  if (d.estimatedOvulationDay) {
    narrative.push({
      heading: L.follicularHeading(d.estimatedOvulationDay),
      body: L.follicularBody(d.follicularLength ?? d.estimatedOvulationDay),
    });
    narrative.push({
      heading: L.ovulationHeading(d.estimatedOvulationDay),
      body:
        d.thermalRiseAmplitude > 0.2
          ? L.ovulationBodyClear(L.fmtAmpl(d.thermalRiseAmplitude))
          : L.ovulationBodyFaint(d.thermalRiseAmplitude > 0),
    });
    if (d.lutealLength) {
      narrative.push({
        heading: L.lutealHeading(d.estimatedOvulationDay + 1),
        body: L.lutealBody(d.lutealLength >= 10, d.lutealLength),
      });
    }
  } else {
    narrative.push({ heading: L.noRiseHeading, body: L.noRiseBody });
  }

  // ── doctor questions ──
  const doctorQuestions: string[] = [];
  if (copy.doctorQuestion) doctorQuestions.push(copy.doctorQuestion);
  if (analysis.label === "ovulation_confirmee" && d.lutealLength && d.lutealLength < 11) {
    doctorQuestions.push(L.shortLutealDoctorQ);
  }

  // ── OOD note ──
  let oodNote: string | null = null;
  if (analysis.oodPercentile >= 80) {
    oodNote = L.oodNote;
  }

  // ── glossary (only terms used) ──
  const usedTerms = new Set<string>();
  const haystack = (
    copy.summary +
    " " +
    narrative.map((n) => n.body).join(" ")
  ).toLowerCase();
  for (const term of Object.keys(L.glossary)) {
    if (haystack.includes(term)) usedTerms.add(term);
  }
  const glossary: GlossaryTerm[] = [...usedTerms].map((term) => ({
    term,
    definition: L.glossary[term],
  }));

  return {
    title: copy.title,
    tone: copy.tone,
    summary: copy.summary,
    confidencePct,
    confident: analysis.confident,
    chips,
    narrative,
    doctorQuestions,
    oodNote,
    glossary,
  };
}

export function toneClass(tone: ResultContent["tone"]): string {
  return `tone-${tone}`;
}

export type { CycleLabel };
