/**
 * resultNarrative.ts — turns a CycleAnalysis into the rich, scrollable result
 * content that keeps users on the page (and creates natural slots for inline
 * ads between sections).
 *
 * Produces:
 *   - the verdict (title/tone/summary from LABEL_COPY)
 *   - a phase-by-phase narrative ("Jour 1-13 : phase folliculaire stable…")
 *   - derived stat chips
 *   - prioritised doctor questions
 *   - glossary terms referenced in the copy
 */

import { LABEL_COPY, type CycleLabel } from "./constants";
import type { CycleAnalysis } from "./inference";

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

const GLOSSARY: Record<string, string> = {
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
};

function fmtAmpl(a: number): string {
  return `${a.toFixed(2).replace(".", ",")} °C`;
}

export function buildResultContent(analysis: CycleAnalysis): ResultContent {
  const copy = LABEL_COPY[analysis.label];
  const d = analysis.derived;
  const confidencePct = Math.round(analysis.confidence * 100);

  // ── stat chips ──
  const chips: StatChip[] = [];
  if (d.estimatedOvulationDay) {
    chips.push({
      label: "Ovulation estimée",
      value: `J${d.estimatedOvulationDay}`,
      hint: "Dernier jour avant la montée thermique (règle 3 sur 6).",
    });
  }
  if (d.follicularLength) {
    chips.push({
      label: "Phase folliculaire",
      value: `${d.follicularLength} j`,
    });
  }
  if (d.lutealLength) {
    chips.push({
      label: "Phase lutéale",
      value: `${d.lutealLength} j`,
      hint: "≥ 10 jours est considéré comme une bonne durée.",
    });
  }
  if (d.thermalRiseAmplitude > 0) {
    chips.push({
      label: "Amplitude de montée",
      value: fmtAmpl(d.thermalRiseAmplitude),
    });
  }
  if (d.lhPeakDay) {
    chips.push({
      label: "Pic LH",
      value: `J${d.lhPeakDay}`,
      hint: d.lhPeakCount > 1 ? `${d.lhPeakCount} pics détectés` : undefined,
    });
  }

  // ── narrative sections ──
  const narrative: NarrativeSection[] = [];
  if (d.estimatedOvulationDay) {
    narrative.push({
      heading: `Jours 1 à ${d.estimatedOvulationDay} — phase folliculaire`,
      body: `Avant l'ovulation, votre température reste dans sa zone basse. C'est normal : cette première moitié du cycle, la phase folliculaire, dure ${
        d.follicularLength ?? d.estimatedOvulationDay
      } jours sur ce cycle. Une phase folliculaire qui varie d'un cycle à l'autre est courante et n'est pas inquiétante en soi.`,
    });
    narrative.push({
      heading: `Autour de J${d.estimatedOvulationDay} — l'ovulation`,
      body:
        d.thermalRiseAmplitude > 0.2
          ? `On observe une montée thermique nette de ${fmtAmpl(
              d.thermalRiseAmplitude,
            )}. C'est le signe que l'ovulation a probablement eu lieu : le corps jaune produit de la progestérone, qui élève la température.`
          : `La transition vers la phase lutéale est ${
              d.thermalRiseAmplitude > 0
                ? "présente mais discrète"
                : "difficile à identifier"
            }. Une montée peu marquée mérite d'être confirmée par un dosage hormonal.`,
    });
    if (d.lutealLength) {
      narrative.push({
        heading: `Jours ${d.estimatedOvulationDay + 1} à fin — phase lutéale`,
        body: `Après l'ovulation, la température reste en plateau ${
          d.lutealLength >= 10 ? "sur une bonne durée" : "sur une durée courte"
        } (${d.lutealLength} jours observés). Une phase lutéale d'au moins 10 jours laisse le temps à une éventuelle implantation.`,
      });
    }
  } else {
    narrative.push({
      heading: "Lecture de la courbe",
      body: "Aucune montée thermique claire n'a pu être isolée sur ce cycle. Cela peut venir d'un cycle sans ovulation, de mesures trop espacées, ou de relevés pris à des heures variables. Mesurer chaque matin à la même heure, avant de se lever, améliore beaucoup la lisibilité.",
    });
  }

  // ── doctor questions ──
  const doctorQuestions: string[] = [];
  if (copy.doctorQuestion) doctorQuestions.push(copy.doctorQuestion);
  if (analysis.label === "ovulation_confirmee" && d.lutealLength && d.lutealLength < 11) {
    doctorQuestions.push(
      "Votre phase lutéale est un peu juste : un dosage de progestérone en phase lutéale peut être utile.",
    );
  }

  // ── OOD note ──
  let oodNote: string | null = null;
  if (analysis.oodPercentile >= 80) {
    oodNote =
      "Votre courbe est assez atypique par rapport aux profils habituels. L'analyse reste indicative — n'hésitez pas à en discuter avec un professionnel de santé.";
  }

  // ── glossary (only terms used) ──
  const usedTerms = new Set<string>();
  const haystack = (
    copy.summary +
    " " +
    narrative.map((n) => n.body).join(" ")
  ).toLowerCase();
  for (const term of Object.keys(GLOSSARY)) {
    if (haystack.includes(term)) usedTerms.add(term);
  }
  const glossary: GlossaryTerm[] = [...usedTerms].map((term) => ({
    term,
    definition: GLOSSARY[term],
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
