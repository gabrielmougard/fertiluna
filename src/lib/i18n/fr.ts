/** French catalog (primary locale). Keys shared with en.ts. */
export const fr = {
  // analysis backend / consent
  "consent.title": "Analyse améliorée par IA",
  "consent.body":
    "Pour une lecture plus précise, votre image peut être analysée par un "
    + "service d'IA externe. Dans ce cas, l'image quitte votre "
    + "appareil le temps de l'analyse. Sans votre accord, l'analyse reste "
    + "100 % sur votre appareil.",
  "consent.accept": "Autoriser l'analyse IA améliorée",
  "consent.decline": "Rester 100 % sur mon appareil",
  "consent.medical":
    "FertiLuna n'est pas un dispositif médical. Les résultats sont "
    + "indicatifs et ne remplacent pas l'avis d'un professionnel de santé.",
  // status messages
  "status.extracted": "Analyse réussie. Vérifiez l'aperçu avant d'importer.",
  "status.low_confidence":
    "Résultat incertain : vérifiez attentivement chaque valeur, ou utilisez "
    + "la calibration manuelle.",
  "status.not_a_chart":
    "Aucune courbe de cycle détectée. Essayez une autre image ou la saisie "
    + "manuelle.",
  "backend.on_device": "Analyse sur l'appareil",
  "backend.cloud": "Analyse IA (cloud)",

  // ── shared chrome (header / footer) ──
  "brand.domain": "FertiLuna",
  "nav.tool": "Analyser ma courbe",
  "nav.tools": "Les outils",
  "nav.about": "À propos",
  "footer.medical":
    "<strong>FertiLuna</strong> n'est pas un dispositif médical. Les "
    + "résultats sont indicatifs et ne remplacent pas l'avis d'un "
    + "professionnel de santé.",
  "footer.privacy":
    "Aucune donnée n'est collectée ni envoyée à un serveur. Toute l'analyse se "
    + "déroule dans votre navigateur.",
  "footer.sourceLabel": "Code source",
  "footer.sourceLead": "Projet open source :",

  // ── tool page (outils/analyse-courbe) ──
  "tool.metaTitle": "Analyse de courbe de température · FertiLuna",
  "tool.metaDescription":
    "Renseignez vos températures basales et vos tests LH jour par jour. Suivi "
    + "Grossesse détecte l'ovulation, évalue vos phases folliculaire et lutéale "
    + "et interprète votre cycle, sans inscription, intégralement dans votre "
    + "navigateur.",
  "tool.eyebrow": "Outil 1 · Analyse de courbe",
  "tool.h1": "Analysez votre courbe",
  "tool.intro":
    "Renseignez vos températures basales (mesurées au réveil, avant tout "
    + "lever) et, le cas échéant, vos tests d'ovulation (LH). Le traitement "
    + "s'exécute <strong>intégralement dans votre navigateur</strong> : aucune "
    + "donnée n'est transmise à un serveur.",
  "tool.useTemp": "Température",
  "tool.useLh": "Tests LH",
  "tool.colDay": "Jour",
  "tool.colTemp": "Temp. (°C)",
  "tool.colLh": "LH (0–3)",
  "tool.analyze": "Analyser mon cycle",
  "tool.analyzing": "Analyse en cours…",
  "tool.demo": "Remplir un exemple",
  "tool.clear": "Effacer",
  "tool.needData":
    "Saisissez au moins quelques jours de données pour lancer l'analyse.",
  "tool.error":
    "Une erreur est survenue pendant l'analyse. Réessayez dans un instant.",
  "tool.modelReady":
    "Modèle prêt et mis en cache ({mb} Mo). Vos analyses suivantes seront "
    + "instantanées et hors-ligne.",
  "tool.modelLoading": "Chargement du modèle… {pct}%",
  "tool.modelPreparing": "Préparation du modèle…",
  "tool.imported":
    "{n} valeurs importées depuis l'image. Vérifiez-les puis lancez l'analyse.",
  "tool.doctorTitle": "À demander à votre médecin",
  "tool.glossaryTitle": "Glossaire : comprendre les termes",
  "tool.legendFollicular": "Phase folliculaire",
  "tool.legendLuteal": "Phase lutéale",
  "tool.legendLhPeak": "Pic LH",
  "tool.confidence": "Confiance {pct}%",
  "tool.confidenceLow": "Confiance limitée ({pct}%)",
  "tool.resultDisclaimer":
    "Cette analyse est indicative. En cas de doute ou de difficulté à "
    + "concevoir, parlez-en à votre médecin ou sage-femme.",

  // ── image import / digitizer (CurveDigitizer + controllers) ──
  "dz.sectionTitle": "Importer une capture d'écran",
  "dz.sectionDesc":
    "Une courbe dans une autre appli (Inito, Premom, Clearblue, Flo…) ou sur "
    + "papier ? Déposez une capture, FertiLuna lit les courbes pour vous. "
    + "<strong>L'image reste sur votre appareil.</strong>",
  "dz.dropTitle": "Déposez votre capture ici",
  "dz.dropSub": "ou cliquez pour choisir une image",
  "dz.dropFormats": "PNG, JPG · l'image ne quitte pas votre appareil",
  "dz.tempLabel": "Température (BBT)",
  "dz.lhLabel": "LH / hormone",
  "dz.import": "Importer dans le tableau",
  "dz.reset": "Changer d'image",
  "dz.advanced": "Ajustement manuel",
  "dz.advancedHint": "(si la lecture automatique se trompe)",
  "dz.manualUpload": "Choisir une image",
  "dz.seriesLabel": "Série :",
  "dz.step1":
    "Calibrez l'axe des <strong>jours</strong> (2 clics sur la ligne ZT/jour).",
  "dz.step2":
    "Calibrez l'axe de <strong>valeurs</strong> de la série choisie. "
    + "Attention, BBT et LH sont souvent sur des axes <strong>opposés</strong>.",
  "dz.step3":
    "Cliquez sur la <strong>courbe</strong> de cette série pour prendre sa "
    + "couleur.",
  "dz.step4": "Changez de série pour la seconde courbe, ou importez.",
  "dz.tol": "Tolérance couleur",
  "dz.redo": "Recalibrer cette série",
  // auto-extract status / loader
  "dz.analyzing": "Analyse de l'image…",
  "dz.analyzingCloud": "Analyse améliorée par IA…",
  "dz.analyzingDevice": "Analyse sur l'appareil…",
  "dz.failed": "La lecture automatique a échoué sur cette image.",
  "dz.unreadable": "Image illisible.",
  "dz.unavailable": "La lecture automatique n'a pas pu être préparée.",
  "dz.tryManual": " Essayez l'ajustement manuel ci-dessous.",
  "dz.noCurve": "Aucune courbe de cycle détectée sur cette image.",
  "dz.noCurveStatus":
    "Aucune courbe exploitable détectée. Vérifiez que l'image est bien un "
    + "graphique de cycle, ou utilisez l'ajustement manuel.",
  "dz.noNet":
    "Aucune courbe nette détectée. Essayez une image plus contrastée ou la "
    + "calibration manuelle.",
  "dz.scaleDetected": "Échelle détectée : {unit} ({min}–{max})",
  "dz.detectedTemp": "température en {unit} ({n} j)",
  "dz.detectedLh": "LH ({n} j)",
  "dz.detected": "Détecté : {parts}.",
  "dz.verifyImport":
    "{detected} Vérifiez l'aperçu, puis importez. Vous corrigerez les valeurs "
    + "dans le tableau si besoin.",
  "dz.lowConfidence":
    "Confiance limitée ({pct} %) — vérifiez attentivement chaque point dans le "
    + "tableau, ou utilisez l'ajustement manuel. {detected}",
  // overlay hover tooltips
  "dz.tipDay": "jour {day}",
  "dz.tipPlot": "Zone du graphique ({method})",
  "dz.tipAxis": "Axe {kind} · {v}",
  "dz.tipRow": "Ligne « {name} »",
  "dz.tipCellEmpty": "{name} (vide)",
  // manual digitizer (prompts)
  "dzm.idle": "Importez une capture d'écran de votre courbe pour commencer.",
  "dzm.day1":
    "Axe des jours (1/2) : cliquez sur un repère de jour connu (ligne ZT/jour), "
    + "puis indiquez son numéro.",
  "dzm.day2":
    "Axe des jours (2/2) : cliquez sur un autre repère de jour, plus à droite.",
  "dzm.value1":
    "« {label} », valeur (1/2) : cliquez sur une graduation de son axe "
    + "({side}), puis indiquez sa valeur.",
  "dzm.value2":
    "« {label} », valeur (2/2) : cliquez sur une autre graduation du même axe.",
  "dzm.color":
    "« {label} » : cliquez directement sur sa courbe pour sélectionner sa "
    + "couleur.",
  "dzm.ready":
    "« {label} » détectée. Vérifiez l'aperçu, ajustez la tolérance, ou passez "
    + "à l'autre série / importez.",
  "dzm.askValue1": "{label}, valeur de cette graduation ({unit}, ex. {ex}) :",
  "dzm.askValue2": "{label}, valeur de l'autre graduation ({unit}, ex. {ex}) :",
  "dzm.noPoint":
    "Aucun point détecté : cliquez précisément sur la courbe (utilisez la "
    + "loupe) ou augmentez la tolérance.",
  "dzm.daysDetected": "{n} jours détectés pour « {label} ».",
  "dzm.sideTemp": "souvent l'axe de DROITE (°C)",
  "dzm.sideLh": "souvent l'axe de GAUCHE",
  "dzm.askDay1": "Numéro de ce jour (ex. 7) :",
  "dzm.askDay2": "Numéro de ce jour (plus à droite, ex. 24) :",

  // ── run history (on-device, IndexedDB) ──
  "history.title": "Mes analyses",
  "history.subtitle":
    "Vos analyses précédentes sont enregistrées sur cet appareil uniquement. "
    + "Rien n'est envoyé en ligne.",
  "history.empty":
    "Aucune analyse enregistrée pour l'instant. Vos prochaines analyses "
    + "apparaîtront ici.",
  "history.open": "Rouvrir",
  "history.delete": "Supprimer",
  "history.clearAll": "Tout effacer",
  "history.clearConfirm":
    "Effacer toutes vos analyses enregistrées sur cet appareil ?",
  "history.saved": "Analyse enregistrée sur cet appareil.",
  "history.restored": "Analyse rechargée. Relancez pour recalculer si besoin.",
  "history.count": "{n} enregistrée(s)",
  "history.withImage": "avec image",
} as const;
