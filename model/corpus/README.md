# Real-screenshot corpus

Hand-labeled real fertility-chart screenshots — the ground truth the CV
digitizer is held to on the distribution we actually deploy on. The synthetic
eval (`scripts/eval_vision_cv.py`) gives volume + free labels, but its
renderer diverges from real apps; this corpus is the reality check.

## Layout

```
corpus/
  <name>.png                 the screenshot
  <name>.labels.json         ground truth (see schema below)
  <name>_overlay.png         pipeline's annotated view (labeling aid)
```

## Workflow (model-assisted labeling)

1. **Seed** labels from the pipeline's prediction (fast start — you correct,
   you don't label from blank):

   ```bash
   python -m scripts.label_assist --images path/to/shots/*.png --out corpus
   ```

   This writes a pre-filled `<name>.labels.json` (`reviewed: false`) and a
   `<name>_overlay.png`.

2. **Correct.** Open `<name>_overlay.png`, compare to the screenshot, and fix
   `<name>.labels.json` where the pipeline got it wrong:
   - `value` — `(2, 35)` normalized `[0,1]` within each series' axis range.
     Row 0 = temp (BBT), row 1 = lh. (0.0 = axis bottom, 1.0 = axis top.)
   - `present` — `(2, 35)` `{0,1}`: was a data point measured that day?
   - `scale_idx` — `0` = celsius, `1` = fahrenheit.
   Then set `"reviewed": true`.

3. **Gate.** Only reviewed labels are scored:

   ```bash
   python -m scripts.eval_corpus --corpus corpus
   ```

## Target

Aim for **~50 reviewed screenshots** spanning the apps + edge cases you expect
in production (Premom FR/EN, Celsius and Fahrenheit, partial-width cycles,
status-bar crops, low-resolution captures, the period-band-mid-chart case).
The `scale_acc` gate activates at ≥ 8 reviewed labels.

## Privacy

These are real users' health charts. Keep the corpus out of any public
artifact: it's for local + CI evaluation only. Strip personal identifiers
from filenames. (This directory should be git-ignored or stored in a private
location depending on your policy.)
