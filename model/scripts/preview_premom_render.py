"""Render a single random Premom-style chart to a PNG so you can eyeball it.

Usage:
    python -m scripts.preview_premom_render --out /tmp/premom.png --seed 7
    # generic (non-Premom) renderer for comparison:
    python -m scripts.preview_premom_render --out /tmp/generic.png --style generic

The saved image is the EXACT 384x224 canvas the model receives (after resize),
plus, with --full, the pre-resize figure so you can read the details.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fertiluna_vision import render as render_mod
from fertiluna_vision.render import render_chart, render_premom_chart


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("/tmp/premom_preview.png"))
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed; omit for a fresh random chart each run.")
    p.add_argument("--style", choices=["premom", "generic"], default="premom")
    p.add_argument("--full", action="store_true",
                   help="Also save the high-res pre-resize figure (<out>_full.png).")
    p.add_argument("--dpi", type=int, default=None,
                   help="Force a high DPI for the pre-resize figure (premom only), e.g. 300, to inspect fine detail. Implies --full.")
    args = p.parse_args()

    if args.dpi is not None:
        args.full = True

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng(seed)

    # Capture the full-resolution figure (before the model-canvas downscale) by
    # intercepting the renderer's figure->PIL step.
    captured = {}
    if args.full:
        _orig = render_mod._fig_to_pil

        def _hook(fig):
            img = _orig(fig)
            captured["full"] = img.copy()
            return img

        render_mod._fig_to_pil = _hook

    if args.style == "premom":
        sample = render_premom_chart(rng, dpi_override=args.dpi)
    else:
        sample = render_chart(rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.image.save(args.out)
    print(f"[preview] style={args.style} seed={seed}")
    print(f"[preview] saved model-canvas image ({sample.image.size}) -> {args.out}")
    if args.full and "full" in captured:
        full_path = args.out.with_name(args.out.stem + "_full" + args.out.suffix)
        captured["full"].save(full_path)
        print(f"[preview] saved high-res figure ({captured['full'].size}) -> {full_path}")
    print(f"[preview] meta: {sample.meta}")
    print(f"[preview] temp present days: {int(sample.present[0].sum())}, "
          f"lh present days: {int(sample.present[1].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
