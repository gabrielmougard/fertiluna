"""Pre-render a fixed synthetic chart dataset to disk.

Rendering with matplotlib is the training bottleneck (~35 charts/s/process).
Re-rendering every epoch wastes time, so we render ONCE to a compact .npz
(uint8 images + float16 labels) and then train many fast epochs over it.

Usage:
    python -m scripts.build_vision_dataset --out data --n 30000 --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fertiluna_vision.constants import IMG_H, IMG_W, N_DAYS, N_SERIES
from fertiluna_vision.render import render_chart


def _render_one(seed: int):
    rng = np.random.default_rng(seed)
    s = render_chart(rng)
    img = np.asarray(s.image, dtype=np.uint8)  # H,W,3
    return img, s.value.astype(np.float16), s.present.astype(np.float16)


def build(out: Path, n: int, seed0: int, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    images = np.empty((n, IMG_H, IMG_W, 3), dtype=np.uint8)
    values = np.empty((n, N_SERIES, N_DAYS), dtype=np.float16)
    presents = np.empty((n, N_SERIES, N_DAYS), dtype=np.float16)

    seeds = [seed0 * 7_000_003 + i for i in range(n)]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (img, val, pres) in enumerate(ex.map(_render_one, seeds, chunksize=16)):
            images[i] = img
            values[i] = val
            presents[i] = pres
            done += 1
            if done % 1000 == 0:
                print(f"  rendered {done}/{n}")

    path = out / f"charts-{n}-seed{seed0}.npz"
    np.savez_compressed(path, images=images, values=values, presents=presents)
    print(f"[dataset] wrote {path} ({path.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    build(args.out, args.n, args.seed, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
