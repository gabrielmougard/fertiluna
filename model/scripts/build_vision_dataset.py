"""Pre-render a fixed synthetic chart dataset to disk.

Rendering with matplotlib is the training bottleneck (~35 charts/s/process).
Re-rendering every epoch wastes time, so we render ONCE to disk and then train
many fast epochs over it.

To keep peak RAM low (a 120k set of 224x384x3 uint8 images is ~31 GB), the
dataset is streamed to disk in chunks via memory-mapped .npy files inside an
output directory, rather than building one giant in-RAM array. The directory
holds:
    images.npy   (N,H,W,3) uint8
    values.npy   (N,S,D)   float16
    presents.npy (N,S,D)   float16
CachedChartDataset memory-maps these, so neither building nor training needs to
hold the whole set in RAM. Pass the directory to --train-npz / --val-npz.

Usage:
    python -m scripts.build_vision_dataset --out data --n 30000 --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from fertiluna_vision.constants import IMG_H, IMG_W, N_DAYS, N_SERIES
from fertiluna_vision.render import render_chart


def _render_one(seed: int):
    rng = np.random.default_rng(seed)
    s = render_chart(rng)
    img = np.asarray(s.image, dtype=np.uint8)  # H,W,3
    return img, s.value.astype(np.float16), s.present.astype(np.float16)


def build(out: Path, n: int, seed0: int, workers: int, chunk: int = 20_000) -> None:
    ds_dir = out / f"charts-{n}-seed{seed0}"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Memory-mapped output arrays: written straight to disk, so peak RAM is only
    # a chunk's worth of results plus the OS page cache (which the kernel evicts
    # under pressure) — independent of n.
    images = open_memmap(
        ds_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(n, IMG_H, IMG_W, 3)
    )
    values = open_memmap(
        ds_dir / "values.npy", mode="w+", dtype=np.float16, shape=(n, N_SERIES, N_DAYS)
    )
    presents = open_memmap(
        ds_dir / "presents.npy", mode="w+", dtype=np.float16, shape=(n, N_SERIES, N_DAYS)
    )

    seeds = [seed0 * 7_000_003 + i for i in range(n)]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (img, val, pres) in enumerate(ex.map(_render_one, seeds, chunksize=16)):
            images[i] = img
            values[i] = val
            presents[i] = pres
            done += 1
            if done % chunk == 0:
                # Flush this chunk to disk and drop dirty pages so RAM stays flat.
                images.flush()
                values.flush()
                presents.flush()
                print(f"  rendered {done}/{n} (flushed)")
            elif done % 1000 == 0:
                print(f"  rendered {done}/{n}")

    images.flush()
    values.flush()
    presents.flush()
    del images, values, presents  # close the memmaps
    total = sum(p.stat().st_size for p in ds_dir.glob("*.npy"))
    print(f"[dataset] wrote {ds_dir}/ ({total/1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--chunk",
        type=int,
        default=20_000,
        help="Flush memmaps to disk every CHUNK samples to keep RAM flat.",
    )
    args = p.parse_args()
    build(args.out, args.n, args.seed, args.workers, args.chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
