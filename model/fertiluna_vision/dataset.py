"""Torch Dataset wrapping the synthetic chart renderer.

Charts are generated on-the-fly (infinite, perfectly-labeled data). A fixed
`length` defines one "epoch"; each __getitem__ renders a fresh chart with a
deterministic per-index seed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import NORM_MEAN, NORM_STD
from .render import render_chart


def _to_tensor(img) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
    arr = (arr - np.array(NORM_MEAN)) / np.array(NORM_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()


class ChartDataset(Dataset):
    def __init__(self, length: int, seed: int = 0):
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        sample = render_chart(rng)
        x = _to_tensor(sample.image)
        value = torch.from_numpy(sample.value)      # (S, D)
        present = torch.from_numpy(sample.present)  # (S, D)
        return x, value, present


class CachedChartDataset(Dataset):
    """Reads pre-rendered charts from an .npz built by build_vision_dataset.py.

    Much faster than re-rendering every epoch. Light, deterministic pixel
    augmentation (brightness/contrast jitter) is applied on the fly to reduce
    overfitting to the fixed render set.
    """

    def __init__(self, npz_path: str, augment: bool = True):
        data = np.load(npz_path)
        self.images = data["images"]      # (N,H,W,3) uint8
        self.values = data["values"]      # (N,S,D) f16
        self.presents = data["presents"]  # (N,S,D) f16
        self.augment = augment

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx].astype(np.float32) / 255.0  # H,W,3
        if self.augment:
            # brightness/contrast jitter + light gaussian noise
            b = np.random.uniform(-0.06, 0.06)
            c = np.random.uniform(0.92, 1.08)
            img = np.clip((img - 0.5) * c + 0.5 + b, 0.0, 1.0)
            if np.random.random() < 0.3:
                img = np.clip(img + np.random.normal(0, 0.02, img.shape), 0, 1)
        img = (img - np.array(NORM_MEAN)) / np.array(NORM_STD)
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        value = torch.from_numpy(self.values[idx].astype(np.float32))
        present = torch.from_numpy(self.presents[idx].astype(np.float32))
        return x, value, present
