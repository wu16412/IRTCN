"""ImageFolder that also returns NAM WLD-144 vectors for Texture-CDM."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torchvision.datasets import ImageFolder


SPLIT_FILES = {
    "train": "training.npz",
    "valid": "validation.npz",
    "test": "testing.npz",
}

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "test": "test",
    "testing": "test",
}


def _path_keys(raw_path: str | Path) -> tuple[str, ...]:
    normalized = str(raw_path).replace("\\", "/")
    path = Path(normalized)
    keys = [normalized]
    if len(path.parts) >= 3:
        keys.extend(
            (
                "/".join(path.parts[-3:]),
                "/".join(path.parts[-3:-1] + (path.stem,)),
            )
        )
    if len(path.parts) >= 2:
        keys.extend((f"{path.parts[-2]}/{path.name}", f"{path.parts[-2]}/{path.stem}"))
    keys.extend((path.name, path.stem))
    return tuple(dict.fromkeys(key for key in keys if key))


def _canonical_split(split: str) -> str:
    key = str(split).strip().lower()
    if key not in _SPLIT_ALIASES:
        raise ValueError(f"unknown NAM split {split!r}")
    return _SPLIT_ALIASES[key]


def load_nam_lookup(
    feature_dir: str | Path,
    splits: Iterable[str] | None = None,
    require_splits: bool = False,
) -> dict[str, np.ndarray]:
    """Load WLD/NAM vectors. Default keeps dual compatibility (all present splits)."""
    from fusion_guide_runtime import record_split_access

    feature_dir = Path(feature_dir)
    requested = tuple(_canonical_split(split) for split in (splits or ("train", "valid", "test")))
    lookup: dict[str, np.ndarray] = {}
    for split in requested:
        fname = SPLIT_FILES[split]
        path = feature_dir / fname
        if not path.is_file():
            if require_splits:
                raise FileNotFoundError(f"required NAM split missing: {path}")
            continue
        record_split_access("array_splits", split)
        with np.load(path, allow_pickle=True) as data:
            nam = np.asarray(data["nam"] if "nam" in data.files else data["lbp"], dtype=np.float32)
            paths = data["paths"]
        if len(paths) != len(nam):
            raise RuntimeError(f"{path}: paths {len(paths)} != nam {len(nam)}")
        if nam.ndim != 2:
            raise ValueError(f"{path}: NAM must be (N,D), got {tuple(nam.shape)}")
        for i, raw in enumerate(paths):
            vec = nam[i]
            for key in _path_keys(str(raw)):
                lookup[key] = vec
    if not lookup:
        raise FileNotFoundError(f"No NAM npz under {feature_dir}")
    return lookup


def resolve_nam(lookup: dict[str, np.ndarray], image_path: str) -> np.ndarray:
    for key in _path_keys(image_path):
        if key in lookup:
            return lookup[key]
    raise KeyError(f"NAM missing for {image_path}")


class ImageFolderWithNam(ImageFolder):
    def __init__(self, root, nam_lookup: dict[str, np.ndarray], transform=None):
        super().__init__(root, transform=transform)
        self.nam_lookup = nam_lookup

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        nam = torch.from_numpy(resolve_nam(self.nam_lookup, path).copy())
        return sample, target, nam


class PaddedRegionStore:
    """One split-wide padded NPZ loaded once and indexed by image path."""

    REQUIRED = ("region_wld", "region_mask", "token_region_overlap", "paths")

    def __init__(self, feature_dir: str | Path, split: str):
        if split not in SPLIT_FILES:
            raise ValueError(f"unknown split: {split}")
        path = Path(feature_dir) / SPLIT_FILES[split]
        if not path.is_file():
            raise FileNotFoundError(f"missing padded region split: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in self.REQUIRED if key not in data.files]
            if missing:
                raise KeyError(f"{path} is missing padded region fields: {missing}")
            self.region_wld = np.asarray(data["region_wld"], dtype=np.float32)
            self.region_mask = np.asarray(data["region_mask"], dtype=np.bool_)
            self.overlap = np.asarray(data["token_region_overlap"], dtype=np.float32)
            paths = np.asarray(data["paths"]).astype(str)
        sample_count = len(paths)
        if self.region_wld.ndim != 3 or self.region_wld.shape[0] != sample_count:
            raise ValueError(f"{path}: region_wld must have shape (N,R,D)")
        region_count = self.region_wld.shape[1]
        if self.region_mask.shape != (sample_count, region_count):
            raise ValueError(f"{path}: region_mask shape does not match region_wld")
        if self.overlap.ndim != 3 or self.overlap.shape[:2] != (sample_count, 49):
            raise ValueError(f"{path}: token_region_overlap must have shape (N,49,R)")
        if self.overlap.shape[2] != region_count:
            raise ValueError(f"{path}: overlap region count does not match region_wld")
        self.lookup: dict[str, int | None] = {}
        for index, raw in enumerate(paths):
            for key in _path_keys(raw):
                previous = self.lookup.setdefault(key, index)
                if previous is not None and previous != index:
                    self.lookup[key] = None

    def get(self, image_path: str) -> dict[str, torch.Tensor]:
        index = None
        for key in _path_keys(image_path):
            candidate = self.lookup.get(key)
            if candidate is not None:
                index = candidate
                break
        if index is None:
            raise KeyError(f"region features missing for {image_path}")
        return {
            "region_wld": torch.from_numpy(self.region_wld[index].copy()),
            "region_mask": torch.from_numpy(self.region_mask[index].copy()),
            "token_region_overlap": torch.from_numpy(self.overlap[index].copy()),
        }


class ImageFolderWithRegions(ImageFolder):
    def __init__(self, root, region_store: PaddedRegionStore, transform=None):
        super().__init__(root, transform=transform)
        self.region_store = region_store

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, self.region_store.get(path)


def unpack_batch(batch):
    """Support (x,y), (x,y,nam), or (x,y,region_dict)."""
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None
