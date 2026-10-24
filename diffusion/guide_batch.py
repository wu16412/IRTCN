"""Batch adapters shared by fusion-guide train and evaluation paths."""
from __future__ import annotations

import torch


def unpack_guide_batch(batch):
    """Return x, y, NAM and region payload while preserving legacy batches."""
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise TypeError("expected a batch tuple/list containing at least (x, y)")
    nam = None
    region = None
    if len(batch) >= 3:
        payload = batch[2]
        if isinstance(payload, dict):
            region = payload
        else:
            nam = payload
    return batch[0], batch[1], nam, region


def region_to_device(region, device):
    if region is None:
        return None
    if not isinstance(region, dict):
        raise TypeError("region payload must be a dictionary")
    return {key: value.to(device) for key, value in region.items()}
