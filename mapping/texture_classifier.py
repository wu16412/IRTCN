"""WLD-144 TextureClassifier used as a frozen Texture-CDM guide."""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


class TextureClassifier(nn.Module):
    """WLD144 -> MLP -> logits. Hidden widths and dropout are configurable."""

    def __init__(
        self,
        input_dim: int = 144,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one width")
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.config = {
            "input_dim": int(input_dim),
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "num_classes": int(num_classes),
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"NAM features must be (B,D), got {tuple(features.shape)}")
        if features.shape[1] != self.config["input_dim"]:
            raise ValueError(
                f"NAM dim {features.shape[1]} != configured {self.config['input_dim']}"
            )
        return self.net(features)


def texture_from_config(config: dict[str, Any]) -> TextureClassifier:
    return TextureClassifier(
        input_dim=int(config["input_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config.get("dropout", 0.1)),
        num_classes=int(config.get("num_classes", 2)),
    )
