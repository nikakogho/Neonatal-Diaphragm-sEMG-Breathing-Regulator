"""
Canonical model registry.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class MLPFeatureRegressor(nn.Module):
    """Two-layer MLP over flattened binned features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected feature tensor shape (B, bins, 12), got {tuple(x.shape)}")
        return self.net(x.flatten(start_dim=1))


class CNN1DFeatureRegressor(nn.Module):
    """Temporal 1D CNN over per-bin grid features."""

    def __init__(self, input_channels: int = 12, base_channels: int = 16, dropout: float = 0.0, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(base_channels * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected feature tensor shape (B, bins, 12), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1)
        return self.head(self.net(x).squeeze(-1))


class CNN1DRawRegressor(nn.Module):
    """1D CNN over raw time windows with flattened spatial channels."""

    def __init__(self, input_features: int = 6 * 8 * 8, base_channels: int = 16, dropout: float = 0.15, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_features, base_channels * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected raw tensor shape (B, time, 6, 8, 8), got {tuple(x.shape)}")
        batch, time, channels, height, width = x.shape
        x = x.reshape(batch, time, channels * height * width).permute(0, 2, 1)
        return self.head(self.net(x))


MODEL_REGISTRY = {
    "mlp_feature": MLPFeatureRegressor,
    "cnn1d_feature": CNN1DFeatureRegressor,
    "cnn1d_raw": CNN1DRawRegressor,
    "mlp_feature_scalar": MLPFeatureRegressor,
    "cnn1d_feature_scalar": CNN1DFeatureRegressor,
    "cnn1d_raw_scalar": CNN1DRawRegressor,
}


def get_model(model_name: str, **kwargs) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**kwargs)


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())
