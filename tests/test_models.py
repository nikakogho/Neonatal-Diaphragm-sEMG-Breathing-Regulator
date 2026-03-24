"""
Tests for the canonical model registry.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models import CNN1DFeatureRegressor, CNN1DRawRegressor, MLPFeatureRegressor, MODEL_REGISTRY, get_model


def test_registry_contains_expected_models():
    assert set(MODEL_REGISTRY) == {
        "mlp_feature",
        "cnn1d_feature",
        "cnn1d_raw",
        "mlp_feature_scalar",
        "cnn1d_feature_scalar",
        "cnn1d_raw_scalar",
    }


def test_feature_models_forward():
    x = torch.randn(4, 4, 12)
    mlp = MLPFeatureRegressor(input_dim=48, hidden_dim=64, dropout=0.0)
    cnn = CNN1DFeatureRegressor(input_channels=12, base_channels=16, dropout=0.0)
    assert mlp(x).shape == (4, 2)
    assert cnn(x).shape == (4, 2)


def test_raw_model_forward():
    x = torch.randn(2, 20, 6, 8, 8)
    model = CNN1DRawRegressor(base_channels=8, dropout=0.1)
    out = model(x)
    assert out.shape == (2, 2)
    assert not torch.isnan(out).any()


def test_model_factory_instantiates_models():
    assert isinstance(get_model("mlp_feature", input_dim=48), nn.Module)
    assert isinstance(get_model("cnn1d_feature", input_channels=12), nn.Module)
    assert isinstance(get_model("cnn1d_raw"), nn.Module)
    assert isinstance(get_model("cnn1d_feature_scalar", input_channels=12, output_dim=1), nn.Module)


def test_model_factory_rejects_unknown_model():
    with pytest.raises(ValueError):
        get_model("unknown")


def test_scalar_model_output_shapes():
    x_feature = torch.randn(4, 4, 12)
    x_raw = torch.randn(2, 20, 6, 8, 8)
    feature_model = get_model("cnn1d_feature_scalar", input_channels=12, output_dim=1)
    raw_model = get_model("cnn1d_raw_scalar", output_dim=1)
    mlp_model = get_model("mlp_feature_scalar", input_dim=48, output_dim=1)
    assert feature_model(x_feature).shape == (4, 1)
    assert mlp_model(x_feature).shape == (4, 1)
    assert raw_model(x_raw).shape == (2, 1)
