"""
Unit Tests

Run with: pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ExperimentConfig, get_default_config, load_config, save_config
from src.utils import (
    get_device,
    RandomNumberGenerator,
    PositionalEncoding,
    BatchSampler,
    DataCollator,
    normalize_coordinates,
)
from src.interfaces import (
    PointSample,
    GeometryFeatures,
    ConditionVector,
    NetworkGOutput,
    SamplingRegion,
)


class TestConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default config creation."""
        config = get_default_config()
        assert isinstance(config, ExperimentConfig)
        assert config.data.num_samples_per_epoch == 2_000_000
        assert config.network_g.hidden_dim == 256

    def test_config_to_dict(self):
        """Test config serialization."""
        config = get_default_config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "network_g" in config_dict

    def test_config_yaml_roundtrip(self, tmp_path):
        """Test YAML save/load roundtrip."""
        config = get_default_config()
        config.network_g.hidden_dim = 128

        yaml_path = tmp_path / "test_config.yaml"
        save_config(config, yaml_path)

        loaded = load_config(yaml_path)
        assert loaded.network_g.hidden_dim == 128


class TestDevice:
    """Test device management."""

    def test_get_device_cpu(self):
        """Test CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_auto(self):
        """Test auto device selection."""
        device = get_device("auto")
        # Should be cuda if available, otherwise cpu
        assert device.type in ["cuda", "cpu"]


class TestRandom:
    """Test random number generation."""

    def test_set_seed(self):
        """Test seed setting."""
        RandomNumberGenerator.set_seed(42)
        assert RandomNumberGenerator.get_seed() == 42

    def test_reproducibility(self):
        """Test random reproducibility."""
        RandomNumberGenerator.set_seed(123)
        v1 = torch.randn(5)

        RandomNumberGenerator.set_seed(123)
        v2 = torch.randn(5)

        assert torch.allclose(v1, v2)


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_output_dimensions(self):
        """Test positional encoding output dimensions."""
        pe = PositionalEncoding(in_dim=3, num_frequencies=6)
        assert pe.out_dim == 39  # 3 + 3*6*2 = 39

    def test_forward(self):
        """Test forward pass."""
        pe = PositionalEncoding(in_dim=3, num_frequencies=6)
        x = torch.randn(10, 3)
        out = pe(x)
        assert out.shape == (10, 39)

    def test_without_raw_input(self):
        """Test without raw input."""
        pe = PositionalEncoding(in_dim=3, num_frequencies=6, include_input=False)
        assert pe.out_dim == 36  # 3*6*2 = 36


class TestBatchSampler:
    """Test batch sampler."""

    def test_length(self):
        """Test sampler length."""
        sampler = BatchSampler(num_samples=100, batch_size=32)
        assert len(sampler) == 4  # ceil(100/32) = 4

    def test_drop_last(self):
        """Test drop last behavior."""
        sampler = BatchSampler(num_samples=100, batch_size=32, drop_last=True)
        assert len(sampler) == 3  # floor(100/32) = 3

    def test_iteration(self):
        """Test iteration."""
        sampler = BatchSampler(num_samples=10, batch_size=3)
        batches = list(iter(sampler))
        # 4 batches: [0,1,2], [3,4,5], [6,7,8], [9]
        assert len(batches) == 4


class TestDataCollator:
    """Test data collator."""

    def test_collate_same_shape(self):
        """Test collating same-shape tensors."""
        collator = DataCollator()
        batch = [
            {"x": torch.randn(3), "y": torch.randn(2)},
            {"x": torch.randn(3), "y": torch.randn(2)},
        ]
        result = collator.collate(batch)
        assert result["x"].shape == (2, 3)
        assert result["y"].shape == (2, 2)

    def test_collate_different_shape(self):
        """Test collating variable-length tensors."""
        collator = DataCollator()
        batch = [
            {"x": torch.randn(3), "y": torch.randn(2)},
            {"x": torch.randn(3), "y": torch.randn(2)},
        ]
        result = collator.collate(batch)
        assert result["x"].shape[0] == 2


class TestCoordinates:
    """Test coordinate normalization."""

    def test_normalize_denormalize(self):
        """Test normalize and denormalize."""
        coords = torch.randn(10, 3)
        normalized, min_val, max_val, range_val = normalize_coordinates(coords)
        assert normalized.min() >= 0
        assert normalized.max() <= 1

        denormalized = normalize_coordinates(normalized, min_val, max_val)[0]
        assert torch.allclose(coords, denormalized, atol=1e-5)


class TestInterfaces:
    """Test interface data structures."""

    def test_point_sample(self):
        """Test PointSample creation."""
        sample = PointSample(
            position=np.array([1.0, 2.0, 3.0]),
            sdf_gt=0.5,
            color_gt=np.array([0.5, 0.5, 0.5]),
            color_lowpass=np.array([0.4, 0.4, 0.4]),
            curvature=np.array([0.1, 0.2]),
            normal=np.array([0.0, 0.0, 1.0]),
            boundary_distance=0.1,
            label_gt=5,
            region=SamplingRegion.SURFACE,
        )
        assert sample.position.shape == (3,)
        assert sample.color_gt.shape == (3,)

    def test_condition_vector(self):
        """Test ConditionVector."""
        cond = ConditionVector(
            color_base=torch.randn(4, 3),
            sdf=torch.randn(4),
            curvature=torch.randn(4, 2),
            normal=torch.randn(4, 3),
            boundary_distance=torch.randn(4),
            global_shape_code=torch.randn(4, 32),
        )
        tensor = cond.to_tensor()
        assert tensor.shape == (4, 42)

    def test_condition_vector_from_tensor(self):
        """Test ConditionVector from tensor."""
        tensor = torch.randn(4, 42)
        cond = ConditionVector.from_tensor(tensor)
        assert cond.color_base.shape == (4, 3)
        assert cond.global_shape_code.shape == (4, 32)


class TestMetrics:
    """Test metrics computation."""

    def test_psnr_perfect(self):
        """Test PSNR for identical images."""
        from src.evaluation import compute_psnr
        x = torch.rand(10, 3)
        psnr = compute_psnr(x, x)
        assert psnr == float("inf")

    def test_psnr_zero(self):
        """Test PSNR for very different images."""
        from src.evaluation import compute_psnr
        x = torch.zeros(10, 3)
        y = torch.ones(10, 3)
        psnr = compute_psnr(x, y)
        assert psnr < 0  # Should be very low

    def test_rmse(self):
        """Test RMSE computation."""
        from src.evaluation import compute_rmse
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        rmse = compute_rmse(x, y)
        assert rmse == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
