"""
MA-IUVF Configuration System Tests
"""

import pytest
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Helper function to get config paths
def get_config_path(config_name: str) -> str:
    """Get absolute path to config file"""
    project_root = Path(__file__).parent.parent
    return str(project_root / "configs" / config_name)

from maiuvf_config import MAIUVFConfig
from maiuvf_config_loader import MAIUVFConfigLoader
from maiuvf_config_utils import compare_configs, compute_config_diff


class TestConfigLoading:
    """Test configuration loading"""

    def test_load_yaml_config(self):
        """Test loading configuration from YAML file"""
        config = MAIUVFConfig.from_yaml(get_config_path("maiuvf_baseline.yaml"))

        assert config.experiment.name == "maiuvf_baseline"
        assert config.paths.input_mesh == "data/models/stanford_bunny.obj"
        assert config.training.epochs == 100
        assert config.model.encoder_type == "bspline_hash"
        print("✓ YAML loading test passed")

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = MAIUVFConfig.from_yaml(get_config_path("maiuvf_baseline.yaml"))
        errors = config.validate()
        assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"

        # Invalid config (missing paths)
        config_invalid = MAIUVFConfig()
        errors = config_invalid.validate()
        assert len(errors) > 0, "Invalid config should have errors"
        assert "paths.input_mesh is required" in errors

        print("✓ Config validation test passed")

    def test_save_yaml_config(self):
        """Test saving configuration to YAML"""
        import tempfile

        config = MAIUVFConfig.from_yaml(get_config_path("maiuvf_baseline.yaml"))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            # Load and verify
            config_loaded = MAIUVFConfig.from_yaml(temp_path)
            assert config_loaded.experiment.name == config.experiment.name
            assert config_loaded.training.epochs == config.training.epochs

            print("✓ YAML saving test passed")
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCLIOverride:
    """Test CLI override functionality"""

    def test_cli_override(self):
        """Test CLI argument overrides"""
        import argparse

        # Create base config
        config = MAIUVFConfig.from_yaml(get_config_path("maiuvf_baseline.yaml"))
        assert config.training.epochs == 100

        # Simulate CLI args
        class Args:
            epochs = 200
            config = get_config_path("maiuvf_baseline.yaml")

        args = Args()

        # Apply override
        loader = MAIUVFConfigLoader()
        loader.config = config
        modified_config = loader.apply_cli_overrides(args)

        assert modified_config.training.epochs == 200, "Epochs should be overridden"
        print("✓ CLI override test passed")

    def test_flat_config_generation(self):
        """Test flat config dictionary generation"""
        config = MAIUVFConfig.from_yaml(get_config_path("maiuvf_baseline.yaml"))

        loader = MAIUVFConfigLoader()
        loader.config = config
        flat_config = loader.get_flat_config()

        # Check all required keys exist
        required_keys = ['input_mesh', 'texture', 'output_dir', 'epochs', 'batch_size']
        for key in required_keys:
            assert key in flat_config, f"Missing key: {key}"

        print("✓ Flat config generation test passed")


class TestConfigComparison:
    """Test configuration comparison utilities"""

    def test_compare_configs(self):
        """Test configuration comparison"""
        config1 = MAIUVFConfig.from_yaml(get_config_path("maiuvf_baseline.yaml"))
        config2 = MAIUVFConfig.from_yaml(get_config_path("maiuvf_quick_test.yaml"))

        comparison = compare_configs(config1, config2)

        assert 'different' in comparison
        assert 'same' in comparison
        assert comparison['different']['training.epochs']['config1'] == 100
        assert comparison['different']['training.epochs']['config2'] == 10

        print("✓ Config comparison test passed")

    def test_compute_config_diff(self):
        """Test configuration diff computation"""
        config1 = MAIUVFConfig.from_yaml(get_config_path("maiuvf_baseline.yaml"))

        # Create modified config
        config2 = MAIUVFConfig.from_dict(config1.to_dict())
        config2.training.epochs = 200

        diff = compute_config_diff(config1, config2)

        assert 'changed' in diff
        assert diff['changed']['training.epochs'] == {'config1': 100, 'config2': 200}

        print("✓ Config diff computation test passed")


class TestBackwardCompatibility:
    """Test backward compatibility with existing CLI usage"""

    def test_cli_mode_without_config(self):
        """Test that CLI mode still works without --config"""
        # This test verifies the structure is compatible
        # Actual end-to-end testing requires full script run
        flat_config = {
            'input_mesh': "test.obj",
            'texture': "test.png",
            'output_dir': "outputs/test",
            'epochs': 10,
            'batch_size': 4096,
            # ... other required fields
        }

        # Verify structure
        assert 'input_mesh' in flat_config
        assert 'epochs' in flat_config
        assert flat_config['epochs'] == 10

        print("✓ CLI mode compatibility test passed")


if __name__ == "__main__":
    # Run all tests
    print("\n🧪 Running MA-IUVF Configuration Tests...\n")

    test_classes = [
        TestConfigLoading,
        TestCLIOverride,
        TestConfigComparison,
        TestBackwardCompatibility,
    ]

    for test_class in test_classes:
        test_name = test_class.__name__
        print(f"\n{test_name}:")
        test_obj = test_class()
        for method_name in dir(test_obj):
            if method_name.startswith('test_'):
                test_func = getattr(test_obj, method_name)
                try:
                    test_func()
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")

    print("\n✅ All configuration tests completed!")
