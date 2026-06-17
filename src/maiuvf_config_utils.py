"""
MA-IUVF Configuration Utilities

Utilities for configuration comparison, diff tracking, and management.
"""

from typing import Dict, Any, Union, List
from pathlib import Path
from maiuvf_config import MAIUVFConfig


def compare_configs(config1: MAIUVFConfig, config2: MAIUVFConfig) -> Dict[str, Any]:
    """
    Compare two configurations and return differences.

    Returns a dictionary with:
    - 'only_in_config1': parameters only in config1
    - 'only_in_config2': parameters only in config2
    - 'different': parameters with different values
    - 'same': parameters with same values
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()

    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary using dotted keys"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat1 = flatten_dict(dict1)
    flat2 = flatten_dict(dict2)

    only_in_config1 = {}
    only_in_config2 = {}
    different = {}
    same = {}

    all_keys = set(flat1.keys()) | set(flat2.keys())

    for key in all_keys:
        if key not in flat1:
            only_in_config2[key] = flat2[key]
        elif key not in flat2:
            only_in_config1[key] = flat1[key]
        elif flat1[key] != flat2[key]:
            different[key] = {
                'config1': flat1[key],
                'config2': flat2[key]
            }
        else:
            same[key] = flat1[key]

    return {
        'only_in_config1': only_in_config1,
        'only_in_config2': only_in_config2,
        'different': different,
        'same': same,
    }


def compute_config_diff(base: MAIUVFConfig, modified: MAIUVFConfig) -> Dict[str, Any]:
    """
    Compute configuration differences (for tracking CLI overrides).

    Returns a dictionary of differences from base to modified.
    """
    comparison = compare_configs(base, modified)

    # We only care about what changed in the modified config
    diff = {
        'added': comparison['only_in_config2'],
        'changed': comparison['different'],
        'removed': comparison['only_in_config1'],
    }

    # Filter out empty sections
    return {k: v for k, v in diff.items() if v}


def save_differences(differences: Dict[str, Any], path: Union[str, Path]):
    """
    Save configuration differences to file.

    Args:
        differences: Diff dictionary from compute_config_diff
        path: Output file path (supports .yaml, .yml, .json)
    """
    path = Path(path)

    if path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(path, 'w') as f:
            yaml.dump(differences, f, default_flow_style=False, sort_keys=False)
    elif path.suffix == ".json":
        import json
        with open(path, 'w') as f:
            json.dump(differences, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def validate_config(config: MAIUVFConfig) -> List[str]:
    """
    Validate configuration and return list of errors (strict validation).

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    return config.validate()


def merge_configs(base: MAIUVFConfig, override: MAIUVFConfig) -> MAIUVFConfig:
    """
    Merge two configurations (override takes precedence).

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    base_dict = base.to_dict()
    override_dict = override.to_dict()

    # Recursively merge
    def deep_merge(base_dict, override_dict):
        result = base_dict.copy()
        for key, value in override_dict.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    merged_dict = deep_merge(base_dict, override_dict)
    return MAIUVFConfig.from_dict(merged_dict)


def config_to_cli_args(config: MAIUVFConfig) -> List[str]:
    """
    Convert configuration to CLI arguments list.

    Useful for reproducing experiments with pure CLI.

    Args:
        config: MAIUVFConfig instance

    Returns:
        List of CLI argument strings (e.g., ["--epochs", "100", "--batch-size", "8192"])
    """
    flat = {
        'input_mesh': config.paths.input_mesh,
        'texture': config.paths.texture,
        'output_dir': config.paths.output_dir,
        'num_samples': config.baking.num_samples,
        'chart_mode': config.baking.chart_mode,
        'epochs': config.training.epochs,
        'batch_size': config.training.batch_size,
        'lr': config.training.learning_rate,
        'device': config.training.device,
        'encoder_type': config.model.encoder_type,
        'activation': config.model.activation,
        'hidden_dim': config.model.hidden_dim,
        'num_layers': config.model.num_layers,
        'positional_enc_freqs': config.model.positional_encoding_freqs,
        'hash_lr': config.model.hash_grid.learning_rate,
        'hash_num_levels': config.model.hash_grid.num_levels,
        'hash_features_per_level': config.model.hash_grid.features_per_level,
        'hash_log2_size': config.model.hash_grid.log2_size,
        'hash_base_res': config.model.hash_grid.base_resolution,
        'hash_max_res': config.model.hash_grid.max_resolution,
        'hash_cuda_backend': config.model.hash_grid.cuda_backend,
        'hash_weight_decay': config.model.hash_grid.weight_decay,
        'mlp_weight_decay': config.model.mlp.weight_decay,
        'metric_weight': config.loss.weights.metric,
        'anchor_weight': config.loss.weights.anchor,
        'cls_weight': config.loss.weights.classification,
        'com_weight': config.loss.weights.centroid,
        'unified_weight': config.loss.weights.unified,
        'loss_schedule': config.loss.schedule.strategy,
        'phase_a_epochs': config.loss.schedule.phase_a_epochs,
        'target_metric_weight': config.loss.schedule.target_weights.metric,
        'target_anchor_weight': config.loss.schedule.target_weights.anchor,
        'target_cls_weight': config.loss.schedule.target_weights.classification,
        'schedule_ramp': config.loss.schedule.ramp,
        'cls_cutoff_epoch': config.loss.schedule.classification_cutoff.epoch,
        'cls_cutoff_value': config.loss.schedule.classification_cutoff.value,
        'keep_anchor_constant': config.loss.schedule.keep_anchor_constant,
        'unified_num_neighbors': config.loss.schedule.unified.num_neighbors,
        'unified_epsilon': config.loss.schedule.unified.epsilon,
        'use_dynamic_sampling': config.dynamic_sampling.enabled,
        'virtual_epoch_size': config.dynamic_sampling.virtual_epoch_size,
        'sigma_ratio': config.dynamic_sampling.sigma_ratio,
        'target_faces': config.rendering.target_faces,
        'render_mode': config.rendering.mode,
        'render_resolution': config.rendering.resolution,
        'quick_test': config.system.quick_test,
    }

    cli_args = []
    for key, value in flat.items():
        if value is not None:
            # Convert underscores to hyphens for CLI
            cli_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    cli_args.append(f"--{cli_key}")
            elif isinstance(value, list):
                cli_args.append(f"--{cli_key}")
                cli_args.extend(str(v) for v in value)
            else:
                cli_args.append(f"--{cli_key}")
                cli_args.append(str(value))

    return cli_args
