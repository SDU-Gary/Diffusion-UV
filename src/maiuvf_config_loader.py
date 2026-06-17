"""
MA-IUVF Configuration Loader

Handles loading YAML configurations with CLI parameter override.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
from maiuvf_config import MAIUVFConfig


class MAIUVFConfigLoader:
    """Load and merge YAML configuration with CLI arguments."""

    def __init__(self):
        self.config: Optional[MAIUVFConfig] = None
        self.cli_args: Optional[argparse.Namespace] = None

    def load_config(self, config_path: Union[str, Path]) -> MAIUVFConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config = MAIUVFConfig.from_yaml(config_path)

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

        return self.config

    def apply_cli_overrides(self, cli_args: argparse.Namespace) -> MAIUVFConfig:
        """Apply CLI argument overrides to configuration."""
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config first.")

        self.cli_args = cli_args

        # Create a mutable copy of the config
        config_dict = self.config.to_dict()

        # Apply overrides based on CLI arguments
        # Only override if the CLI argument was explicitly provided (not default)

        # Paths
        if hasattr(cli_args, 'input_mesh') and cli_args.input_mesh:
            config_dict['paths']['input_mesh'] = cli_args.input_mesh
        if hasattr(cli_args, 'texture') and cli_args.texture:
            config_dict['paths']['texture'] = cli_args.texture
        if hasattr(cli_args, 'output_dir') and cli_args.output_dir:
            config_dict['paths']['output_dir'] = cli_args.output_dir

        # Baking
        if hasattr(cli_args, 'num_samples') and cli_args.num_samples:
            config_dict['baking']['num_samples'] = cli_args.num_samples
        if hasattr(cli_args, 'chart_mode') and cli_args.chart_mode:
            config_dict['baking']['chart_mode'] = cli_args.chart_mode

        # Training
        if hasattr(cli_args, 'epochs') and cli_args.epochs:
            config_dict['training']['epochs'] = cli_args.epochs
        if hasattr(cli_args, 'batch_size') and cli_args.batch_size:
            config_dict['training']['batch_size'] = cli_args.batch_size
        if hasattr(cli_args, 'lr') and cli_args.lr:
            config_dict['training']['learning_rate'] = cli_args.lr

        # Model
        if hasattr(cli_args, 'encoder_type') and cli_args.encoder_type:
            config_dict['model']['encoder_type'] = cli_args.encoder_type
        if hasattr(cli_args, 'activation') and cli_args.activation:
            config_dict['model']['activation'] = cli_args.activation
        if hasattr(cli_args, 'hidden_dim') and cli_args.hidden_dim:
            config_dict['model']['hidden_dim'] = cli_args.hidden_dim
        if hasattr(cli_args, 'num_layers') and cli_args.num_layers:
            config_dict['model']['num_layers'] = cli_args.num_layers
        if hasattr(cli_args, 'positional_enc_freqs') and cli_args.positional_enc_freqs:
            config_dict['model']['positional_encoding_freqs'] = cli_args.positional_enc_freqs

        # Hash grid
        if hasattr(cli_args, 'hash_lr') and cli_args.hash_lr is not None:
            config_dict['model']['hash_grid']['learning_rate'] = cli_args.hash_lr
        if hasattr(cli_args, 'hash_num_levels') and cli_args.hash_num_levels:
            config_dict['model']['hash_grid']['num_levels'] = cli_args.hash_num_levels
        if hasattr(cli_args, 'hash_features_per_level') and cli_args.hash_features_per_level:
            config_dict['model']['hash_grid']['features_per_level'] = cli_args.hash_features_per_level
        if hasattr(cli_args, 'hash_log2_size') and cli_args.hash_log2_size:
            config_dict['model']['hash_grid']['log2_size'] = cli_args.hash_log2_size
        if hasattr(cli_args, 'hash_base_res') and cli_args.hash_base_res:
            config_dict['model']['hash_grid']['base_resolution'] = cli_args.hash_base_res
        if hasattr(cli_args, 'hash_max_res') and cli_args.hash_max_res:
            config_dict['model']['hash_grid']['max_resolution'] = cli_args.hash_max_res
        if hasattr(cli_args, 'hash_cuda_backend') and cli_args.hash_cuda_backend:
            config_dict['model']['hash_grid']['cuda_backend'] = cli_args.hash_cuda_backend
        if hasattr(cli_args, 'hash_weight_decay') and cli_args.hash_weight_decay:
            config_dict['model']['hash_grid']['weight_decay'] = cli_args.hash_weight_decay
        if hasattr(cli_args, 'mlp_weight_decay') and cli_args.mlp_weight_decay:
            config_dict['model']['mlp']['weight_decay'] = cli_args.mlp_weight_decay

        # Loss weights
        if hasattr(cli_args, 'metric_weight') and cli_args.metric_weight:
            config_dict['loss']['weights']['metric'] = cli_args.metric_weight
        if hasattr(cli_args, 'anchor_weight') and cli_args.anchor_weight:
            config_dict['loss']['weights']['anchor'] = cli_args.anchor_weight
        if hasattr(cli_args, 'cls_weight') and cli_args.cls_weight:
            config_dict['loss']['weights']['classification'] = cli_args.cls_weight
        if hasattr(cli_args, 'com_weight') and cli_args.com_weight:
            config_dict['loss']['weights']['centroid'] = cli_args.com_weight
        if hasattr(cli_args, 'unified_weight') and cli_args.unified_weight:
            config_dict['loss']['weights']['unified'] = cli_args.unified_weight

        # Loss schedule
        if hasattr(cli_args, 'loss_schedule') and cli_args.loss_schedule:
            config_dict['loss']['schedule']['strategy'] = cli_args.loss_schedule
        if hasattr(cli_args, 'phase_a_epochs') and cli_args.phase_a_epochs:
            config_dict['loss']['schedule']['phase_a_epochs'] = cli_args.phase_a_epochs
        if hasattr(cli_args, 'target_metric_weight') and cli_args.target_metric_weight:
            config_dict['loss']['schedule']['target_weights']['metric'] = cli_args.target_metric_weight
        if hasattr(cli_args, 'target_anchor_weight') and cli_args.target_anchor_weight:
            config_dict['loss']['schedule']['target_weights']['anchor'] = cli_args.target_anchor_weight
        if hasattr(cli_args, 'target_cls_weight') and cli_args.target_cls_weight:
            config_dict['loss']['schedule']['target_weights']['classification'] = cli_args.target_cls_weight
        if hasattr(cli_args, 'schedule_ramp') and cli_args.schedule_ramp:
            config_dict['loss']['schedule']['ramp'] = cli_args.schedule_ramp

        # Classification cutoff
        if hasattr(cli_args, 'cls_cutoff_epoch') and cli_args.cls_cutoff_epoch:
            config_dict['loss']['schedule']['classification_cutoff']['epoch'] = cli_args.cls_cutoff_epoch
        if hasattr(cli_args, 'cls_cutoff_value') and cli_args.cls_cutoff_value:
            config_dict['loss']['schedule']['classification_cutoff']['value'] = cli_args.cls_cutoff_value

        # Anchor constant
        if hasattr(cli_args, 'keep_anchor_constant'):
            config_dict['loss']['schedule']['keep_anchor_constant'] = cli_args.keep_anchor_constant

        # Unified loss
        if hasattr(cli_args, 'unified_num_neighbors') and cli_args.unified_num_neighbors:
            config_dict['loss']['schedule']['unified']['num_neighbors'] = cli_args.unified_num_neighbors
        if hasattr(cli_args, 'unified_epsilon') and cli_args.unified_epsilon:
            config_dict['loss']['schedule']['unified']['epsilon'] = cli_args.unified_epsilon

        # Dynamic sampling
        if hasattr(cli_args, 'use_dynamic_sampling'):
            config_dict['dynamic_sampling']['enabled'] = cli_args.use_dynamic_sampling
        if hasattr(cli_args, 'virtual_epoch_size') and cli_args.virtual_epoch_size:
            config_dict['dynamic_sampling']['virtual_epoch_size'] = cli_args.virtual_epoch_size
        if hasattr(cli_args, 'sigma_ratio') and cli_args.sigma_ratio:
            config_dict['dynamic_sampling']['sigma_ratio'] = cli_args.sigma_ratio

        # Rendering
        if hasattr(cli_args, 'target_faces') and cli_args.target_faces:
            config_dict['rendering']['target_faces'] = cli_args.target_faces
        if hasattr(cli_args, 'render_mode') and cli_args.render_mode:
            config_dict['rendering']['mode'] = cli_args.render_mode
        if hasattr(cli_args, 'render_resolution') and cli_args.render_resolution:
            config_dict['rendering']['resolution'] = cli_args.render_resolution

        # System
        if hasattr(cli_args, 'device') and cli_args.device:
            config_dict['system']['device'] = cli_args.device
        if hasattr(cli_args, 'quick_test'):
            config_dict['system']['quick_test'] = cli_args.quick_test

        # Reconstruct configuration
        self.config = MAIUVFConfig.from_dict(config_dict)

        return self.config

    def get_flat_config(self) -> Dict[str, Any]:
        """Get flat configuration dictionary compatible with existing functions."""
        if self.config is None:
            raise ValueError("No configuration loaded")

        cfg = self.config

        return {
            # Paths
            'input_mesh': cfg.paths.input_mesh,
            'texture': cfg.paths.texture,
            'output_dir': cfg.paths.output_dir,

            # Baking
            'num_samples': cfg.baking.num_samples,
            'chart_mode': cfg.baking.chart_mode,

            # Training
            'epochs': cfg.training.epochs,
            'batch_size': cfg.training.batch_size,
            'lr': cfg.training.learning_rate,
            'device': cfg.training.device,

            # Model
            'encoder_type': cfg.model.encoder_type,
            'activation': cfg.model.activation,
            'hidden_dim': cfg.model.hidden_dim,
            'num_layers': cfg.model.num_layers,
            'positional_enc_freqs': cfg.model.positional_encoding_freqs,

            # Hash grid
            'hash_lr': cfg.model.hash_grid.learning_rate,
            'hash_num_levels': cfg.model.hash_grid.num_levels,
            'hash_features_per_level': cfg.model.hash_grid.features_per_level,
            'hash_log2_size': cfg.model.hash_grid.log2_size,
            'hash_base_res': cfg.model.hash_grid.base_resolution,
            'hash_max_res': cfg.model.hash_grid.max_resolution,
            'hash_cuda_backend': cfg.model.hash_grid.cuda_backend,
            'hash_weight_decay': cfg.model.hash_grid.weight_decay,
            'mlp_weight_decay': cfg.model.mlp.weight_decay,

            # Loss weights
            'metric_weight': cfg.loss.weights.metric,
            'anchor_weight': cfg.loss.weights.anchor,
            'cls_weight': cfg.loss.weights.classification,
            'com_weight': cfg.loss.weights.centroid,
            'unified_weight': cfg.loss.weights.unified,

            # Loss schedule
            'loss_schedule': cfg.loss.schedule.strategy,
            'phase_a_epochs': cfg.loss.schedule.phase_a_epochs,
            'target_metric_weight': cfg.loss.schedule.target_weights.metric,
            'target_anchor_weight': cfg.loss.schedule.target_weights.anchor,
            'target_cls_weight': cfg.loss.schedule.target_weights.classification,
            'schedule_ramp': cfg.loss.schedule.ramp,
            'cls_cutoff_epoch': cfg.loss.schedule.classification_cutoff.epoch,
            'cls_cutoff_value': cfg.loss.schedule.classification_cutoff.value,
            'keep_anchor_constant': cfg.loss.schedule.keep_anchor_constant,
            'unified_num_neighbors': cfg.loss.schedule.unified.num_neighbors,
            'unified_epsilon': cfg.loss.schedule.unified.epsilon,

            # Dynamic sampling
            'use_dynamic_sampling': cfg.dynamic_sampling.enabled,
            'virtual_epoch_size': cfg.dynamic_sampling.virtual_epoch_size,
            'sigma_ratio': cfg.dynamic_sampling.sigma_ratio,

            # Rendering
            'target_faces': cfg.rendering.target_faces,
            'render_mode': cfg.rendering.mode,
            'render_resolution': cfg.rendering.resolution,

            # System
            'device': cfg.system.device,
            'quick_test': cfg.system.quick_test,
            'seed': cfg.system.seed,
        }
