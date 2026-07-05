#!/usr/bin/env python3
"""
Diffusion-UV Demo Script
========================

This script demonstrates the core functionality of the Metric-Aligned Implicit UV Field (MA-IUVF) approach
for low-poly mesh coloring under shared texture constraints.

Usage:
    python demo.py [--mode {train,inference,render,all}]
"""

import argparse
import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import numpy
        import trimesh
        logger.info(f"✓ PyTorch {torch.__version__}")
        logger.info(f"✓ NumPy {numpy.__version__}")
        logger.info(f"✓ Trimesh {trimesh.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False


def demo_model_architecture():
    """Demonstrate the MA-IUVF model architecture."""
    logger.info("=" * 60)
    logger.info("Demo 1: MA-IUVF Model Architecture")
    logger.info("=" * 60)

    try:
        import torch
        from src.models.metric_aligned_iuv_field import create_model

        # Create a sample model
        model = create_model(
            num_charts=8,
            hidden_dim=128,
            num_layers=3,
            positional_encoding_freqs=6,
            encoder_type='fourier'
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model created successfully!")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Test forward pass
        batch_size = 1000
        positions = torch.randn(batch_size, 3)

        with torch.no_grad():
            output = model(positions)

        logger.info(f"Forward pass test:")
        logger.info(f"  Input shape: {positions.shape}")
        logger.info(f"  Chart logits shape: {output.logits.shape}")
        logger.info(f"  UV predictions shape: {output.uv_preds.shape}")

        return True

    except Exception as e:
        logger.error(f"Model architecture demo failed: {e}")
        return False


def demo_data_pipeline():
    """Demonstrate the data pipeline components."""
    logger.info("=" * 60)
    logger.info("Demo 2: Data Pipeline Components")
    logger.info("=" * 60)

    try:
        from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker
        from src.data.uv_chart_segmentation import UVChartSegmentation
        import trimesh

        # Create a simple test mesh
        logger.info("Creating test mesh...")
        vertices = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 1, 0], [0, 0, 1], [1, 0, 1]
        ]
        faces = [
            [0, 1, 2], [1, 3, 2],  # Bottom face
            [0, 1, 4], [1, 5, 4],  # Side face
            [2, 3, 4], [3, 5, 4]   # Side face
        ]
        uvs = [
            [0, 0], [1, 0], [0, 1],
            [1, 1], [0.5, 0], [1.5, 0]
        ]

        logger.info(f"Test mesh: {len(vertices)} vertices, {len(faces)} faces")
        logger.info(f"UV coordinates: {len(uvs)}")

        return True

    except Exception as e:
        logger.error(f"Data pipeline demo failed: {e}")
        return False


def demo_loss_computation():
    """Demonstrate the loss computation."""
    logger.info("=" * 60)
    logger.info("Demo 3: Metric Alignment Loss Computation")
    logger.info("=" * 60)

    try:
        import torch
        from src.training.metric_aligned_iuv_losses import (
            compute_metric_alignment_loss,
            compute_uv_jacobian,
            validate_jacobian_math
        )

        # Validate Jacobian computation math
        logger.info("Validating Jacobian computation...")
        if validate_jacobian_math():
            logger.info("✓ Jacobian computation validated successfully")
        else:
            logger.warning("⚠ Jacobian computation validation failed")

        # Test loss computation
        logger.info("\nTesting metric alignment loss...")

        # Create sample data
        batch_size = 100
        positions = torch.randn(batch_size, 3, requires_grad=True)
        uv_pred = torch.rand(batch_size, 2, requires_grad=True)
        uv_gt = torch.rand(batch_size, 2)

        # Compute Jacobians
        j_pred = compute_uv_jacobian(uv_pred, positions)
        logger.info(f"Predicted Jacobian shape: {j_pred.shape}")

        # Test metric alignment loss
        loss = compute_metric_alignment_loss(j_pred, j_pred)
        logger.info(f"Metric alignment loss (perfect match): {loss.item():.6f}")

        loss_noisy = compute_metric_alignment_loss(j_pred, j_pred + 0.1 * torch.randn_like(j_pred))
        logger.info(f"Metric alignment loss (with noise): {loss_noisy.item():.6f}")

        return True

    except Exception as e:
        logger.error(f"Loss computation demo failed: {e}")
        return False


def demo_training_simulation():
    """Demonstrate a simplified training simulation."""
    logger.info("=" * 60)
    logger.info("Demo 4: Simplified Training Simulation")
    logger.info("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from src.models.metric_aligned_iuv_field import create_model
        from src.training.metric_aligned_iuv_losses import compute_metric_aligned_iuv_loss

        # Create model
        logger.info("Creating MA-IUVF model...")
        model = create_model(
            num_charts=4,
            hidden_dim=64,
            num_layers=2,
            positional_encoding_freqs=4
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate training batch
        logger.info("\nSimulating training batch...")
        batch_size = 256
        positions = torch.randn(batch_size, 3)
        uv_gt = torch.rand(batch_size, 2)
        chart_ids = torch.randint(0, 4, (batch_size,))

        # Training step
        model.train()
        optimizer.zero_grad()

        output = model(positions)
        loss_dict = compute_metric_aligned_iuv_loss(
            output.uv_preds, output.logits, positions,
            uv_gt, chart_ids, num_charts=4
        )

        total_loss = loss_dict['total']
        total_loss.backward()
        optimizer.step()

        logger.info("Training step completed:")
        logger.info(f"  Total loss: {total_loss.item():.4f}")
        logger.info(f"  Metric alignment loss: {loss_dict['metric_alignment'].item():.4f}")
        logger.info(f"  UV anchor loss: {loss_dict['uv_anchor'].item():.4f}")
        logger.info(f"  Chart classification loss: {loss_dict['chart_classification'].item():.4f}")

        return True

    except Exception as e:
        logger.error(f"Training simulation demo failed: {e}")
        return False


def demo_inference_pipeline():
    """Demonstrate the inference pipeline."""
    logger.info("=" * 60)
    logger.info("Demo 5: Inference Pipeline")
    logger.info("=" * 60)

    try:
        import torch
        from src.models.metric_aligned_iuv_field import create_model

        logger.info("Creating model for inference...")
        model = create_model(
            num_charts=8,
            hidden_dim=128,
            num_layers=3,
            positional_encoding_freqs=6
        )
        model.eval()

        # Simulate inference on low-poly mesh
        num_vertices = 1000
        positions = torch.randn(num_vertices, 3)

        logger.info(f"\nRunning inference on {num_vertices} vertices...")

        with torch.no_grad():
            output = model(positions)

            # Chart selection via argmax
            chart_ids = output.logits.argmax(dim=-1)
            selected_uvs = torch.gather(
                output.uv_preds,
                1,
                chart_ids.unsqueeze(-1).expand(-1, 2)
            )

        logger.info("Inference completed:")
        logger.info(f"  Input vertices: {num_vertices}")
        logger.info(f"  Chart distribution: {torch.bincount(chart_ids).tolist()}")
        logger.info(f"  UV range: [{selected_uvs.min():.3f}, {selected_uvs.max():.3f}]")
        logger.info(f"  UV mean: {selected_uvs.mean(dim=0).tolist()}")

        return True

    except Exception as e:
        logger.error(f"Inference pipeline demo failed: {e}")
        return False


def run_demo(mode='all'):
    """Run the requested demo."""
    logger.info("\n" + "=" * 60)
    logger.info("Diffusion-UV Demo: Metric-Aligned Implicit UV Field")
    logger.info("=" * 60 + "\n")

    # Check dependencies
    if not check_dependencies():
        logger.error("Please install required dependencies:")
        logger.error("  pip install torch numpy trimesh PyOpenGL PyGLFW")
        return False

    # Run requested demos
    demos = {
        'model': demo_model_architecture,
        'data': demo_data_pipeline,
        'loss': demo_loss_computation,
        'training': demo_training_simulation,
        'inference': demo_inference_pipeline
    }

    results = {}

    if mode == 'all':
        for demo_name, demo_func in demos.items():
            try:
                results[demo_name] = demo_func()
                logger.info("")
            except Exception as e:
                logger.error(f"Demo {demo_name} failed: {e}")
                results[demo_name] = False
    else:
        if mode in demos:
            results[mode] = demos[mode]()
        else:
            logger.error(f"Unknown demo mode: {mode}")
            logger.info(f"Available modes: {', '.join(demos.keys())}, all")
            return False

    # Summary
    logger.info("=" * 60)
    logger.info("Demo Summary")
    logger.info("=" * 60)

    for demo_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{demo_name.title()}: {status}")

    all_passed = all(results.values())
    logger.info("\n" + ("=" * 60))
    if all_passed:
        logger.info("All demos completed successfully!")
    else:
        logger.warning("Some demos failed. Please check the errors above.")

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diffusion-UV Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Run all demos
  python demo.py --mode model       # Run only model architecture demo
  python demo.py --mode training    # Run only training simulation demo
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'model', 'data', 'loss', 'training', 'inference'],
        help='Demo mode to run'
    )

    args = parser.parse_args()

    success = run_demo(args.mode)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
