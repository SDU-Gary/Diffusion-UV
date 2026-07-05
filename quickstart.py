#!/usr/bin/env python3
"""
Diffusion-UV Quick Start Script
================================

This script provides a quick demonstration of the MA-IUVF pipeline for reproducible research.
It generates synthetic test data and demonstrates the core workflow.

Usage:
    python quickstart.py [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_synthetic_mesh(num_faces=1000):
    """Generate a synthetic mesh with UV coordinates for testing."""
    import numpy as np
    import trimesh

    logger.info(f"Generating synthetic mesh with {num_faces} faces...")

    # Create a sphere-like mesh
    mesh = trimesh.creation.icosphere(subdivision=4)

    # Add UV coordinates
    logger.info("Adding UV coordinates...")
    uv = trimesh.uv.unwrap_uv_coords(mesh, method='xatlas')

    logger.info(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh, uv


def run_maiuvf_pipeline(mesh, uv, output_dir, num_charts=8):
    """Run the MA-IUVF pipeline."""
    logger.info("=" * 60)
    logger.info("MA-IUVF Pipeline Quick Start")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': 'MA-IUVF',
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'num_charts': num_charts,
        'steps': []
    }

    try:
        # Step 1: Data preparation
        logger.info("\nStep 1: Data Preparation")
        logger.info("-" * 40)

        # Save mesh to temporary OBJ file
        temp_obj = output_dir / "test_mesh.obj"
        mesh.export(temp_obj)
        results['steps'].append({
            'name': 'data_preparation',
            'status': 'completed',
            'output_file': str(temp_obj)
        })
        logger.info(f"✓ Mesh saved to {temp_obj}")

        # Step 2: Model initialization
        logger.info("\nStep 2: Model Initialization")
        logger.info("-" * 40)

        import torch
        from src.models.metric_aligned_iuv_field import create_model

        model = create_model(
            num_charts=num_charts,
            hidden_dim=128,
            num_layers=3,
            positional_encoding_freqs=6,
            encoder_type='fourier'
        )

        total_params = sum(p.numel() for p in model.parameters())
        results['model_parameters'] = total_params
        results['steps'].append({
            'name': 'model_initialization',
            'status': 'completed',
            'parameters': total_params
        })

        logger.info(f"✓ Model created with {total_params:,} parameters")

        # Step 3: Training simulation
        logger.info("\nStep 3: Training Simulation")
        logger.info("-" * 40)

        from src.training.metric_aligned_iuv_losses import compute_metric_aligned_iuv_loss

        # Create synthetic training data
        batch_size = 1000
        num_iterations = 10

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        training_losses = []
        for iteration in range(num_iterations):
            # Generate synthetic batch
            positions = torch.randn(batch_size, 3)
            uv_gt = torch.rand(batch_size, 2)
            chart_ids = torch.randint(0, num_charts, (batch_size,))

            # Forward pass
            model.train()
            optimizer.zero_grad()
            output = model(positions)

            # Compute loss
            loss_dict = compute_metric_aligned_iuv_loss(
                output.uv_preds, output.logits, positions,
                uv_gt, chart_ids, num_charts=num_charts,
                metric_weight=1.0,
                anchor_weight=0.1,
                chart_weight=1.0
            )

            # Backward pass
            loss_dict['total'].backward()
            optimizer.step()

            training_losses.append(loss_dict['total'].item())

            if (iteration + 1) % 5 == 0:
                logger.info(f"  Iteration {iteration+1}/{num_iterations}: "
                          f"Loss = {loss_dict['total'].item():.4f}")

        results['steps'].append({
            'name': 'training_simulation',
            'status': 'completed',
            'iterations': num_iterations,
            'final_loss': training_losses[-1],
            'loss_reduction': training_losses[0] - training_losses[-1]
        })

        logger.info(f"✓ Training completed: {training_losses[0]:.4f} → {training_losses[-1]:.4f}")

        # Step 4: Inference
        logger.info("\nStep 4: Inference")
        logger.info("-" * 40)

        model.eval()
        with torch.no_grad():
            # Run inference on mesh vertices
            vertex_positions = torch.tensor(mesh.vertices, dtype=torch.float32)
            output = model(vertex_positions)

            # Chart selection
            chart_ids = output.logits.argmax(dim=-1)
            selected_uvs = torch.gather(
                output.uv_preds,
                1,
                chart_ids.unsqueeze(-1).expand(-1, 2)
            )

        chart_distribution = torch.bincount(chart_ids).tolist()
        results['steps'].append({
            'name': 'inference',
            'status': 'completed',
            'num_predictions': len(vertex_positions),
            'chart_distribution': chart_distribution
        })

        logger.info(f"✓ Inference completed for {len(vertex_positions)} vertices")
        logger.info(f"  Chart distribution: {chart_distribution}")

        # Step 5: Results summary
        logger.info("\nStep 5: Results Summary")
        logger.info("-" * 40)

        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Results saved to {results_file}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Execution Summary")
        logger.info("=" * 60)
        logger.info(f"Input mesh: {results['num_vertices']} vertices, {results['num_faces']} faces")
        logger.info(f"Model parameters: {results['model_parameters']:,}")
        logger.info(f"Training iterations: {results['steps'][2]['iterations']}")
        logger.info(f"Final loss: {results['steps'][2]['final_loss']:.4f}")
        logger.info(f"Loss reduction: {results['steps'][2]['loss_reduction']:.4f}")
        logger.info(f"Chart distribution: {results['steps'][3]['chart_distribution']}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diffusion-UV Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python quickstart.py --output-dir ./outputs/quickstart
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/quickstart',
        help='Output directory for results'
    )

    parser.add_argument(
        '--num-faces',
        type=int,
        default=1000,
        help='Number of faces for synthetic mesh'
    )

    parser.add_argument(
        '--num-charts',
        type=int,
        default=8,
        help='Number of UV charts'
    )

    args = parser.parse_args()

    # Generate synthetic mesh
    mesh, uv = generate_synthetic_mesh(args.num_faces)

    # Run pipeline
    success = run_maiuvf_pipeline(
        mesh, uv,
        output_dir=args.output_dir,
        num_charts=args.num_charts
    )

    if success:
        logger.info("\n✓ Quick start completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        return 0
    else:
        logger.error("\n✗ Quick start failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
