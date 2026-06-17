"""
SDF Network Validation Script

This script validates the SDF network using three metrics:

1. Surface Cosine Similarity: Compare SDF normals with mesh normals on surface
   - Target: > 0.99
   - Purpose: Verify SDF normals match ground truth on surface

2. Eikonal Property Verification: Check ||∇SDF|| = 1 everywhere
   - Target: < 1e-3
   - Purpose: Verify gradient magnitude constraint

3. Vector Field Slice Visualization: Visual inspection of normal field
   - Target: Smooth flow without vortices or discontinuities
   - Purpose: Visual validation of normal field smoothness
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
import sys

# Add project root to path
sys.path.append('/home/kyrie/Diffusion-UV')

from src.models.sdf_network import SDFNetwork
from src.data.gpu_constant_baker import load_mesh_constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_surface_cosine_similarity(
    sdf_net: SDFNetwork,
    mesh_constants_path: str,
    num_samples: int = 100000,
    device: str = "cuda",
) -> float:
    """
    Compute cosine similarity between SDF normals and mesh normals on surface

    Cosine similarity = n_sdf · n_mesh (should be close to 1.0)

    Args:
        sdf_net: SDF network
        mesh_constants_path: Path to mesh constants file
        num_samples: Number of surface samples
        device: Computing device

    Returns:
        mean_cosine_sim: Mean cosine similarity (target: > 0.99)
    """
    logger.info("Validating surface cosine similarity...")

    # Load mesh constants
    constants, metadata = load_mesh_constants(mesh_constants_path, map_location=device)
    face_vertices = constants["face_vertices"]
    face_normals = constants["face_normals"]
    face_probs = constants["face_probs"]

    # Normalize face probabilities
    if face_probs.sum() <= 0:
        raise ValueError("face_probs sum must be > 0")
    face_probs = face_probs / face_probs.sum()

    # Sample surface points
    face_idx = torch.multinomial(face_probs, num_samples, replacement=True)
    sel_verts = face_vertices[face_idx]  # [B, 3, 3]
    sel_normals = face_normals[face_idx]  # [B, 3, 3]

    # Barycentric interpolation
    u = torch.rand(num_samples, device=device)
    v = torch.rand(num_samples, device=device)
    is_over = (u + v) > 1.0
    u = torch.where(is_over, 1.0 - u, u)
    v = torch.where(is_over, 1.0 - v, v)
    w = 1.0 - u - v
    bary = torch.stack([u, v, w], dim=-1)  # [B, 3]

    # Interpolate vertex positions and normals
    surface_pos = torch.bmm(bary.unsqueeze(1), sel_verts).squeeze(1)  # [B, 3]
    mesh_normals = torch.bmm(bary.unsqueeze(1), sel_normals).squeeze(1)  # [B, 3]
    mesh_normals = torch.nn.functional.normalize(mesh_normals, dim=-1, eps=1e-6)

    # Get SDF normals via autograd
    sdf_net.eval()
    surface_pos.requires_grad_(True)

    sdf_vals = sdf_net(surface_pos)
    sdf_normals = torch.autograd.grad(
        outputs=sdf_vals.sum(),
        inputs=surface_pos,
        create_graph=False,
    )[0]

    sdf_normals = torch.nn.functional.normalize(sdf_normals, dim=-1, eps=1e-6)

    # Cosine similarity
    cosine_sim = torch.sum(sdf_normals * mesh_normals, dim=-1)  # [B]
    mean_cosine_sim = cosine_sim.mean().item()
    std_cosine_sim = cosine_sim.std().item()
    min_cosine_sim = cosine_sim.min().item()

    logger.info(f"  Surface Cosine Similarity: {mean_cosine_sim:.4f} ± {std_cosine_sim:.4f}")
    logger.info(f"  Min cosine similarity: {min_cosine_sim:.4f}")

    return mean_cosine_sim


def validate_eikonal_property(
    sdf_net: SDFNetwork,
    bbox_min: tuple,
    bbox_max: tuple,
    num_samples: int = 10000,
    device: str = "cuda",
) -> float:
    """
    Verify Eikonal property: ||∇SDF|| = 1 everywhere in space

    Eikonal error = mean(| ||∇SDF|| - 1 |)

    Args:
        sdf_net: SDF network
        bbox_min: Bounding box min (x, y, z)
        bbox_max: Bounding box max (x, y, z)
        num_samples: Number of random samples
        device: Computing device

    Returns:
        eikonal_error: Mean | ||∇SDF|| - 1 | (target: < 1e-3)
    """
    logger.info("Validating Eikonal property...")

    # Sample points in bounding box
    bbox_min = torch.tensor(bbox_min, device=device)
    bbox_max = torch.tensor(bbox_max, device=device)

    samples = torch.rand(num_samples, 3, device=device)  # [0, 1]
    samples = bbox_min + samples * (bbox_max - bbox_min)

    # Compute gradients
    sdf_net.eval()
    samples.requires_grad_(True)

    sdf_vals = sdf_net(samples)
    grad = torch.autograd.grad(
        outputs=sdf_vals.sum(),
        inputs=samples,
        create_graph=False,
    )[0]

    grad_norm = torch.norm(grad, dim=-1)  # [B]
    eikonal_error = torch.mean(torch.abs(grad_norm - 1.0)).item()

    grad_mean = grad_norm.mean().item()
    grad_std = grad_norm.std().item()
    grad_min = grad_norm.min().item()
    grad_max = grad_norm.max().item()

    logger.info(f"  Gradient norm: {grad_mean:.6f} ± {grad_std:.6f} (min={grad_min:.6f}, max={grad_max:.6f})")
    logger.info(f"  Eikonal error: {eikonal_error:.6f}")

    return eikonal_error


def visualize_vector_field_slice(
    sdf_net: SDFNetwork,
    bbox_min: tuple,
    bbox_max: tuple,
    slice_axis: str = "z",
    slice_value: float = 0.5,
    grid_res: int = 32,
    output_path: str = "vector_field_slice.png",
    device: str = "cuda",
):
    """
    Visualize normal vector field on a 2D slice

    Creates a quiver plot showing normal vectors on a 2D slice through the volume.
    Should show smooth flow without vortices or discontinuities.

    Args:
        sdf_net: SDF network
        bbox_min: Bounding box min (x, y, z)
        bbox_max: Bounding box max (x, y, z)
        slice_axis: Axis to slice ("x", "y", or "z")
        slice_value: Relative position along slice axis [0, 1]
        grid_res: Resolution of 2D grid
        output_path: Path to save visualization
        device: Computing device
    """
    logger.info(f"Visualizing vector field slice (axis={slice_axis}, value={slice_value:.2f})...")

    # Create 2D grid
    bbox_min = torch.tensor(bbox_min, device=device)
    bbox_max = torch.tensor(bbox_max, device=device)

    if slice_axis == "z":
        x = torch.linspace(bbox_min[0], bbox_max[0], grid_res, device=device)
        y = torch.linspace(bbox_min[1], bbox_max[1], grid_res, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        Z = torch.full_like(X, slice_value * (bbox_max[2] - bbox_min[2]) + bbox_min[2])
    elif slice_axis == "y":
        x = torch.linspace(bbox_min[0], bbox_max[0], grid_res, device=device)
        z = torch.linspace(bbox_min[2], bbox_max[2], grid_res, device=device)
        X, Z = torch.meshgrid(x, z, indexing='ij')
        Y = torch.full_like(X, slice_value * (bbox_max[1] - bbox_min[1]) + bbox_min[1])
    elif slice_axis == "x":
        y = torch.linspace(bbox_min[1], bbox_max[1], grid_res, device=device)
        z = torch.linspace(bbox_min[2], bbox_max[2], grid_res, device=device)
        Y, Z = torch.meshgrid(y, z, indexing='ij')
        X = torch.full_like(Y, slice_value * (bbox_max[0] - bbox_min[0]) + bbox_min[0])
    else:
        raise ValueError(f"Unknown slice_axis: {slice_axis}")

    # Flatten for batch processing
    positions = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    # Compute normals
    sdf_net.eval()
    positions.requires_grad_(True)

    sdf_vals = sdf_net(positions)
    normals = torch.autograd.grad(
        outputs=sdf_vals.sum(),
        inputs=positions,
        create_graph=False,
    )[0]

    normals = normals.detach().cpu().numpy()
    positions = positions.detach().cpu().numpy()

    # Reshape for plotting
    if slice_axis == "z":
        U = normals[:, 0].reshape(grid_res, grid_res)
        V = normals[:, 1].reshape(grid_res, grid_res)
        X_plot = X.cpu().numpy().reshape(grid_res, grid_res)
        Y_plot = Y.cpu().numpy().reshape(grid_res, grid_res)
    elif slice_axis == "y":
        U = normals[:, 0].reshape(grid_res, grid_res)
        V = normals[:, 2].reshape(grid_res, grid_res)
        X_plot = X.cpu().numpy().reshape(grid_res, grid_res)
        Y_plot = Z.cpu().numpy().reshape(grid_res, grid_res)
    else:  # slice_axis == "x"
        U = normals[:, 1].reshape(grid_res, grid_res)
        V = normals[:, 2].reshape(grid_res, grid_res)
        X_plot = Y.cpu().numpy().reshape(grid_res, grid_res)
        Y_plot = Z.cpu().numpy().reshape(grid_res, grid_res)

    # Plot quiver
    plt.figure(figsize=(10, 10))
    plt.quiver(X_plot, Y_plot, U, V, angles='xy', scale_units='xy', scale=1, alpha=0.6)
    plt.xlabel(f"{slice_axis.upper()} axis")
    plt.ylabel(f"{slice_axis.upper()} axis")
    plt.title(f"SDF Normal Vector Field (slice {slice_axis}={slice_value:.2f})")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Vector field slice saved to {output_path}")


def run_sdf_validation(
    sdf_checkpoint: str,
    mesh_constants_path: str,
    output_dir: str,
    num_surface_samples: int = 100000,
    num_eikonal_samples: int = 10000,
    device: str = "cuda",
):
    """
    Run all three validation metrics

    Args:
        sdf_checkpoint: Path to SDF network checkpoint
        mesh_constants_path: Path to mesh constants file
        output_dir: Output directory for visualizations
        num_surface_samples: Number of surface samples for cosine similarity
        num_eikonal_samples: Number of samples for Eikonal validation
        device: Computing device

    Returns:
        validation_passed: True if all validation criteria met
    """
    logger.info("=" * 80)
    logger.info("SDF Network Validation")
    logger.info("=" * 80)
    logger.info(f"SDF checkpoint: {sdf_checkpoint}")
    logger.info(f"Mesh constants: {mesh_constants_path}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load SDF network
    logger.info("\nLoading SDF network...")
    sdf_net = SDFNetwork(
        num_levels=8,
        log2_hashmap_size=12,
        base_res=8,
        max_res=128,
        hidden_dim=32,
        num_layers=2,
        cuda_backend="torch",
    ).to(device)

    checkpoint = torch.load(sdf_checkpoint, map_location=device)
    sdf_net.load_state_dict(checkpoint['model_state_dict'])
    sdf_net.eval()

    logger.info(f"Loaded SDF network from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load mesh metadata
    constants, metadata = load_mesh_constants(mesh_constants_path, map_location=device)
    bbox_min = metadata["bbox_min"]
    bbox_max = metadata["bbox_max"]

    # Validation 1: Surface Cosine Similarity
    logger.info("\n" + "=" * 80)
    logger.info("Validation 1: Surface Cosine Similarity")
    logger.info("=" * 80)
    cosine_sim = validate_surface_cosine_similarity(
        sdf_net, mesh_constants_path, num_samples=num_surface_samples, device=device
    )

    # Validation 2: Eikonal Property
    logger.info("\n" + "=" * 80)
    logger.info("Validation 2: Eikonal Property")
    logger.info("=" * 80)
    eikonal_error = validate_eikonal_property(
        sdf_net, bbox_min, bbox_max, num_samples=num_eikonal_samples, device=device
    )

    # Validation 3: Vector Field Visualization
    logger.info("\n" + "=" * 80)
    logger.info("Validation 3: Vector Field Visualization")
    logger.info("=" * 80)

    # Create multiple slices
    for axis in ['x', 'y', 'z']:
        for value in [0.3, 0.5, 0.7]:
            output_path = str(Path(output_dir) / f"vector_field_{axis}_{value:.1f}.png")
            visualize_vector_field_slice(
                sdf_net, bbox_min, bbox_max,
                slice_axis=axis, slice_value=value, grid_res=32,
                output_path=output_path,
                device=device,
            )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SDF Validation Summary")
    logger.info("=" * 80)
    logger.info(f"  Cosine Similarity: {cosine_sim:.4f} (target: > 0.99)")
    logger.info(f"  Eikonal Error: {eikonal_error:.6f} (target: < 1e-3)")
    logger.info(f"  Vector Fields: {output_dir}/vector_field_*.png")

    validation_passed = cosine_sim > 0.99 and eikonal_error < 1e-3

    if validation_passed:
        logger.info("\n✅ SDF network validation PASSED")
        logger.info("   All criteria met:")
        logger.info(f"   - Cosine similarity > 0.99: {cosine_sim:.4f} ✅")
        logger.info(f"   - Eikonal error < 1e-3: {eikonal_error:.6f} ✅")
        logger.info("   - Vector fields visually smooth (manual inspection)")
    else:
        logger.warning("\n⚠️  SDF network validation needs improvement")
        if cosine_sim <= 0.99:
            logger.warning(f"   - Cosine similarity too low: {cosine_sim:.4f} (need > 0.99)")
        if eikonal_error >= 1e-3:
            logger.warning(f"   - Eikonal error too high: {eikonal_error:.6f} (need < 1e-3)")

    return validation_passed


def main():
    parser = argparse.ArgumentParser(description="Validate SDF network")
    parser.add_argument(
        "--sdf-checkpoint",
        type=str,
        required=True,
        help="Path to SDF network checkpoint"
    )
    parser.add_argument(
        "--mesh-constants",
        type=str,
        required=True,
        help="Path to mesh constants file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sdf_validation",
        help="Output directory"
    )
    parser.add_argument(
        "--num-surface-samples",
        type=int,
        default=100000,
        help="Number of surface samples"
    )
    parser.add_argument(
        "--num-eikonal-samples",
        type=int,
        default=10000,
        help="Number of Eikonal samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computing device"
    )

    args = parser.parse_args()

    # Run validation
    validation_passed = run_sdf_validation(
        sdf_checkpoint=args.sdf_checkpoint,
        mesh_constants_path=args.mesh_constants,
        output_dir=args.output_dir,
        num_surface_samples=args.num_surface_samples,
        num_eikonal_samples=args.num_eikonal_samples,
        device=args.device,
    )

    # Exit with appropriate code
    sys.exit(0 if validation_passed else 1)


if __name__ == "__main__":
    main()
