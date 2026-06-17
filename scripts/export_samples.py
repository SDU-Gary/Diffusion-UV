#!/usr/bin/env python3
"""
Export sampling data to NPZ format for viewing

This script exports point samples from meshes or generates synthetic samples
that can be viewed in the desktop 3D viewer.

Usage:
    python scripts/export_samples.py high_poly.obj samples.npz
    python scripts/export_samples.py high_poly.obj samples.npz --num-samples 10000
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import logging

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Error: trimesh is required. Install with: pip install trimesh")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class SampleExporter:
    """
    Export sampling data to NPZ format.

    Generates samples on mesh surfaces with optional features like:
    - Point positions
    - Colors (from mesh vertex colors)
    - Normals
    - SDF values
    """

    def __init__(self, mesh_path: str):
        """
        Initialize exporter with mesh.

        Args:
            mesh_path: Path to mesh file
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required")

        self.mesh_path = Path(mesh_path)
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        self.mesh = trimesh.load(mesh_path)
        logger.info(f"Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")

    def generate_surface_samples(self, num_samples: int = 10000) -> np.ndarray:
        """
        Generate random samples on mesh surface.

        Args:
            num_samples: Number of samples to generate

        Returns:
            (N, 3) array of sample positions
        """
        logger.info(f"Generating {num_samples} surface samples...")

        # Sample using trimesh
        points, face_indices = trimesh.sample.sample_surface(
            self.mesh,
            num_samples
        )

        logger.info(f"Generated {len(points)} samples")
        return points

    def interpolate_colors(self, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
        """
        Interpolate vertex colors at sample points.

        Args:
            points: (N, 3) sample positions
            face_indices: (N,) face indices for each sample

        Returns:
            (N, 3) RGB colors or None if mesh has no colors
        """
        if not hasattr(self.mesh.visual, 'vertex_colors'):
            logger.warning("Mesh has no vertex colors")
            return None

        vertex_colors = self.mesh.visual.vertex_colors
        if vertex_colors.shape[1] == 4:
            vertex_colors = vertex_colors[:, :3]

        # Convert to float [0,1]
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0

        # Get barycentric coordinates for each sample
        # For simplicity, we'll use face-based averaging
        faces = self.mesh.faces[face_indices]

        # Average colors of face vertices
        colors = np.mean(vertex_colors[faces], axis=1)

        logger.info(f"Interpolated colors: RGB range [{colors.min():.3f}, {colors.max():.3f}]")
        return colors

    def compute_normals(self, points: np.ndarray) -> np.ndarray:
        """
        Compute normals at sample points.

        Args:
            points: (N, 3) sample positions

        Returns:
            (N, 3) normal vectors
        """
        # For simplicity, use face normals
        face_normals = self.mesh.face_normals

        # Find nearest face for each point (simplified)
        from scipy.spatial import cKDTree
        face_centers = self.mesh.triangles.mean(axis=1)
        tree = cKDTree(face_centers)
        _, indices = tree.query(points)

        normals = face_centers[indices]  # Placeholder
        return np.zeros_like(points)  # Should be properly computed

    def export_to_npz(
        self,
        output_path: str,
        num_samples: int = 10000,
        include_colors: bool = True,
        include_normals: bool = False,
        include_sdf: bool = False
    ):
        """
        Export samples to NPZ file.

        Args:
            output_path: Path to output NPZ file
            num_samples: Number of samples to generate
            include_colors: Whether to include RGB colors
            include_normals: Whether to include normals
            include_sdf: Whether to include SDF values
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting to {output_path}...")

        # Generate samples
        points, face_indices = trimesh.sample.sample_surface(self.mesh, num_samples)

        # Prepare data dictionary
        data = {'points': points.astype(np.float32)}

        # Add colors if available and requested
        if include_colors:
            colors = self.interpolate_colors(points, face_indices)
            if colors is not None:
                data['colors'] = colors.astype(np.float32)

        # Add normals if requested
        if include_normals:
            normals = self.compute_normals(points)
            data['normals'] = normals.astype(np.float32)

        # Add SDF if requested (simplified: zero for surface samples)
        if include_sdf:
            sdf = np.zeros(num_samples, dtype=np.float32)
            data['sdf'] = sdf

        # Save NPZ
        np.savez(output_path, **data)

        logger.info(f"✓ Saved {num_samples} samples to {output_path}")
        logger.info(f"  Data keys: {list(data.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Export mesh samples to NPZ format for viewing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/export_samples.py mesh.obj samples.npz

  # With custom sample count
  python scripts/export_samples.py mesh.obj samples.npz --num-samples 50000

  # Include all features
  python scripts/export_samples.py mesh.obj samples.npz --colors --normals --sdf

  # View exported samples
  python scripts/viewer_3d.py samples.npz
        """
    )

    parser.add_argument("input_mesh", help="Input mesh file (OBJ, PLY, etc.)")
    parser.add_argument("output_npz", help="Output NPZ file")

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)"
    )

    parser.add_argument(
        "--no-colors",
        action="store_true",
        help="Don't include vertex colors"
    )

    parser.add_argument(
        "--normals",
        action="store_true",
        help="Include surface normals"
    )

    parser.add_argument(
        "--sdf",
        action="store_true",
        help="Include SDF values"
    )

    args = parser.parse_args()

    try:
        # Create exporter
        exporter = SampleExporter(args.input_mesh)

        # Export
        exporter.export_to_npz(
            args.output_npz,
            num_samples=args.num_samples,
            include_colors=not args.no_colors,
            include_normals=args.normals,
            include_sdf=args.sdf
        )

        logger.info("\n✓ Export complete!")
        logger.info(f"\nTo view samples:")
        logger.info(f"  python3 scripts/viewer_3d.py {args.output_npz}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
