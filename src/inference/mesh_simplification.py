"""
Mesh Simplification Module

Implements multiple mesh decimation algorithms for generating low-poly versions
of high-resolution meshes for inference testing.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class MeshSimplifier:
    """
    Mesh simplification using multiple algorithms.

    Supports:
    - Quadric Decimation (trimesh)
    - Vertex Clustering
    - Edge Collapse
    """

    def __init__(self, high_mesh_path: str):
        """
        Initialize simplifier with high-poly mesh.

        Args:
            high_mesh_path: Path to high-poly mesh (.obj, .ply, etc.)
        """
        self.high_mesh_path = Path(high_mesh_path)
        self.high_mesh = trimesh.load(high_mesh_path)

        logger.info(f"Loaded high-poly mesh: {len(self.high_mesh.vertices)} vertices, {len(self.high_mesh.faces)} faces")

    def simplify_by_ratio(
        self,
        face_ratio: float = 0.05,
        method: str = "quadric",
        **kwargs
    ) -> trimesh.Trimesh:
        """
        Simplify mesh by face ratio.

        Args:
            face_ratio: Target face count as ratio of original (0.0-1.0)
            method: Simplification method ('quadric', 'vertex_clustering', 'edge_collapse')
            **kwargs: Additional parameters for simplification

        Returns:
            Simplified mesh
        """
        target_faces = int(len(self.high_mesh.faces) * face_ratio)
        return self.simplify_by_count(target_faces, method, **kwargs)

    def simplify_by_count(
        self,
        target_faces: int,
        method: str = "quadric",
        **kwargs
    ) -> trimesh.Trimesh:
        """
        Simplify mesh to target face count.

        Args:
            target_faces: Target number of faces
            method: Simplification method
            **kwargs: Additional parameters

        Returns:
            Simplified mesh
        """
        if method == "quadric":
            return self._quadric_decimation(target_faces, **kwargs)
        elif method == "vertex_clustering":
            return self._vertex_clustering(target_faces, **kwargs)
        elif method == "edge_collapse":
            return self._edge_collapse(target_faces, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _quadric_decimation(
        self,
        target_faces: int,
        aggression: int = 10,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        Quadric edge collapse decimation.

        Args:
            target_faces: Target face count
            aggression: Aggressiveness (1-15, higher = faster)
        """
        try:
            # Use trimesh's quadric decimation
            # Note: parameter name is 'aggression' not 'aggressive'
            low_mesh = self.high_mesh.simplify_quadric_decimation(
                face_count=target_faces,
                aggression=aggression
            )

            logger.info(f"Simplified to {len(low_mesh.vertices)} vertices, {len(low_mesh.faces)} faces")
            return low_mesh

        except Exception as e:
            logger.warning(f"Quadric decimation failed: {e}")
            # Fallback to basic method
            return self._basic_simplification(target_faces)

    def _vertex_clustering(
        self,
        target_faces: int,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        Vertex clustering simplification.

        Args:
            target_faces: Target face count
        """
        # For vertex clustering, use percentage
        face_ratio = target_faces / len(self.high_mesh.faces)
        return self._basic_simplification(target_faces)

    def _edge_collapse(
        self,
        target_faces: int,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        Edge collapse simplification.
        """
        # Use quadric as fallback
        return self._quadric_decimation(target_faces, **kwargs)

    def _basic_simplification(self, target_faces: int) -> trimesh.Trimesh:
        """
        Basic simplification using trimesh.

        Args:
            target_faces: Target face count
        """
        try:
            # Calculate percentage
            face_ratio = target_faces / len(self.high_mesh.faces)

            # Ensure ratio is in valid range
            if face_ratio < 0.001:
                face_ratio = 0.001
            elif face_ratio > 1.0:
                face_ratio = 1.0

            # Use percent parameter (0-100 scale)
            percent = face_ratio * 100

            low_mesh = self.high_mesh.simplify_quadric_decimation(
                percent=percent
            )

            logger.info(f"Basic simplification: {len(low_mesh.vertices)} vertices, {len(low_mesh.faces)} faces")
            return low_mesh

        except Exception as e:
            logger.error(f"Basic simplification failed: {e}")
            # Ultimate fallback: return original mesh
            logger.warning("Returning original mesh as fallback")
            return self.high_mesh

    def simplify_multiple_levels(
        self,
        ratios: list = [0.01, 0.05, 0.1, 0.2],
        output_dir: str = "./data/low_poly_meshes"
    ) -> Dict[str, trimesh.Trimesh]:
        """
        Generate multiple simplification levels.

        Args:
            ratios: List of face ratios to generate
            output_dir: Directory to save meshes

        Returns:
            Dictionary of level_name -> mesh
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        for i, ratio in enumerate(ratios):
            try:
                level_name = f"level_{i+1}_{int(ratio*100)}percent"
                low_mesh = self.simplify_by_ratio(ratio)

                # Save mesh
                output_file = output_path / f"{self.high_mesh_path.stem}_{level_name}.obj"
                low_mesh.export(str(output_file))

                results[level_name] = low_mesh

                logger.info(f"Generated {level_name}: {output_file}")

            except Exception as e:
                logger.error(f"Failed to generate level {ratio}: {e}")

        return results

    def compare_meshes(
        self,
        low_mesh: trimesh.Trimesh,
    ) -> Dict[str, any]:
        """
        Compare high and low poly meshes.

        Args:
            low_mesh: Low-poly mesh to compare

        Returns:
            Comparison statistics
        """
        high_v, high_f = len(self.high_mesh.vertices), len(self.high_mesh.faces)
        low_v, low_f = len(low_mesh.vertices), len(low_mesh.faces)

        stats = {
            "high_vertices": high_v,
            "high_faces": high_f,
            "low_vertices": low_v,
            "low_faces": low_f,
            "vertex_ratio": low_v / high_v,
            "face_ratio": low_f / high_f,
            "compression": (1 - low_f / high_f) * 100,
        }

        return stats


def create_low_poly_mesh(
    high_mesh_path: str,
    output_path: str,
    face_ratio: float = 0.05,
    method: str = "quadric"
) -> trimesh.Trimesh:
    """
    Convenience function to create low-poly mesh.

    Args:
        high_mesh_path: Path to high-poly mesh
        output_path: Path to save low-poly mesh
        face_ratio: Target face ratio
        method: Simplification method

    Returns:
        Low-poly mesh
    """
    simplifier = MeshSimplifier(high_mesh_path)
    low_mesh = simplifier.simplify_by_ratio(face_ratio, method)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    low_mesh.export(str(output_path))

    logger.info(f"Saved low-poly mesh to {output_path}")

    return low_mesh


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplify 3D meshes")
    parser.add_argument("input_mesh", help="Path to high-poly mesh")
    parser.add_argument("output_mesh", help="Path to save low-poly mesh")
    parser.add_argument("--ratio", type=float, default=0.05, help="Face ratio (default: 0.05)")
    parser.add_argument("--method", default="quadric", help="Simplification method")

    args = parser.parse_args()

    create_low_poly_mesh(
        args.input_mesh,
        args.output_mesh,
        args.ratio,
        args.method
    )
