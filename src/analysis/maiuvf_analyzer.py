"""
MA-IUVF Analyzer - Shared infrastructure for all experiments

Provides common utilities for:
- Model loading and prediction
- Jacobian computation via autograd
- Surface sampling
- Chart boundary detection
- Thickness computation
- Visualization generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import logging
import json
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

from ..inference.metric_aligned_iuv_inference import MetricAlignedIUVInference
from ..training.metric_aligned_iuv_losses import compute_uv_jacobian
from ..data.metric_aligned_iuv_baker import MetricAlignedIUVBaker


@dataclass
class AnalysisResult:
    """Standardized output format for all experiments"""
    experiment_name: str
    data: np.ndarray  # Numerical results
    metadata: Dict[str, Any]  # Experiment metadata
    figures: List[str]  # Paths to generated figures

    def save(self, output_dir: str):
        """Save results to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save data as CSV
        import pandas as pd
        df = pd.DataFrame(self.data)
        df.to_csv(output_dir / "data.csv", index=False)

        # Save metadata as JSON
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved results to {output_dir}")


class MAIUVFAnalyzer:
    """
    Shared analyzer for MA-IUVF experiments

    Provides common utilities for:
    - Model loading and inference
    - Jacobian computation
    - Surface sampling
    - Chart boundary analysis
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize analyzer

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device for inference ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing MA-IUVF analyzer on {self.device}")

        # Load model
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.inference = MetricAlignedIUVInference(
            checkpoint_path=checkpoint_path,
            device=device
        )
        self.model = self.inference.model

        # Extract metadata
        self.baker_metadata = self.inference.metadata.get('baker_metadata', {})
        self.num_charts = self.baker_metadata.get('num_charts', 8)
        self.checkpoint_path = Path(checkpoint_path)

        logger.info(f"Model loaded: {self.num_charts} charts")

    def get_network_outputs(
        self,
        positions: np.ndarray,
        return_probs: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get network outputs for 3D positions

        Args:
            positions: [N, 3] 3D positions
            return_probs: Whether to return softmax probabilities

        Returns:
            Dict with keys:
                - 'logits': [N, C] classification logits
                - 'uv_preds': [N, C, 2] UV predictions for all charts
                - 'probs': [N, C] softmax probabilities (if return_probs=True)
                - 'chart_ids': [N] predicted chart IDs (argmax)
                - 'selected_uvs': [N, 2] UV predictions for selected charts
        """
        # Get predictions from inference API
        logits, uv_preds = self.inference.predict(positions)

        # Compute predicted chart IDs
        chart_ids = logits.argmax(axis=-1)

        # Gather selected UVs
        B = len(positions)
        range_indices = np.arange(B)
        selected_uvs = uv_preds[range_indices, chart_ids]

        result = {
            'logits': logits,
            'uv_preds': uv_preds,
            'chart_ids': chart_ids,
            'selected_uvs': selected_uvs
        }

        if return_probs:
            # Compute softmax probabilities
            logits_tensor = torch.from_numpy(logits)
            probs = F.softmax(logits_tensor, dim=-1).numpy()
            result['probs'] = probs

        return result

    def compute_jacobians(
        self,
        positions: np.ndarray,
        chart_ids: np.ndarray,
        batch_size: int = 512
    ) -> np.ndarray:
        """
        Compute UV Jacobians via autograd

        Args:
            positions: [N, 3] 3D positions
            chart_ids: [N] chart IDs for each position
            batch_size: Batch size for processing

        Returns:
            jacobians: [N, 2, 3] UV Jacobian matrices
        """
        device = self.device
        n_points = len(positions)

        jacobians_list = []

        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            batch_positions = positions[i:batch_end]
            batch_chart_ids = chart_ids[i:batch_end]

            # Convert to tensors
            pos_tensor = torch.from_numpy(batch_positions).float().to(device)
            chart_tensor = torch.from_numpy(batch_chart_ids).long().to(device)

            # Enable gradients
            pos_tensor = pos_tensor.clone().detach().requires_grad_(True)

            # Get model predictions
            with torch.set_grad_enabled(True):
                model_output = self.model(pos_tensor)

                # Gather UV predictions for selected charts
                B = pos_tensor.shape[0]
                range_indices = torch.arange(B, device=device)
                selected_uv = model_output.uv_preds[range_indices, chart_tensor]

                # Compute Jacobian
                jacobian = compute_uv_jacobian(selected_uv, pos_tensor)

            # Move to CPU and store
            jacobians_list.append(jacobian.detach().cpu().numpy())

            # Clear gradients to free memory
            del pos_tensor, selected_uv, jacobian
            torch.cuda.empty_cache()

        # Concatenate all batches
        jacobians = np.concatenate(jacobians_list, axis=0)

        return jacobians

    def compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute information entropy from probability distribution

        Args:
            probs: [N, C] probability distributions (sum to 1)

        Returns:
            entropy: [N] entropy values H = -sum(p * log(p))
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        probs = np.clip(probs, eps, 1.0)

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs), axis=-1)

        return entropy

    def sample_mesh_surface(
        self,
        mesh: trimesh.Trimesh,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points uniformly on mesh surface

        Args:
            mesh: Input mesh
            num_points: Number of points to sample

        Returns:
            positions: [N, 3] sampled positions
            face_indices: [N] face indices for each point
        """
        # Sample using trimesh (method on mesh object)
        positions = mesh.sample(count=num_points, return_index=True)
        if len(positions) == 2:
            positions, face_indices = positions
        else:
            positions = positions
            face_indices = np.zeros(len(positions), dtype=int)

        return positions, face_indices

    def get_chart_boundaries(self) -> List[Dict]:
        """
        Extract chart boundary information from baker

        Returns:
            List of boundary edges with metadata
        """
        if 'uv_seams' not in self.baker_metadata:
            logger.warning("No UV seams found in baker_metadata")
            return []

        uv_seams = self.baker_metadata['uv_seams']

        boundaries = []
        for seam in uv_seams:
            boundaries.append({
                'face1': seam[0],
                'face2': seam[1],
                'edge': seam[2] if len(seam) > 2 else None
            })

        return boundaries

    def compute_thickness(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        mesh: trimesh.Trimesh,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Compute local thickness via ray casting

        Args:
            vertices: [N, 3] vertex positions
            normals: [N, 3] vertex normals
            mesh: Mesh for ray intersection
            epsilon: Small offset to avoid self-intersection

        Returns:
            thickness: [N] thickness values
        """
        n_vertices = len(vertices)
        thickness = np.zeros(n_vertices)

        # Cast rays in normal direction
        ray_origins = vertices + normals * epsilon
        ray_directions = normals

        # Find intersections
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        # Compute thickness for each vertex
        for i in range(n_vertices):
            if i < len(locations):
                # Thickness = distance to first intersection
                thickness[i] = np.linalg.norm(locations[i] - vertices[i])
            else:
                # No intersection found, use bbox diagonal as max
                thickness[i] = mesh.bounding_box.extents.max()

        return thickness

    def compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient and p-value

        Args:
            x: First variable
            y: Second variable

        Returns:
            r: Correlation coefficient
            p: P-value
        """
        r, p = pearsonr(x, y)
        return r, p

    def load_mesh_from_checkpoint(self) -> trimesh.Trimesh:
        """
        Load mesh from checkpoint metadata

        Returns:
            Loaded mesh
        """
        # Get mesh path from metadata
        mesh_path = self.baker_metadata.get('mesh_path')
        if not mesh_path:
            raise ValueError("No mesh_path found in baker_metadata")

        # Load mesh
        logger.info(f"Loading mesh: {mesh_path}")
        mesh = trimesh.load(mesh_path)

        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]

        return mesh

    def get_deformation_energy(
        self,
        jacobians: np.ndarray
    ) -> np.ndarray:
        """
        Compute deformation energy from Jacobians

        Args:
            jacobians: [N, 2, 3] Jacobian matrices

        Returns:
            energy: [N] deformation energy values (Dirichlet energy)
        """
        # Compute Frobenius norm squared
        # energy = ||J||_F² = sum(σ_i²) where σ are singular values
        frobenius_norm_sq = np.sum(jacobians ** 2, axis=(1, 2))

        return frobenius_norm_sq

    def get_normal_derivative(
        self,
        jacobians: np.ndarray,
        normals: np.ndarray
    ) -> np.ndarray:
        """
        Compute normal directional derivative

        Args:
            jacobians: [N, 2, 3] Jacobian matrices
            normals: [N, 3] normal vectors

        Returns:
            D_normal: [N] directional derivative magnitude ||J @ n||_2
        """
        # Compute J @ n for each point
        # [N, 2, 3] @ [N, 3, 1] = [N, 2, 1]
        normal_derivs = np.einsum('nij,nj->ni', jacobians, normals)

        # Compute L2 norm
        D_normal = np.linalg.norm(normal_derivs, axis=1)

        return D_normal

    def create_output_dir(self, experiment_name: str) -> Path:
        """
        Create output directory for experiment

        Args:
            experiment_name: Name of experiment

        Returns:
            Path to output directory
        """
        # Create outputs directory
        outputs_dir = Path("outputs/maiuvf_analysis")
        experiment_dir = outputs_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        return experiment_dir

    def save_results(
        self,
        result: AnalysisResult,
        output_dir: str
    ):
        """
        Save analysis results to disk

        Args:
            result: AnalysisResult to save
            output_dir: Output directory path
        """
        result.save(output_dir)
        logger.info(f"Results saved to {output_dir}")
