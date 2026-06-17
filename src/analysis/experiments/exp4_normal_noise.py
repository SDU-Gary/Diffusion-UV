"""
Experiment 4: Normal Gradient Noise Validation

Purpose: Test if full-space L_metric constraint wastes network capacity
on meaningless normal direction gradients.

Hypothesis: In ideal manifold parameterization, UV coordinates should not
change when moving along the normal direction (D_normal ≈ 0). If D_normal
shows significant non-zero values, it indicates the network is learning
unnecessary normal gradients.
"""

import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

from ..maiuvf_analyzer import MAIUVFAnalyzer, AnalysisResult
from ...inference.metric_aligned_iuv_inference import MetricAlignedIUVInference
from ...training.metric_aligned_iuv_losses import compute_uv_jacobian
from ..utils import (
    plot_histogram, plot_correlation_scatter, plot_mesh_colored,
    create_analysis_report, generate_statistics_table
)


def run_experiment4(
    analyzer: MAIUVFAnalyzer,
    mesh: trimesh.Trimesh,
    num_samples: int = 10000,
    output_dir: str = None
) -> AnalysisResult:
    """
    Run Experiment 4: Normal Gradient Noise Validation

    Args:
        analyzer: MAIUVFAnalyzer instance
        mesh: High-poly mesh
        num_samples: Number of surface points to sample
        output_dir: Output directory

    Returns:
        AnalysisResult with numerical data, figures, and metadata
    """
    logger.info("Starting Experiment 4: Normal Gradient Noise Validation")

    if output_dir is None:
        output_dir = analyzer.create_output_dir("exp4_normal_noise")

    # Step 1: Sample surface points
    logger.info(f"Sampling {num_samples} points on mesh surface")
    positions, face_indices = analyzer.sample_mesh_surface(mesh, num_samples)

    # Step 2: Compute normals at sample points
    logger.info("Computing surface normals")
    normals = np.zeros_like(positions)

    for i, face_idx in enumerate(face_indices):
        # Get face vertices
        face = mesh.faces[face_idx]
        vertices = mesh.vertices[face]

        # Get face normal
        face_normal = mesh.face_normals[face_idx]

        # Interpolate normal (simplified: use face normal)
        normals[i] = face_normal

    # Step 3: Compute Jacobians via autograd
    logger.info("Computing Jacobians via autograd")
    # Use argmax chart IDs for Jacobian computation
    outputs = analyzer.get_network_outputs(positions, return_probs=False)
    chart_ids = outputs['chart_ids']

    jacobians = analyzer.compute_jacobians(positions, chart_ids, batch_size=512)

    # Step 4: Compute normal directional derivatives
    logger.info("Computing normal directional derivatives")
    D_normal = analyzer.get_normal_derivative(jacobians, normals)

    # Step 5: Analyze distribution
    stats = generate_statistics_table(D_normal)

    logger.info("Normal Derivative Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.6e}")

    # Compute curvature (simplified: use face area as proxy)
    curvatures = np.array([mesh.area_faces[face_idx] for face_idx in face_indices])

    # Correlate D_normal with curvature
    from scipy.stats import pearsonr
    corr_coef, p_value = pearsonr(D_normal, curvatures)

    logger.info(f"Correlation with curvature: r={corr_coef:.4f}, p={p_value:.4e}")

    # Save numerical results
    data = {
        'point_id': np.arange(num_samples),
        'D_normal': D_normal,
        'curvature': curvatures,
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2]
    }

    # Generate figures
    figures = []

    # Figure 1: Histogram of D_normal distribution
    fig_path = str(Path(output_dir) / "noise_histogram")
    plot_histogram(D_normal, 'Normal Derivative D_normal',
                   'Distribution of Normal Directional Derivatives', fig_path)
    figures.append(fig_path)

    # Figure 2: Correlation with curvature
    fig_path = str(Path(output_dir) / "curvature_vs_noise")
    plot_correlation_scatter(curvatures, D_normal,
                            'Curvature (face area)', 'D_normal',
                            'Curvature vs Normal Derivative Noise',
                            fig_path, corr_coef, p_value)
    figures.append(fig_path)

    # Figure 3: Mesh colored by D_normal
    fig_path = str(Path(output_dir) / "noise_on_mesh")
    plot_mesh_colored(mesh.vertices, mesh.faces, D_normal,
                     fig_path, 'Normal Gradient Noise on Mesh')
    figures.append(fig_path)

    # Create metadata
    metadata = {
        'experiment_name': 'Normal Gradient Noise Validation',
        'hypothesis': 'In ideal manifold parameterization, D_normal should be ≈0. '
                     'Significant non-zero values indicate wasted capacity on normal gradients.',
        'methodology': f'''Sampled {num_samples} points uniformly on mesh surface.
        For each point:
        1. Computed surface normal
        2. Computed UV Jacobian via autograd [2×3]
        3. Computed normal directional derivative: D_normal = ||J @ n||_2
        4. Analyzed distribution and correlation with curvature''',
        'num_samples': num_samples,
        'statistics': stats,
        'correlation_with_curvature': {
            'r': corr_coef,
            'p': p_value
        },
        'interpretation': _generate_interpretation(stats, corr_coef, p_value)
    }

    # Create results summary
    results_summary = {
        'Mean D_normal': f"{stats['mean']:.6e}",
        'Std D_normal': f"{stats['std']:.6e}",
        'Median D_normal': f"{stats['median']:.6e}",
        '95th percentile': f"{stats['p95']:.6e}",
        '99th percentile': f"{stats['p99']:.6e}",
        'Correlation with curvature': f"r={corr_coef:.4f}, p={p_value:.4e}"
    }

    # Create report
    create_analysis_report(
        experiment_name='Normal Gradient Noise Validation',
        metadata=metadata,
        results_summary=results_summary,
        output_dir=output_dir,
        figures=figures
    )

    logger.info("Experiment 4 complete")

    return AnalysisResult(
        experiment_name='exp4_normal_noise',
        data=data,
        metadata=metadata,
        figures=figures
    )


def _generate_interpretation(
    stats: Dict,
    corr_coef: float,
    p_value: float
) -> str:
    """Generate interpretation of results"""

    interpretation = f"""
### Key Findings

1. **Normal Gradient Magnitude**:
   - Mean D_normal = {stats['mean']:.6e}
   - 95% of points have D_normal < {stats['p95']:.6e}
   - 99% of points have D_normal < {stats['p99']:.6e}

2. **Correlation with Curvature**:
   - Correlation coefficient r = {corr_coef:.4f}
   - P-value = {p_value:.4e}

### Interpretation

"""

    # Interpret mean D_normal
    if stats['mean'] < 1e-3:
        interpretation += "✅ **Low Mean**: Mean D_normal is very small (<1e-3), indicating the network "
        "is NOT learning significant normal gradients. This suggests good manifold-awareness."
    elif stats['mean'] < 1e-2:
        interpretation += "⚠️ **Moderate Mean**: Mean D_normal is noticeable (1e-3 to 1e-2), indicating "
        "some normal gradient learning. This may be acceptable but worth investigating."
    else:
        interpretation += "❌ **High Mean**: Mean D_normal is large (>1e-2), indicating the network IS "
        "learning significant normal gradients. This strongly suggests the full-space L_metric "
        "constraint is wasting capacity on normal direction."

    # Interpret correlation
    interpretation += "\n\n"

    if abs(corr_coef) < 0.1 and p_value > 0.05:
        interpretation += "✅ **No Correlation**: D_normal does not correlate with curvature, "
        "suggesting normal gradients are uniformly low (good)."
    elif abs(corr_coef) > 0.3 and p_value < 0.05:
        interpretation += "⚠️ **Positive Correlation**: D_normal increases with curvature, "
        "suggesting the network struggles more with high-curvature regions. This may indicate "
        "the need for tangent-space projection in L_metric."
    else:
        interpretation += "ℹ️ **Weak Correlation**: Some relationship exists but is not strong. "
        "Further investigation needed."

    interpretation += "\n\n### Recommendations\n\n"

    if stats['mean'] > 1e-2 or (abs(corr_coef) > 0.3 and p_value < 0.05):
        interpretation += "1. **Implement Tangent Space Projection**: Modify L_metric to project "
        "Jacobian onto local tangent plane before computing loss.\n"
        interpretation += "2. **Add Normal Gradient Regularization**: Add explicit penalty term "
        "to discourage normal direction gradients.\n"

    interpretation += "3. **Compare with Fourier Encoding**: Run same analysis with Fourier-encoded "
    "model to see if issue is hash grid specific."

    return interpretation
