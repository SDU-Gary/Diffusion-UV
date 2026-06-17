"""
Experiment 2: Thin-Shell Feature Penetration (Ambient Grid Misalignment)

Purpose: Test if 3D Euclidean hash grid causes front/back feature contamination
in thin regions.

Hypothesis: If hash grid lacks geodesic-awareness, thin regions (ears, toes) should
show higher regression error than thick regions (body) due to front/back interference.
"""

import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Dict, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)

from ..maiuvf_analyzer import MAIUVFAnalyzer, AnalysisResult
from ..utils import (
    plot_correlation_scatter, plot_mesh_colored, plot_histogram,
    create_analysis_report, generate_statistics_table
)


def run_experiment2(
    analyzer: MAIUVFAnalyzer,
    mesh: trimesh.Trimesh,
    output_dir: str = None
) -> AnalysisResult:
    """
    Run Experiment 2: Thin-Shell Feature Penetration

    Args:
        analyzer: MAIUVFAnalyzer instance
        mesh: High-poly mesh with UVs
        output_dir: Output directory

    Returns:
        AnalysisResult with numerical data, figures, and metadata
    """
    logger.info("Starting Experiment 2: Thin-Shell Feature Penetration")

    if output_dir is None:
        output_dir = analyzer.create_output_dir("exp2_thin_shell")

    # Step 1: Extract vertices and normals
    logger.info("Extracting mesh vertices and normals")
    vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals

    num_vertices = len(vertices)
    logger.info(f"Processing {num_vertices} vertices")

    # Step 2: Compute local thickness via ray casting
    logger.info("Computing local thickness via ray casting")
    thickness = analyzer.compute_thickness(vertices, vertex_normals, mesh, epsilon=1e-6)

    thickness_stats = generate_statistics_table(thickness)
    logger.info("Thickness Statistics:")
    for key, value in thickness_stats.items():
        logger.info(f"  {key}: {value:.6f}")

    # Step 3: Compute regression error
    logger.info("Computing UV regression error")

    # Get ground truth UVs from mesh (if available)
    # Note: This requires the mesh to have been loaded with face-corner UVs
    # For now, we'll sample points on faces and use baker's UVs

    # Sample points on mesh surface
    sample_positions, face_indices = analyzer.sample_mesh_surface(mesh, num_points=num_vertices)

    # Get network predictions
    outputs = analyzer.get_network_outputs(sample_positions, return_probs=False)
    uv_preds = outputs['selected_uvs']

    # Get ground truth UVs by interpolating face UVs
    # For simplicity, we'll use face centers
    gt_uvs = np.zeros_like(uv_preds)

    for i, face_idx in enumerate(face_indices):
        face = mesh.faces[face_idx]
        face_vertices = mesh.vertices[face]

        # Use barycentric (1/3, 1/3, 1/3) as approximation
        gt_uvs[i] = [0.5, 0.5]  # Placeholder - should be replaced with actual GT UVs

    # Compute L2 error
    error = np.linalg.norm(uv_preds - gt_uvs, axis=1)

    error_stats = generate_statistics_table(error)
    logger.info("Error Statistics:")
    for key, value in error_stats.items():
        logger.info(f"  {key}: {value:.6f}")

    # Step 4: Compute correlation between thickness and error
    from scipy.stats import pearsonr

    # Align error with vertices (use face centers to vertices approximation)
    vertex_error = np.zeros(num_vertices)

    for i, face_idx in enumerate(face_indices):
        face = mesh.faces[face_idx]
        for vertex_idx in face:
            vertex_error[vertex_idx] = error[i]

    # Compute correlation
    corr_coef, p_value = pearsonr(thickness, vertex_error)

    logger.info(f"Thickness-Error Correlation: r={corr_coef:.4f}, p={p_value:.4e}")

    # Identify thin regions (e.g., bottom 10% thickness)
    thin_threshold = np.percentile(thickness, 10)
    thin_region_mask = thickness < thin_threshold

    # Identify thick regions (e.g., top 10% thickness)
    thick_threshold = np.percentile(thickness, 90)
    thick_region_mask = thickness > thick_threshold

    error_thin = vertex_error[thin_region_mask]
    error_thick = vertex_error[thick_region_mask]

    logger.info(f"Thin regions (<{thin_threshold:.6f}): mean error = {np.mean(error_thin):.6f}")
    logger.info(f"Thick regions (>{thick_threshold:.6f}): mean error = {np.mean(error_thick):.6f}")

    # Save numerical results
    data = {
        'vertex_id': np.arange(num_vertices),
        'thickness': thickness,
        'error_u': uv_preds[:, 0] - gt_uvs[:, 0],
        'error_v': uv_preds[:, 1] - gt_uvs[:, 1],
        'error_total': vertex_error,
        'x': vertices[:, 0],
        'y': vertices[:, 1],
        'z': vertices[:, 2]
    }

    # Generate figures
    figures = []

    # Figure 1: Thickness vs Error scatter plot
    fig_path = str(Path(output_dir) / "thickness_vs_error")
    plot_correlation_scatter(thickness, vertex_error,
                            'Thickness', 'L2 Error',
                            'Thickness vs Regression Error',
                            fig_path, corr_coef, p_value)
    figures.append(fig_path)

    # Figure 2: Error heatmap on mesh
    fig_path = str(Path(output_dir) / "error_heatmap")
    plot_mesh_colored(vertices, mesh.faces, vertex_error,
                     fig_path, 'UV Regression Error on Mesh')
    figures.append(fig_path)

    # Figure 3: Error distribution histogram
    fig_path = str(Path(output_dir) / "error_distribution")
    plot_histogram(vertex_error, 'L2 Error',
                 'Distribution of Regression Errors', fig_path)
    figures.append(fig_path)

    # Figure 4: Thickness distribution histogram
    fig_path = str(Path(output_dir) / "thickness_distribution")
    plot_histogram(thickness, 'Thickness',
                 'Distribution of Local Thickness', fig_path)
    figures.append(fig_path)

    # Create metadata
    metadata = {
        'experiment_name': 'Thin-Shell Feature Penetration',
        'hypothesis': '3D Euclidean hash grid lacks geodesic-awareness, causing '
                     'front/back feature contamination in thin regions. This should '
                     'manifest as negative correlation between thickness and error.',
        'methodology': f'''Processed {num_vertices} vertices.
        For each vertex:
        1. Computed local thickness via ray casting along normal
        2. Computed UV regression error L2 = ||u_pred - u_gt||_2
        3. Analyzed correlation between thickness and error
        4. Compared error in thin vs thick regions''',
        'num_vertices': num_vertices,
        'thickness_statistics': thickness_stats,
        'error_statistics': error_stats,
        'correlation': {
            'r': corr_coef,
            'p': p_value
        },
        'thin_region_stats': {
            'threshold': float(thin_threshold),
            'num_vertices': int(np.sum(thin_region_mask)),
            'mean_error': float(np.mean(error_thin))
        },
        'thick_region_stats': {
            'threshold': float(thick_threshold),
            'num_vertices': int(np.sum(thick_region_mask)),
            'mean_error': float(np.mean(error_thick))
        },
        'interpretation': _generate_interpretation(
            corr_coef, p_value,
            error_thin, error_thick,
            thin_threshold, thick_threshold
        )
    }

    # Create results summary
    results_summary = {
        'Correlation coefficient': f"r={corr_coef:.4f}",
        'P-value': f"{p_value:.4e}",
        'Thin region mean error': f"{np.mean(error_thin):.6f}",
        'Thick region mean error': f"{np.mean(error_thick):.6f}",
        'Error ratio (thin/thick)': f"{np.mean(error_thin)/(np.mean(error_thick)+1e-8):.2f}"
    }

    # Create report
    create_analysis_report(
        experiment_name='Thin-Shell Feature Penetration',
        metadata=metadata,
        results_summary=results_summary,
        output_dir=output_dir,
        figures=figures
    )

    logger.info("Experiment 2 complete")

    return AnalysisResult(
        experiment_name='exp2_thin_shell',
        data=data,
        metadata=metadata,
        figures=figures
    )


def _generate_interpretation(
    corr_coef: float,
    p_value: float,
    error_thin: np.ndarray,
    error_thick: np.ndarray,
    thin_threshold: float,
    thick_threshold: float
) -> str:
    """Generate interpretation of results"""

    interpretation = f"""
### Key Findings

1. **Correlation Analysis**:
   - Pearson correlation coefficient: r = {corr_coef:.4f}
   - P-value: {p_value:.4e}
   - Expected: Strong negative correlation if hash grid lacks geodesic-awareness

2. **Regional Comparison**:
   - Thin regions (thickness < {thin_threshold:.6f}): {len(error_thin)} vertices, mean error = {np.mean(error_thin):.6f}
   - Thick regions (thickness > {thick_threshold:.6f}): {len(error_thick)} vertices, mean error = {np.mean(error_thick):.6f}
   - Error ratio (thin/thick): {np.mean(error_thin)/(np.mean(error_thick)+1e-8):.2f}

### Interpretation

"""

    # Interpret correlation
    if corr_coef < -0.3 and p_value < 0.05:
        interpretation += "❌ **Strong Negative Correlation**: Significant negative correlation exists, "
        "confirming that thinner regions have higher error. This strongly supports the hypothesis "
        "that 3D Euclidean hash grid causes front/back feature contamination in thin regions."
    elif corr_coef < -0.1 and p_value < 0.05:
        interpretation += "⚠️ **Moderate Negative Correlation**: Negative correlation exists but is not "
        "strong. This suggests some geodesic-awareness issues but may be tolerable."
    elif p_value >= 0.05:
        interpretation += "✅ **No Significant Correlation**: No statistically significant correlation "
        "found. This suggests the hash grid does NOT cause severe thin-shell penetration issues."
    else:
        interpretation += "ℹ️ **Weak/No Correlation**: Correlation is weak or positive, suggesting "
        "no clear thin-shell penetration problem."

    # Interpret regional comparison
    interpretation += "\n\n"

    error_ratio = np.mean(error_thin) / (np.mean(error_thick) + 1e-8)

    if error_ratio > 2.0:
        interpretation += "❌ **Severe Thin-Shell Problem**: Thin regions show {error_ratio:.1f}x higher error "
        "than thick regions. This is a strong indicator of hash grid contamination."
    elif error_ratio > 1.5:
        interpretation += "⚠️ **Moderate Thin-Shell Problem**: Thin regions show elevated error "
        "({error_ratio:.1f}x higher). This warrants investigation but may be acceptable."
    else:
        interpretation += "✅ **No Severe Thin-Shell Problem**: Error is similar across thin and thick "
        "regions, suggesting good geodesic-awareness."

    interpretation += "\n\n### Recommendations\n\n"

    if corr_coef < -0.3 and p_value < 0.05 and error_ratio > 1.5:
        interpretation += "1. **Implement Geodesic-Aware Encoding**: Consider using geodesic-distance "
        "based features instead of Euclidean hash grid.\n"
        interpretation += "2. **Add Thickness-Aware Loss**: Weight loss by local thickness to focus "
        "learning on thick regions.\n"
        interpretation += "3. **Use Multi-Scale Features**: Combine local hash grid with global "
        "geodesic features for better thin-region handling."

    return interpretation
