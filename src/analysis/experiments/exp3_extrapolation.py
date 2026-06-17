"""
Experiment 3: Non-Manifold Extrapolation Robustness

Purpose: Test network performance degradation when querying points off the training surface.

Hypothesis: If network fails at non-manifold extrapolation, points far from the surface
should show higher deformation energy (Jacobian singular values), indicating linear
distortion when偏离流形.
"""

import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

from ..maiuvf_analyzer import MAIUVFAnalyzer, AnalysisResult
from ..utils import (
    plot_correlation_scatter, plot_mesh_colored, plot_histogram,
    create_analysis_report, generate_statistics_table
)


def run_experiment3(
    analyzer: MAIUVFAnalyzer,
    high_poly: trimesh.Trimesh,
    low_poly: trimesh.Trimesh = None,
    num_triangles: int = 1000,
    points_per_triangle: int = 100,
    distance_percentile: float = 90.0,
    output_dir: str = None
) -> AnalysisResult:
    """
    Run Experiment 3: Non-Manifold Extrapolation Robustness

    Args:
        analyzer: MAIUVFAnalyzer instance
        high_poly: High-poly mesh (training surface)
        low_poly: Low-poly mesh (optional, will generate if None)
        num_triangles: Number of triangles to sample
        points_per_triangle: Points to sample per triangle
        distance_percentile: Percentile for "far-from-surface" threshold
        output_dir: Output directory

    Returns:
        AnalysisResult with numerical data, figures, and metadata
    """
    logger.info("Starting Experiment 3: Non-Manifold Extrapolation Robustness")

    if output_dir is None:
        output_dir = analyzer.create_output_dir("exp3_extrapolation")

    # Step 1: Generate or load low-poly mesh
    if low_poly is None:
        logger.info("Generating simplified low-poly mesh")
        # Simplify high-poly mesh to target 500 faces
        low_poly = high_poly.simplify_quadric_decimation(face_count=500)
        logger.info(f"Generated low-poly mesh with {len(low_poly.faces)} faces")

    # Step 2: Identify far-from-surface triangles
    logger.info("Identifying far-from-surface triangles")

    # Compute triangle centers for low-poly
    tri_centers = low_poly.triangles_center

    # Find nearest points on high-poly surface
    nearest, distance, _ = high_poly.nearest.on_surface(tri_centers)

    # Set threshold for "far" triangles
    threshold = np.percentile(distance, distance_percentile)
    far_mask = distance > threshold

    far_triangles = np.where(far_mask)[0]
    num_far = len(far_triangles)

    logger.info(f"Found {num_far}/{len(distance)} far triangles (>{threshold:.6f})")

    if num_far == 0:
        logger.warning("No far triangles found, using top 10% by distance")
        num_far = max(1, len(distance) // 10)
        far_triangles = np.argsort(distance)[-num_far:]

    # Limit number of triangles
    if num_far > num_triangles:
        far_triangles = far_triangles[:num_triangles]
        num_far = num_triangles

    # Step 3: Sample points on far triangles
    logger.info(f"Sampling {num_far * points_per_triangle} points on far triangles")

    all_positions = []
    all_sdf_distances = []
    all_tri_indices = []

    for tri_idx in far_triangles:
        # Get triangle vertices
        triangle = low_poly.triangles[tri_idx]

        # Sample barycentric coordinates
        for _ in range(points_per_triangle):
            # Generate random barycentric coordinates
            r1 = np.random.rand()
            r2 = np.random.rand()

            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2

            r3 = 1 - r1 - r2

            # Compute 3D position
            position = r1 * triangle[0] + r2 * triangle[1] + r3 * triangle[2]

            # Compute SDF distance (signed distance to high-poly surface)
            sdf_distance = low_poly.nearest.signed_distance([position])[0]

            all_positions.append(position)
            all_sdf_distances.append(abs(sdf_distance))  # Use absolute distance
            all_tri_indices.append(tri_idx)

    positions = np.array(all_positions)
    sdf_distances = np.array(all_sdf_distances)
    tri_indices = np.array(all_tri_indices)

    logger.info(f"Sampled {len(positions)} points")

    # Step 4: Compute deformation energy
    logger.info("Computing deformation energy (Jacobian via autograd)")

    # Get chart IDs for sampled points
    outputs = analyzer.get_network_outputs(positions, return_probs=False)
    chart_ids = outputs['chart_ids']

    # Compute Jacobians
    jacobians = analyzer.compute_jacobians(positions, chart_ids, batch_size=512)

    # Compute deformation energy (Dirichlet energy)
    deformation_energy = analyzer.get_deformation_energy(jacobians)

    energy_stats = generate_statistics_table(deformation_energy)
    logger.info("Deformation Energy Statistics:")
    for key, value in energy_stats.items():
        logger.info(f"  {key}: {value:.6e}")

    # Step 5: Compute correlation between SDF distance and deformation energy
    from scipy.stats import pearsonr

    corr_coef, p_value = pearsonr(sdf_distances, deformation_energy)

    logger.info(f"SDF-Energy Correlation: r={corr_coef:.4f}, p={p_value:.4e}")

    # Analyze by distance quantiles
    close_threshold = np.percentile(sdf_distances, 33)
    far_threshold = np.percentile(sdf_distances, 67)

    close_mask = sdf_distances < close_threshold
    medium_mask = (sdf_distances >= close_threshold) & (sdf_distances < far_threshold)
    far_mask_distances = sdf_distances >= far_threshold

    energy_close = deformation_energy[close_mask]
    energy_medium = deformation_energy[medium_mask]
    energy_far = deformation_energy[far_mask_distances]

    logger.info(f"Close points (<{close_threshold:.6f}): {len(energy_close)}, mean energy = {np.mean(energy_close):.6e}")
    logger.info(f"Medium points: {len(energy_medium)}, mean energy = {np.mean(energy_medium):.6e}")
    logger.info(f"Far points (>{far_threshold:.6f}): {len(energy_far)}, mean energy = {np.mean(energy_far):.6e}")

    # Save numerical results
    data = {
        'point_id': np.arange(len(positions)),
        'triangle_id': tri_indices,
        'sdf_distance': sdf_distances,
        'deformation_energy': deformation_energy,
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2]
    }

    # Generate figures
    figures = []

    # Figure 1: SDF distance vs Deformation energy scatter plot
    fig_path = str(Path(output_dir) / "sdf_vs_energy")
    plot_correlation_scatter(sdf_distances, deformation_energy,
                            '|SDF Distance|', 'Deformation Energy',
                            'SDF Distance vs Deformation Energy',
                            fig_path, corr_coef, p_value)
    figures.append(fig_path)

    # Figure 2: Far triangles colored by distance
    fig_path = str(Path(output_dir) / "far_triangles")
    # Color low-poly mesh by triangle distances
    tri_colors = distance.copy()
    tri_colors[~far_mask] = np.nan  # Hide non-far triangles
    plot_mesh_colored(low_poly.vertices, low_poly.faces, tri_colors,
                     fig_path, 'Far-From-Surface Triangles')
    figures.append(fig_path)

    # Figure 3: Deformation energy distribution
    fig_path = str(Path(output_dir) / "energy_distribution")
    plot_histogram(deformation_energy, 'Deformation Energy',
                 'Distribution of Deformation Energy', fig_path)
    figures.append(fig_path)

    # Figure 4: SDF distance distribution
    fig_path = str(Path(output_dir) / "sdf_distribution")
    plot_histogram(sdf_distances, '|SDF Distance|',
                 'Distribution of SDF Distances', fig_path)
    figures.append(fig_path)

    # Create metadata
    metadata = {
        'experiment_name': 'Non-Manifold Extrapolation Robustness',
        'hypothesis': 'Network should fail at non-manifold extrapolation, showing '
                     'positive correlation between SDF distance and deformation energy. '
                     'Points far from surface should exhibit linear distortion.',
        'methodology': f'''Generated/loaded low-poly mesh with {len(low_poly.faces)} faces.
        Identified {num_far} far triangles (distance > {threshold:.6f}).
        Sampled {len(positions)} points using barycentric coordinates.
        For each point:
        1. Computed SDF distance to high-poly surface
        2. Computed deformation energy (Dirichlet energy from Jacobian)
        3. Analyzed correlation between distance and energy''',
        'num_triangles': int(num_far),
        'points_per_triangle': int(points_per_triangle),
        'total_points': int(len(positions)),
        'distance_threshold': float(threshold),
        'deformation_energy_statistics': energy_stats,
        'correlation': {
            'r': corr_coef,
            'p': p_value
        },
        'close_region_stats': {
            'threshold': float(close_threshold),
            'num_points': int(len(energy_close)),
            'mean_energy': float(np.mean(energy_close))
        },
        'medium_region_stats': {
            'num_points': int(len(energy_medium)),
            'mean_energy': float(np.mean(energy_medium))
        },
        'far_region_stats': {
            'threshold': float(far_threshold),
            'num_points': int(len(energy_far)),
            'mean_energy': float(np.mean(energy_far))
        },
        'interpretation': _generate_interpretation(
            corr_coef, p_value,
            energy_close, energy_medium, energy_far,
            close_threshold, far_threshold
        )
    }

    # Create results summary
    results_summary = {
        'Correlation coefficient': f"r={corr_coef:.4f}",
        'P-value': f"{p_value:.4e}",
        'Close region mean energy': f"{np.mean(energy_close):.6e}",
        'Medium region mean energy': f"{np.mean(energy_medium):.6e}",
        'Far region mean energy': f"{np.mean(energy_far):.6e}",
        'Energy ratio (far/close)': f"{np.mean(energy_far)/(np.mean(energy_close)+1e-8):.2f}"
    }

    # Create report
    create_analysis_report(
        experiment_name='Non-Manifold Extrapolation Robustness',
        metadata=metadata,
        results_summary=results_summary,
        output_dir=output_dir,
        figures=figures
    )

    logger.info("Experiment 3 complete")

    return AnalysisResult(
        experiment_name='exp3_extrapolation',
        data=data,
        metadata=metadata,
        figures=figures
    )


def _generate_interpretation(
    corr_coef: float,
    p_value: float,
    energy_close: np.ndarray,
    energy_medium: np.ndarray,
    energy_far: np.ndarray,
    close_threshold: float,
    far_threshold: float
) -> str:
    """Generate interpretation of results"""

    interpretation = f"""
### Key Findings

1. **Correlation Analysis**:
   - Pearson correlation coefficient: r = {corr_coef:.4f}
   - P-value: {p_value:.4e}
   - Expected: Strong positive correlation if non-manifold extrapolation fails

2. **Regional Comparison**:
   - Close region (SDF < {close_threshold:.6f}): {len(energy_close)} points, mean energy = {np.mean(energy_close):.6e}
   - Medium region: {len(energy_medium)} points, mean energy = {np.mean(energy_medium):.6e}
   - Far region (SDF > {far_threshold:.6f}): {len(energy_far)} points, mean energy = {np.mean(energy_far):.6e}
   - Energy ratio (far/close): {np.mean(energy_far)/(np.mean(energy_close)+1e-8):.2f}

### Interpretation

"""

    # Interpret correlation
    if corr_coef > 0.3 and p_value < 0.05:
        interpretation += "❌ **Strong Positive Correlation**: Significant positive correlation exists, "
        "confirming that points farther from surface show higher deformation energy. This strongly "
        "supports the hypothesis that the network fails at non-manifold extrapolation."
    elif corr_coef > 0.1 and p_value < 0.05:
        interpretation += "⚠️ **Moderate Positive Correlation**: Positive correlation exists but is not "
        "strong. This suggests some extrapolation issues but may be tolerable."
    elif p_value >= 0.05:
        interpretation += "✅ **No Significant Correlation**: No statistically significant correlation "
        "found. This suggests the network handles non-manifold extrapolation reasonably well."
    else:
        interpretation += "ℹ️ **Weak/No Correlation**: Correlation is weak or negative, suggesting "
        "no clear extrapolation problem."

    # Interpret regional comparison
    interpretation += "\n\n"

    energy_ratio = np.mean(energy_far) / (np.mean(energy_close) + 1e-8)

    if energy_ratio > 2.0:
        interpretation += "❌ **Severe Extrapolation Failure**: Far points show {energy_ratio:.1f}x higher "
        "deformation energy than close points. This is a strong indicator of non-manifold extrapolation "
        "failure, suggesting the network only learns surface behavior."
    elif energy_ratio > 1.5:
        interpretation += "⚠️ **Moderate Extrapolation Degradation**: Far points show elevated energy "
        "({energy_ratio:.1f}x higher). This suggests extrapolation issues but may be acceptable."
    else:
        interpretation += "✅ **Good Extrapolation**: Energy is similar across distance ranges, "
        "suggesting the network generalizes well off the surface."

    interpretation += "\n\n### Recommendations\n\n"

    if corr_coef > 0.3 and p_value < 0.05 and energy_ratio > 1.5:
        interpretation += "1. **Expand Training Sampling**: Include off-surface samples in training data "
        "to improve extrapolation robustness.\n"
        interpretation += "2. **Add SDF-Aware Regularization**: Penalize large gradients in normal "
        "direction to enforce manifold constraint.\n"
        interpretation += "3. **Use Off-Surface Validation**: Monitor performance on off-surface "
        "points during training to detect extrapolation failure early."

    return interpretation
