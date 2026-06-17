"""
Experiment 1: Seam Jump & Network Hesitation (Continuity Test)

Purpose: Test if C² continuous B-Spline features cause "transition zone stretching"
or classification collapse across chart boundaries.

Hypothesis: If continuous features cannot handle discrete chart jumps, we should observe:
1. Entropy spike near chart boundaries (network hesitation)
2. UV trajectory shows fuzzy curve crossing charts instead of clean jump
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

from ..maiuvf_analyzer import MAIUVFAnalyzer, AnalysisResult
from ...data.metric_aligned_iuv_baker import MetricAlignedIUVBaker
from ..utils import (
    plot_entropy_along_line, plot_uv_trajectory, plot_chart_distribution,
    create_analysis_report, generate_statistics_table
)


def run_experiment1(
    analyzer: MAIUVFAnalyzer,
    mesh_path: str,
    num_points: int = 1000,
    line_length: float = 0.01,
    output_dir: str = None
) -> AnalysisResult:
    """
    Run Experiment 1: Seam Jump & Network Hesitation

    Args:
        analyzer: MAIUVFAnalyzer instance
        mesh_path: Path to mesh with UV charts
        num_points: Number of points to sample along transverse line
        line_length: Length of transverse line (relative to bbox diagonal)
        output_dir: Output directory

    Returns:
        AnalysisResult with numerical data, figures, and metadata
    """
    logger.info("Starting Experiment 1: Seam Jump & Network Hesitation")

    if output_dir is None:
        output_dir = analyzer.create_output_dir("exp1_seam_continuity")

    # Step 1: Load mesh and extract chart boundaries
    logger.info(f"Loading mesh: {mesh_path}")
    baker = MetricAlignedIUVBaker(mesh_path, use_obj_parser=True, seed=42)

    # Get chart information
    face_chart_ids = baker._face_chart_id if hasattr(baker, '_face_chart_id') else None
    chart_info = baker._chart_info if hasattr(baker, '_chart_info') else {}

    # Get UV seams
    uv_seams = chart_info.get('uv_seams', [])
    num_charts = chart_info.get('num_charts', analyzer.num_charts)

    logger.info(f"Found {num_charts} charts, {len(uv_seams)} UV seams")

    if len(uv_seams) == 0:
        logger.warning("No UV seams found, using default chart boundary")
        # Create synthetic boundary between chart 0 and 1
        uv_seams = [(0, 1, None)]

    # Step 2: Select a seam and construct transverse line
    # Use first seam (e.g., red/purple boundary = charts 0 and 1)
    face1_idx, face2_idx, edge = uv_seams[0]

    # Get face vertices
    face1 = baker.face_vertices[face1_idx]
    face2 = baker.face_vertices[face2_idx]

    # Find shared edge
    shared_edge = None
    for i in range(3):
        for j in range(3):
            if torch.equal(face1[i], face2[j]) or torch.allclose(face1[i], face2[j], atol=1e-6):
                if shared_edge is None:
                    shared_edge = [face1[i]]
                else:
                    shared_edge.append(face1[i])

    if shared_edge is None or len(shared_edge) < 2:
        # Fallback: use first edge of face1
        shared_edge = [face1[0], face1[1]]

    # Compute edge midpoint
    edge_midpoint = (shared_edge[0] + shared_edge[1]) / 2

    # Ensure numpy format
    if torch.is_tensor(edge_midpoint):
        edge_midpoint = edge_midpoint.numpy()
    if torch.is_tensor(shared_edge[0]):
        shared_edge = [e.numpy() for e in shared_edge]
    else:
        shared_edge = [shared_edge[0], shared_edge[1]]

    # Compute edge direction
    edge_direction = shared_edge[1] - shared_edge[0]
    edge_direction = edge_direction / (np.linalg.norm(edge_direction) + 1e-8)

    # Compute face normal
    face_normal = baker.face_normals[face1_idx]
    if torch.is_tensor(face_normal):
        face_normal = face_normal.numpy()

    # Compute perpendicular direction (cross product)
    perp_direction = np.cross(edge_direction, face_normal)
    perp_direction = perp_direction / (np.linalg.norm(perp_direction) + 1e-8)

    # Compute bbox diagonal for scaling
    vertices = baker.face_vertices.reshape(-1, 3)
    if torch.is_tensor(vertices):
        vertices = vertices.numpy()
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

    # Scale line length
    line_length_scaled = line_length * bbox_diagonal

    logger.info(f"Line length: {line_length_scaled:.6f} (1% of bbox diagonal)")

    # Step 3: Sample points along transverse line
    t_values = np.linspace(-0.5, 0.5, num_points)
    positions = edge_midpoint + perp_direction[np.newaxis, :] * t_values[:, np.newaxis] * line_length_scaled

    logger.info(f"Sampled {num_points} points along transverse line")

    # Step 4: Query network
    logger.info("Querying network for predictions")
    outputs = analyzer.get_network_outputs(positions, return_probs=True)

    logits = outputs['logits']
    uv_preds = outputs['uv_preds']
    probs = outputs['probs']
    chart_ids = outputs['chart_ids']
    selected_uvs = outputs['selected_uvs']

    # Step 5: Compute entropy
    entropy = analyzer.compute_entropy(probs)

    # Analyze entropy spike near seam
    # Find indices near middle of line (closest to seam)
    mid_idx = num_points // 2
    window_size = num_points // 10
    near_seam_indices = np.arange(mid_idx - window_size, mid_idx + window_size)

    entropy_near_seam = entropy[near_seam_indices]
    entropy_away_from_seam = np.concatenate([entropy[:mid_idx - window_size],
                                            entropy[mid_idx + window_size:]])

    entropy_stats = {
        'mean_near_seam': np.mean(entropy_near_seam),
        'mean_away_from_seam': np.mean(entropy_away_from_seam),
        'max_near_seam': np.max(entropy_near_seam),
        'entropy_ratio': np.mean(entropy_near_seam) / (np.mean(entropy_away_from_seam) + 1e-8)
    }

    logger.info("Entropy Statistics:")
    for key, value in entropy_stats.items():
        logger.info(f"  {key}: {value:.6f}")

    # Analyze UV trajectory
    # Compute UV path length
    uv_path_length = 0.0
    for i in range(1, num_points):
        uv_path_length += np.linalg.norm(selected_uvs[i] - selected_uvs[i-1])

    # Check if UV path crosses other charts
    chart_changes = np.sum(chart_ids[1:] != chart_ids[:-1])

    logger.info(f"UV path length: {uv_path_length:.6f}")
    logger.info(f"Chart ID changes: {chart_changes}")

    # Save numerical results
    data = {
        'position_x': positions[:, 0],
        'position_y': positions[:, 1],
        'position_z': positions[:, 2],
        'distance_along_line': t_values * line_length_scaled,
        'entropy': entropy,
        'chart_id': chart_ids,
        'u': selected_uvs[:, 0],
        'v': selected_uvs[:, 1]
    }

    # Add probabilities for each chart
    for c in range(analyzer.num_charts):
        data[f'prob_chart_{c}'] = probs[:, c]

    # Generate figures
    figures = []

    # Figure 1: Entropy along line
    fig_path = str(Path(output_dir) / "entropy_line")
    plot_entropy_along_line(positions, entropy, fig_path)
    figures.append(fig_path)

    # Figure 2: UV trajectory in 2D
    fig_path = str(Path(output_dir) / "uv_trajectory")
    plot_uv_trajectory(selected_uvs, fig_path, chart_ids)
    figures.append(fig_path)

    # Figure 3: Chart probability distribution at seam midpoint
    fig_path = str(Path(output_dir) / "chart_distribution")
    plot_chart_distribution(probs[mid_idx], fig_path)
    figures.append(fig_path)

    # Create metadata
    metadata = {
        'experiment_name': 'Seam Jump & Network Hesitation',
        'hypothesis': 'C² continuous B-Spline features should cause network hesitation '
                     '(entropy spike) at chart boundaries if they cannot handle discrete jumps.',
        'methodology': f'''Selected chart boundary (faces {face1_idx} and {face2_idx}).
        Constructed transverse line of length {line_length_scaled:.6f} ({line_length*100:.1f}% of bbox diagonal).
        Sampled {num_points} points along line.
        For each point:
        1. Computed classification probabilities and entropy
        2. Extracted UV predictions
        3. Analyzed entropy spike and UV trajectory''',
        'num_points': num_points,
        'line_length': line_length_scaled,
        'face1_idx': int(face1_idx),
        'face2_idx': int(face2_idx),
        'entropy_statistics': entropy_stats,
        'uv_path_length': float(uv_path_length),
        'chart_changes': int(chart_changes),
        'interpretation': _generate_interpretation(entropy_stats, uv_path_length, chart_changes, num_charts)
    }

    # Create results summary
    results_summary = {
        'Entropy (near seam)': f"{entropy_stats['mean_near_seam']:.6f}",
        'Entropy (away from seam)': f"{entropy_stats['mean_away_from_seam']:.6f}",
        'Entropy ratio': f"{entropy_stats['entropy_ratio']:.2f}",
        'Max entropy (near seam)': f"{entropy_stats['max_near_seam']:.6f}",
        'UV path length': f"{uv_path_length:.6f}",
        'Chart ID changes': f"{chart_changes}"
    }

    # Create report
    create_analysis_report(
        experiment_name='Seam Jump & Network Hesitation',
        metadata=metadata,
        results_summary=results_summary,
        output_dir=output_dir,
        figures=figures
    )

    logger.info("Experiment 1 complete")

    return AnalysisResult(
        experiment_name='exp1_seam_continuity',
        data=data,
        metadata=metadata,
        figures=figures
    )


def _generate_interpretation(
    entropy_stats: Dict,
    uv_path_length: float,
    chart_changes: int,
    num_charts: int
) -> str:
    """Generate interpretation of results"""

    interpretation = f"""
### Key Findings

1. **Entropy Analysis**:
   - Mean entropy near seam: {entropy_stats['mean_near_seam']:.6f}
   - Mean entropy away from seam: {entropy_stats['mean_away_from_seam']:.6f}
   - Entropy ratio (near/away): {entropy_stats['entropy_ratio']:.2f}
   - Max entropy near seam: {entropy_stats['max_near_seam']:.6f}

2. **UV Trajectory**:
   - Total path length: {uv_path_length:.6f}
   - Chart ID changes along path: {chart_changes}

### Interpretation

"""

    # Interpret entropy ratio
    if entropy_stats['entropy_ratio'] < 1.2:
        interpretation += "✅ **No Entropy Spike**: Entropy near seam is similar to away from seam, "
        "indicating the network does NOT show hesitation at the boundary. This suggests "
        "continuous features handle the discrete jump well."
    elif entropy_stats['entropy_ratio'] < 1.5:
        interpretation += "⚠️ **Moderate Entropy Increase**: Entropy increases slightly near seam, "
        "indicating some network hesitation but not severe. This may be acceptable behavior."
    else:
        interpretation += "❌ **Strong Entropy Spike**: Entropy increases dramatically near seam, "
        "indicating severe network hesitation. This suggests continuous features cannot "
        "handle the discrete chart boundary cleanly."

    # Interpret UV trajectory
    interpretation += "\n\n"

    if chart_changes < 3:
        interpretation += "✅ **Clean Chart Transition**: UV trajectory stays within few charts, "
        "showing relatively clean transition across boundary."
    elif chart_changes < 10:
        interpretation += "⚠️ **Moderate Chart Hopping**: UV trajectory visits multiple charts, "
        "indicating some uncertainty in chart classification near boundary."
    else:
        interpretation += "❌ **Chaotic Chart Hopping**: UV trajectory visits many charts, "
        "indicating severe confusion in chart classification. This suggests the network "
        "cannot make a clean decision at the boundary."

    interpretation += "\n\n### Recommendations\n\n"

    if entropy_stats['entropy_ratio'] > 1.5 or chart_changes > 10:
        interpretation += "1. **Add Boundary-Aware Loss**: Modify loss function to explicitly "
        "penalize uncertainty at chart boundaries.\n"
        interpretation += "2. **Use Discrete Features**: Consider replacing continuous B-Spline "
        "features with discrete encoding near boundaries.\n"
        interpretation += "3. **Chart Boundary Refinement**: Increase chart boundary resolution "
        "in training data."

    return interpretation
