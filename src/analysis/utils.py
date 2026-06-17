"""
Visualization and utility functions for MA-IUVF analysis

Provides plotting utilities and common helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Set up matplotlib for high-quality output
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14


def setup_plot_style():
    """Configure matplotlib for publication-ready plots"""
    plt.style.use('seaborn-v0_8-darkgrid')


def save_figure(fig, output_path: str, close: bool = True):
    """
    Save figure to file

    Args:
        fig: matplotlib figure
        output_path: Path to save figure
        close: Whether to close figure after saving
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PNG
    fig.savefig(str(output_path) + '.png', bbox_inches='tight', dpi=300)

    # Save as PDF for vector graphics
    fig.savefig(str(output_path) + '.pdf', bbox_inches='tight')

    if close:
        plt.close(fig)

    logger.info(f"Saved figure: {output_path}")


def plot_entropy_along_line(
    positions: np.ndarray,
    entropy: np.ndarray,
    output_path: str
):
    """
    Plot entropy along a line (for seam crossing experiment)

    Args:
        positions: [N, 3] positions along line
        entropy: [N] entropy values
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute distance along line
    distances = np.linalg.norm(positions - positions[0], axis=1)

    ax.plot(distances, entropy, 'b-', linewidth=2)
    ax.set_xlabel('Distance along line (3D units)')
    ax.set_ylabel('Entropy H')
    ax.set_title('Network Hesitation at Chart Boundary')
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_path)


def plot_uv_trajectory(
    uv_coords: np.ndarray,
    output_path: str,
    chart_ids: Optional[np.ndarray] = None
):
    """
    Plot UV trajectory in 2D

    Args:
        uv_coords: [N, 2] UV coordinates
        output_path: Path to save figure
        chart_ids: [N] optional chart IDs for coloring
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if chart_ids is not None:
        # Color by chart ID
        scatter = ax.scatter(uv_coords[:, 0], uv_coords[:, 1],
                           c=chart_ids, cmap='tab10', s=10)
        plt.colorbar(scatter, ax=ax, label='Chart ID')
    else:
        ax.plot(uv_coords[:, 0], uv_coords[:, 1], 'b.-', linewidth=1, markersize=3)

    ax.set_xlabel('U coordinate')
    ax.set_ylabel('V coordinate')
    ax.set_title('UV Trajectory Across Chart Boundary')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_path)


def plot_chart_distribution(
    probs: np.ndarray,
    output_path: str
):
    """
    Plot chart probability distribution as bar chart

    Args:
        probs: [C] probability distribution
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    num_charts = len(probs)
    ax.bar(range(num_charts), probs, color='steelblue')
    ax.set_xlabel('Chart ID')
    ax.set_ylabel('Probability')
    ax.set_title('Chart Probability Distribution')
    ax.set_xticks(range(num_charts))
    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, output_path)


def plot_correlation_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
    corr_coef: Optional[float] = None,
    p_value: Optional[float] = None
):
    """
    Plot correlation scatter plot

    Args:
        x: First variable
        y: Second variable
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title
        output_path: Path to save figure
        corr_coef: Correlation coefficient (optional)
        p_value: P-value (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x, y, alpha=0.5, s=10)

    # Add correlation info to title
    if corr_coef is not None:
        title += f"\\n(r = {corr_coef:.3f}"
        if p_value is not None:
            title += f", p = {p_value:.2e}"
        title += ")"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_path)


def plot_histogram(
    data: np.ndarray,
    xlabel: str,
    title: str,
    output_path: str,
    bins: int = 50
):
    """
    Plot histogram

    Args:
        data: Data values
        xlabel: X-axis label
        title: Plot title
        output_path: Path to save figure
        bins: Number of bins
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean = np.mean(data)
    std = np.std(data)
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3e}')
    ax.legend()

    save_figure(fig, output_path)


def plot_mesh_colored(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    title: str = "Mesh Colored by Value"
):
    """
    Plot 3D mesh colored by scalar values

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] face indices
        colors: [V] color values
        output_path: Path to save figure
        title: Plot title
    """
    try:
        import trimesh

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Set vertex colors
        # Normalize colors to [0, 1]
        colors_norm = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)

        # Create colormap
        cmap = plt.get_cmap('hot')
        vertex_colors = cmap(colors_norm)

        mesh.visual.vertex_colors = vertex_colors

        # Plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=faces, cmap='hot',
                       shade=True, alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        # Save
        save_figure(fig, output_path)

    except ImportError:
        logger.warning("trimesh not available for 3D plotting")
        # Fallback to scatter plot
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(vertices[:, 0], vertices[:, 1],
                           c=colors, cmap='hot', s=1, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Value')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title + ' (2D projection)')
        ax.grid(True, alpha=0.3)

        save_figure(fig, output_path)


def create_analysis_report(
    experiment_name: str,
    metadata: Dict,
    results_summary: Dict,
    output_dir: str,
    figures: List[str]
):
    """
    Create markdown analysis report

    Args:
        experiment_name: Name of experiment
        metadata: Experiment metadata
        results_summary: Summary of results
        output_dir: Output directory
        figures: List of figure paths
    """
    output_dir = Path(output_dir)
    report_path = output_dir / "report.md"

    with open(report_path, 'w') as f:
        f.write(f"# {experiment_name}\n\n")

        f.write("## Hypothesis\n\n")
        f.write(f"{metadata.get('hypothesis', 'N/A')}\n\n")

        f.write("## Methodology\n\n")
        f.write(f"{metadata.get('methodology', 'N/A')}\n\n")

        f.write("## Results\n\n")
        for key, value in results_summary.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")

        f.write("## Visualizations\n\n")
        for fig_path in figures:
            fig_name = Path(fig_path).stem
            f.write(f"### {fig_name}\n\n")
            f.write(f"![{fig_name}]({fig_path}.png)\n\n")

        f.write("## Interpretation\n\n")
        f.write(f"{metadata.get('interpretation', 'Analysis pending...')}\n\n")

    logger.info(f"Created report: {report_path}")


def generate_statistics_table(
    data: np.ndarray,
    percentiles: List[int] = [50, 90, 95, 99]
) -> Dict:
    """
    Generate statistics table for data

    Args:
        data: Input data
        percentiles: Percentiles to compute

    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data)
    }

    for p in percentiles:
        stats[f'p{p}'] = np.percentile(data, p)

    return stats
