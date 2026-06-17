#!/usr/bin/env python
"""
MA-IUVF Analysis Script

Run experiments to analyze MA-IUVF model behavior:
- Experiment 1: Seam Jump & Network Hesitation (Continuity Test)
- Experiment 2: Thin-Shell Feature Penetration (Ambient Grid Misalignment)
- Experiment 3: Non-Manifold Extrapolation Robustness
- Experiment 4: Normal Gradient Noise Validation

Usage:
    # Run single experiment
    python scripts/run_maiuvf_analysis.py exp1 --checkpoint <path> --output-dir <dir>

    # Run all experiments
    python scripts/run_maiuvf_analysis.py all --checkpoint <path> --output-dir <dir>
"""

import argparse
import sys
import logging
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.maiuvf_analyzer import MAIUVFAnalyzer
from src.analysis.experiments import (
    run_experiment1, run_experiment2,
    run_experiment3, run_experiment4
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MA-IUVF Analysis Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run seam continuity experiment
  python scripts/run_maiuvf_analysis.py exp1 \\
      --checkpoint outputs/maiuvf_phase1/best.pt \\
      --mesh data/models/stanford_bunny_procedural.obj \\
      --output-dir outputs/maiuvf_analysis/

  # Run normal gradient noise experiment
  python scripts/run_maiuvf_analysis.py exp4 \\
      --checkpoint outputs/maiuvf_phase1/best.pt \\
      --mesh data/models/stanford_bunny_procedural.obj \\
      --output-dir outputs/maiuvf_analysis/

  # Run all experiments
  python scripts/run_maiuvf_analysis.py all \\
      --checkpoint outputs/maiuvf_phase1/best.pt \\
      --mesh data/models/stanford_bunny_procedural.obj \\
      --output-dir outputs/maiuvf_analysis/
        """
    )

    parser.add_argument(
        'experiment',
        choices=['exp1', 'exp2', 'exp3', 'exp4', 'all'],
        help='Experiment to run (or "all" for all experiments)'
    )

    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--mesh',
        default='data/models/stanford_bunny_procedural.obj',
        help='Path to mesh file (default: stanford_bunny_procedural.obj)'
    )

    parser.add_argument(
        '--output-dir',
        default='outputs/maiuvf_analysis',
        help='Output directory (default: outputs/maiuvf_analysis)'
    )

    parser.add_argument(
        '--device',
        default='cuda',
        help='Device for inference (default: cuda)'
    )

    # Experiment-specific parameters
    parser.add_argument(
        '--exp1-num-points',
        type=int,
        default=1000,
        help='Number of points for seam continuity test (default: 1000)'
    )

    parser.add_argument(
        '--exp1-line-length',
        type=float,
        default=0.01,
        help='Line length for seam test (default: 0.01)'
    )

    parser.add_argument(
        '--exp4-num-samples',
        type=int,
        default=10000,
        help='Number of samples for normal noise test (default: 10000)'
    )

    parser.add_argument(
        '--exp3-num-triangles',
        type=int,
        default=1000,
        help='Number of triangles for extrapolation test (default: 1000)'
    )

    parser.add_argument(
        '--exp3-points-per-triangle',
        type=int,
        default=100,
        help='Points per triangle for extrapolation test (default: 100)'
    )

    return parser.parse_args()


def load_mesh(mesh_path: str):
    """Load mesh file"""
    import trimesh

    logger.info(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]

    logger.info(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    return mesh


def run_experiment_1(args, analyzer, mesh):
    """Run Experiment 1: Seam Continuity"""
    logger.info("="*60)
    logger.info("Running Experiment 1: Seam Jump & Network Hesitation")
    logger.info("="*60)

    start_time = time.time()

    result = run_experiment1(
        analyzer=analyzer,
        mesh_path=args.mesh,
        num_points=args.exp1_num_points,
        line_length=args.exp1_line_length,
        output_dir=Path(args.output_dir) / "exp1_seam_continuity"
    )

    elapsed = time.time() - start_time
    logger.info(f"Experiment 1 completed in {elapsed:.2f} seconds")

    return result


def run_experiment_2(args, analyzer, mesh):
    """Run Experiment 2: Thin-Shell Penetration"""
    logger.info("="*60)
    logger.info("Running Experiment 2: Thin-Shell Feature Penetration")
    logger.info("="*60)

    start_time = time.time()

    result = run_experiment2(
        analyzer=analyzer,
        mesh=mesh,
        output_dir=Path(args.output_dir) / "exp2_thin_shell"
    )

    elapsed = time.time() - start_time
    logger.info(f"Experiment 2 completed in {elapsed:.2f} seconds")

    return result


def run_experiment_3(args, analyzer, mesh):
    """Run Experiment 3: Non-Manifold Extrapolation"""
    logger.info("="*60)
    logger.info("Running Experiment 3: Non-Manifold Extrapolation Robustness")
    logger.info("="*60)

    start_time = time.time()

    result = run_experiment3(
        analyzer=analyzer,
        high_poly=mesh,
        low_poly=None,  # Will generate automatically
        num_triangles=args.exp3_num_triangles,
        points_per_triangle=args.exp3_points_per_triangle,
        output_dir=Path(args.output_dir) / "exp3_extrapolation"
    )

    elapsed = time.time() - start_time
    logger.info(f"Experiment 3 completed in {elapsed:.2f} seconds")

    return result


def run_experiment_4(args, analyzer, mesh):
    """Run Experiment 4: Normal Gradient Noise"""
    logger.info("="*60)
    logger.info("Running Experiment 4: Normal Gradient Noise Validation")
    logger.info("="*60)

    start_time = time.time()

    result = run_experiment4(
        analyzer=analyzer,
        mesh=mesh,
        num_samples=args.exp4_num_samples,
        output_dir=Path(args.output_dir) / "exp4_normal_noise"
    )

    elapsed = time.time() - start_time
    logger.info(f"Experiment 4 completed in {elapsed:.2f} seconds")

    return result


def create_comprehensive_report(args, results):
    """Create comprehensive analysis report across all experiments"""
    output_dir = Path(args.output_dir)
    report_path = output_dir / "comprehensive_report.md"

    logger.info(f"Creating comprehensive report: {report_path}")

    with open(report_path, 'w') as f:
        f.write("# MA-IUVF Comprehensive Analysis Report\n\n")

        f.write("## Overview\n\n")
        f.write(f"**Checkpoint**: {args.checkpoint}\n\n")
        f.write(f"**Mesh**: {args.mesh}\n\n")
        f.write(f"**Output Directory**: {args.output_dir}\n\n")

        f.write("## Experiments Summary\n\n")

        for exp_name, result in results.items():
            f.write(f"### {result.experiment_name}\n\n")

            # Add key findings
            if 'statistics' in result.metadata:
                stats = result.metadata['statistics']
                f.write("**Key Statistics:**\n\n")
                for key, value in stats.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

            if 'correlation' in result.metadata:
                corr = result.metadata['correlation']
                f.write(f"**Correlation**: r={corr['r']:.4f}, p={corr['p']:.4e}\n\n")

            # Link to detailed report
            exp_dir = output_dir / result.experiment_name
            f.write(f"📄 [Detailed Report]({exp_dir}/report.md)\n\n")

        f.write("## Conclusions\n\n")
        f.write("See individual experiment reports for detailed interpretations and recommendations.\n\n")

    logger.info(f"Comprehensive report created: {report_path}")


def main():
    """Main entry point"""
    args = parse_args()

    logger.info("MA-IUVF Analysis Experiments")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Mesh: {args.mesh}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    # Initialize analyzer
    try:
        analyzer = MAIUVFAnalyzer(args.checkpoint, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    # Load mesh
    try:
        mesh = load_mesh(args.mesh)
    except Exception as e:
        logger.error(f"Failed to load mesh: {e}")
        return 1

    # Run experiments
    results = {}
    overall_start = time.time()

    try:
        if args.experiment == 'all':
            logger.info("Running all experiments...")

            # Run in recommended priority order
            results['exp4'] = run_experiment_4(args, analyzer, mesh)
            results['exp1'] = run_experiment_1(args, analyzer, mesh)
            results['exp2'] = run_experiment_2(args, analyzer, mesh)
            results['exp3'] = run_experiment_3(args, analyzer, mesh)

            # Create comprehensive report
            create_comprehensive_report(args, results)

        elif args.experiment == 'exp1':
            run_experiment_1(args, analyzer, mesh)

        elif args.experiment == 'exp2':
            run_experiment_2(args, analyzer, mesh)

        elif args.experiment == 'exp3':
            run_experiment_3(args, analyzer, mesh)

        elif args.experiment == 'exp4':
            run_experiment_4(args, analyzer, mesh)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    overall_elapsed = time.time() - overall_start
    logger.info(f"All experiments completed in {overall_elapsed:.2f} seconds ({overall_elapsed/60:.1f} minutes)")

    logger.info(f"Results saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
