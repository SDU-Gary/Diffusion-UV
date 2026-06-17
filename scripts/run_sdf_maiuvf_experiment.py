"""
End-to-End SDF + MA-IUVF Experiment Pipeline

This script runs the complete experiment pipeline:

1. Pre-train SDF network (5-10 epochs)
2. Validate SDF normals (3 metrics)
3. Train MA-IUVF with SDF normals (10 epochs)
4. Measure D_normal (Experiment 4)

Expected results:
- SDF validation: cosine similarity > 0.99, Eikonal error < 1e-3
- Classification accuracy > 97.52% (baseline)
- D_normal reduced from 1.427 (mesh normals) → < 0.5 (SDF normals)
"""

import torch
from pathlib import Path
import logging
import argparse
import sys
import subprocess

# Add project root to path
sys.path.append('/home/kyrie/Diffusion-UV')

from scripts.train_sdf_pretrain import train_sdf_network
from scripts.validate_sdf_normals import run_sdf_validation
from scripts.train_maiuvf_with_sdf import train_maiuvf_with_sdf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_sdf_experiment(
    mesh_constants_path: str,
    output_base_dir: str,
    sdf_epochs: int = 5,
    maiuvf_epochs: int = 10,
    surface_batch_size: int = 16384,
    off_surface_batch_size: int = 16384,
    batch_size: int = 16384,
    metric_weight: float = 0.01,
    anchor_weight: float = 1.0,
    cls_weight: float = 1.0,
    normal_reg_weight: float = 0.01,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Complete SDF + MA-IUVF experiment

    Steps:
    1. Pre-train SDF network (5 epochs)
    2. Validate SDF normals (3 metrics)
    3. Train MA-IUVF with SDF normals (10 epochs)
    4. Measure D_normal (Experiment 4)

    Args:
        mesh_constants_path: Path to mesh constants file
        output_base_dir: Base output directory
        sdf_epochs: SDF pre-training epochs
        maiuvf_epochs: MA-IUVF training epochs
        surface_batch_size: SDF surface batch size
        off_surface_batch_size: SDF off-surface batch size
        batch_size: MA-IUVF batch size
        metric_weight: Metric loss weight
        anchor_weight: Anchor loss weight
        cls_weight: Classification loss weight
        normal_reg_weight: Normal regularization weight
        lr: Learning rate
        device: Computing device
        seed: Random seed

    Returns:
        best_cls_acc: Best classification accuracy achieved
    """
    logger.info("=" * 80)
    logger.info("End-to-End SDF + MA-IUVF Experiment")
    logger.info("=" * 80)
    logger.info(f"Mesh constants: {mesh_constants_path}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"SDF epochs: {sdf_epochs}")
    logger.info(f"MA-IUVF epochs: {maiuvf_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Pre-train SDF network
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Pre-training SDF Network")
    logger.info("=" * 80)

    sdf_output_dir = output_base_dir / "sdf_pretrain"

    try:
        best_sdf_loss = train_sdf_network(
            mesh_constants_path=mesh_constants_path,
            output_dir=str(sdf_output_dir),
            num_epochs=sdf_epochs,
            surface_batch_size=surface_batch_size,
            off_surface_batch_size=off_surface_batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )

        logger.info(f"SDF pre-training completed. Best loss: {best_sdf_loss:.6f}")
    except Exception as e:
        logger.error(f"SDF pre-training failed: {e}")
        logger.warning("Continuing anyway (using existing checkpoint if available)...")

    sdf_checkpoint = sdf_output_dir / 'best.pt'

    if not sdf_checkpoint.exists():
        logger.error(f"SDF checkpoint not found: {sdf_checkpoint}")
        logger.error("Cannot continue without SDF network!")
        return None

    # Step 2: Validate SDF normals
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Validating SDF Normals")
    logger.info("=" * 80)

    sdf_validation_dir = output_base_dir / "sdf_validation"

    try:
        validation_passed = run_sdf_validation(
            sdf_checkpoint=str(sdf_checkpoint),
            mesh_constants_path=mesh_constants_path,
            output_dir=str(sdf_validation_dir),
            device=device,
        )

        if validation_passed:
            logger.info("✅ SDF validation passed!")
        else:
            logger.warning("⚠️  SDF validation needs improvement")
            logger.warning("Continuing anyway...")
    except Exception as e:
        logger.error(f"SDF validation failed: {e}")
        logger.warning("Continuing anyway...")

    # Step 3: Train MA-IUVF with SDF normals
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Training MA-IUVF with SDF Normals")
    logger.info("=" * 80)

    maiuvf_output_dir = output_base_dir / "maiuvf_with_sdf"
    maiuvf_checkpoint = maiuvf_output_dir / 'best.pt'

    try:
        best_cls_acc = train_maiuvf_with_sdf(
            sdf_checkpoint=str(sdf_checkpoint),
            mesh_constants_path=mesh_constants_path,
            output_dir=str(maiuvf_output_dir),
            num_epochs=maiuvf_epochs,
            batch_size=batch_size,
            metric_weight=metric_weight,
            anchor_weight=anchor_weight,
            cls_weight=cls_weight,
            normal_reg_weight=normal_reg_weight,
            lr=lr,
            device=device,
            seed=seed,
        )

        logger.info(f"MA-IUVF training completed. Best accuracy: {best_cls_acc:.4f}")
    except Exception as e:
        logger.error(f"MA-IUVF training failed: {e}")
        return None

    # Step 4: Measure D_normal (Experiment 4)
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Measuring D_normal (Experiment 4)")
    logger.info("=" * 80)

    exp4_output_dir = output_base_dir / "exp4_normal_noise"

    try:
        # Run Experiment 4
        logger.info("Running Experiment 4 to measure D_normal...")

        from src.analysis.experiments.exp4_normal_noise import run_experiment

        run_experiment(
            checkpoint=str(maiuvf_checkpoint),
            mesh_constants_path=mesh_constants_path,
            output_dir=str(exp4_output_dir),
            device=device,
        )

        logger.info(f"Experiment 4 completed. Results: {exp4_output_dir}/report.md")

        # Try to read and display D_normal results
        try:
            import pandas as pd

            report_path = exp4_output_dir / "report.md"
            if report_path.exists():
                logger.info(f"\nExperiment 4 Results:")
                with open(report_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if "Mean D_normal" in line or "D_normal" in line:
                        logger.info(f"  {line.strip()}")
        except:
            logger.warning("Could not read Experiment 4 report")

    except Exception as e:
        logger.error(f"Experiment 4 failed: {e}")
        logger.warning("Continuing anyway...")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Experiment Complete!")
    logger.info("=" * 80)
    logger.info(f"Classification Accuracy: {best_cls_acc*100:.2f}%")
    logger.info(f"  - Baseline (no tangent): 97.52%")
    logger.info(f"  - Mesh normals (tangent): 95.62%")
    logger.info(f"  - SDF normals (tangent): {best_cls_acc*100:.2f}%")
    logger.info(f"\nExperiment 4 results: {exp4_output_dir}/report.md")
    logger.info(f"SDF validation results: {sdf_validation_dir}/")
    logger.info(f"Checkpoints:")
    logger.info(f"  - SDF: {sdf_checkpoint}")
    logger.info(f"  - MA-IUVF: {maiuvf_checkpoint}")
    logger.info("=" * 80)

    return best_cls_acc


def main():
    parser = argparse.ArgumentParser(description="Run complete SDF + MA-IUVF experiment")
    parser.add_argument(
        "--mesh-constants",
        type=str,
        required=True,
        help="Path to mesh constants file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sdf_maiuvf_experiment",
        help="Base output directory"
    )
    parser.add_argument(
        "--sdf-epochs",
        type=int,
        default=5,
        help="SDF pre-training epochs"
    )
    parser.add_argument(
        "--maiuvf-epochs",
        type=int,
        default=10,
        help="MA-IUVF training epochs"
    )
    parser.add_argument(
        "--surface-batch-size",
        type=int,
        default=16384,
        help="SDF surface batch size"
    )
    parser.add_argument(
        "--off-surface-batch-size",
        type=int,
        default=16384,
        help="SDF off-surface batch size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
        help="MA-IUVF batch size"
    )
    parser.add_argument(
        "--metric-weight",
        type=float,
        default=0.01,
        help="Metric loss weight"
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=1.0,
        help="Anchor loss weight"
    )
    parser.add_argument(
        "--cls-weight",
        type=float,
        default=1.0,
        help="Classification loss weight"
    )
    parser.add_argument(
        "--normal-reg-weight",
        type=float,
        default=0.01,
        help="Normal gradient regularization weight"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computing device"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Run complete experiment
    best_cls_acc = run_complete_sdf_experiment(
        mesh_constants_path=args.mesh_constants,
        output_base_dir=args.output_dir,
        sdf_epochs=args.sdf_epochs,
        maiuvf_epochs=args.maiuvf_epochs,

        if validation_passed:
            logger.info("✅ SDF validation passed!")
        else:
            logger.warning("⚠️  SDF validation needs improvement")
            logger.warning("Continuing anyway...")
    except Exception as e:
        logger.error(f"SDF validation failed: {e}")
        logger.warning("Continuing anyway...")

    # Step 3: Train MA-IUVF with SDF normals
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Training MA-IUVF with SDF Normals")
    logger.info("=" * 80)

    maiuvf_output_dir = output_base_dir / "maiuvf_with_sdf"
    maiuvf_checkpoint = maiuvf_output_dir / 'best.pt'

    try:
        best_cls_acc = train_maiuvf_with_sdf(
            sdf_checkpoint=str(sdf_checkpoint),
            mesh_constants_path=mesh_constants_path,
            output_dir=str(maiuvf_output_dir),
            num_epochs=maiuvf_epochs,
            batch_size=batch_size,
            metric_weight=metric_weight,
            anchor_weight=anchor_weight,
            cls_weight=cls_weight,
            normal_reg_weight=normal_reg_weight,
            lr=lr,
            device=device,
            seed=seed,
        )

        logger.info(f"MA-IUVF training completed. Best accuracy: {best_cls_acc:.4f}")
    except Exception as e:
        logger.error(f"MA-IUVF training failed: {e}")
        return None

    # Step 4: Measure D_normal (Experiment 4)
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Measuring D_normal (Experiment 4)")
    logger.info("=" * 80)

    exp4_output_dir = output_base_dir / "exp4_normal_noise"

    try:
        # Run Experiment 4
        logger.info("Running Experiment 4 to measure D_normal...")

        from src.analysis.experiments.exp4_normal_noise import run_experiment

        run_experiment(
            checkpoint=str(maiuvf_checkpoint),
            mesh_constants_path=mesh_constants_path,
            output_dir=str(exp4_output_dir),
            device=device,
        )

        logger.info(f"Experiment 4 completed. Results: {exp4_output_dir}/report.md")

        # Try to read and display D_normal results
        try:
            import pandas as pd

            report_path = exp4_output_dir / "report.md"
            if report_path.exists():
                logger.info(f"\nExperiment 4 Results:")
                with open(report_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Mean D_normal" in line or "D_normal" in line:
                            logger.info(f"  {line.strip()}")
        except:
            logger.warning("Could not read Experiment 4 report")

    except Exception as e:
        logger.error(f"Experiment 4 failed: {e}")
        logger.warning("Continuing anyway...")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Experiment Complete!")
    logger.info("=" * 80)
    logger.info(f"Classification Accuracy: {best_cls_acc*100:.2f}%")
    logger.info(f"  - Baseline (no tangent): 97.52%")
    logger.info(f"  - Mesh normals (tangent): 95.62%")
    logger.info(f"  - SDF normals (tangent): {best_cls_acc*100:.2f}%")
    logger.info(f"\nExperiment 4 results: {exp4_output_dir}/report.md")
    logger.info(f"SDF validation results: {sdf_validation_dir}/")
    logger.info(f"Checkpoints:")
    logger.info(f"  - SDF: {sdf_checkpoint}")
    logger.info(f"  - MA-IUVF: {maiuvf_checkpoint}")
    logger.info("=" * 80)

    return best_cls_acc


def main():
    parser = argparse.ArgumentParser(description="Run complete SDF + MA-IUVF experiment")
    parser.add_argument(
        "--mesh-constants",
        type=str,
        required=True,
        help="Path to mesh constants file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sdf_maiuvf_experiment",
        help="Base output directory"
    )
    parser.add_argument(
        "--sdf-epochs",
        type=int,
        default=5,
        help="SDF pre-training epochs"
    )
    parser.add_argument(
        "--maiuvf-epochs",
        type=int,
        default=10,
        help="MA-IUVF training epochs"
    )
    parser.add_argument(
        "--surface-batch-size",
        type=int,
        default=16384,
        help="SDF surface batch size"
    )
    parser.add_argument(
        "--off-surface-batch-size",
        type=int,
        default=16384,
        help="SDF off-surface batch size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
        help="MA-IUVF batch size"
    )
    parser.add_argument(
        "--metric-weight",
        type=float,
        default=0.01,
        help="Metric loss weight"
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=1.0,
        help="Anchor loss weight"
    )
    parser.add_argument(
        "--cls-weight",
        type=float,
        default=1.0,
        help="Classification loss weight"
    )
    parser.add_argument(
        "--normal-reg-weight",
        type=float,
        default=0.01,
        help="Normal gradient regularization weight"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computing device"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Run complete experiment
    best_cls_acc = run_complete_sdf_experiment(
        mesh_constants_path=args.mesh_constants,
        output_base_dir=args.output_dir,
        sdf_epochs=args.sdf_epochs,
        maiuvf_epochs=args.maiuvf_epochs,
        surface_batch_size=args.surface_batch_size,
        off_surface_batch_size=args.off_surface_batch_size,
        batch_size=args.batch_size,
        metric_weight=args.metric_weight,
        anchor_weight=args.anchor_weight,
        cls_weight=args.cls_weight,
        normal_reg_weight=args.normal_reg_weight,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    if best_cls_acc is not None:
        logger.info(f"\n✅ Experiment completed successfully!")
        logger.info(f"Final accuracy: {best_cls_acc*100:.2f}%")
        sys.exit(0)
    else:
        logger.error(f"\n❌ Experiment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
