"""
MA-IUVF Training with SDF Normals

This script trains MA-IUVF with SDF-derived normals for tangent space projection.

Key integration points:
1. Load frozen SDF network (pre-trained)
2. Get SDF normals via autograd (create_graph=False)
3. Use SDF normals for tangent space projection
4. Add normal gradient regularization (prevent D_normal explosion)

Training strategy:
- Decoupled: SDF is frozen, only MA-IUVF is trained
- Smooth normals: SDF provides C^∞ continuous normals
- Regularized: Light normal regularization prevents zero-space explosion

Expected results:
- Classification accuracy > 97.52% (baseline)
- D_normal reduced from 1.427 (mesh normals) → < 0.5 (SDF normals)
"""

import torch
import torch.optim as optim
from pathlib import Path
import logging
import argparse
import sys
import time

# Add project root to path
sys.path.append('/home/kyrie/Diffusion-UV')

from src.models.sdf_network import SDFNetwork
from src.models.metric_aligned_iuv_field import create_model
from src.data.gpu_dataset import GPUDynamicSampleGenerator
from src.training.metric_aligned_iuv_losses import (
    gather_chart_uvs,
    compute_uv_jacobian,
    project_to_tangent_space,
    compute_anchor_loss,
    compute_chart_com_loss,
    compute_classification_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_maiuvf_with_sdf(
    sdf_checkpoint: str,
    mesh_constants_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 16384,
    metric_weight: float = 0.01,
    anchor_weight: float = 1.0,
    com_weight: float = 0.0,
    cls_weight: float = 1.0,
    normal_reg_weight: float = 0.01,
    lr: float = 1e-3,
    hash_lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Train MA-IUVF with SDF normals (tangent space projection)

    Args:
        sdf_checkpoint: Path to pre-trained SDF network checkpoint
        mesh_constants_path: Path to mesh constants file
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        metric_weight: Metric loss weight
        anchor_weight: Anchor loss weight
        com_weight: Center of mass loss weight
        cls_weight: Classification loss weight
        normal_reg_weight: Normal gradient regularization weight
        lr: Learning rate for MLP
        hash_lr: Learning rate for hash grid
        device: Computing device
        seed: Random seed
    """
    logger.info("=" * 80)
    logger.info("MA-IUVF Training with SDF Normals")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  SDF checkpoint: {sdf_checkpoint}")
    logger.info(f"  mesh_constants: {mesh_constants_path}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  num_epochs: {num_epochs}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  metric_weight: {metric_weight}")
    logger.info(f"  anchor_weight: {anchor_weight}")
    logger.info(f"  cls_weight: {cls_weight}")
    logger.info(f"  normal_reg_weight: {normal_reg_weight}")
    logger.info(f"  lr: {lr}, hash_lr: {hash_lr}")
    logger.info(f"  device: {device}")
    logger.info(f"  seed: {seed}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Step 1: Load frozen SDF network
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Frozen SDF Network")
    logger.info("=" * 80)

    # Load mesh metadata for SDF bbox
    from src.data.gpu_constant_baker import load_mesh_constants
    constants, metadata = load_mesh_constants(mesh_constants_path, device=device)
    bbox_min = metadata["bbox_min"]
    bbox_max = metadata["bbox_max"]

    sdf_net = SDFNetwork(
        num_levels=8,
        log2_hashmap_size=12,
        base_res=8,
        max_res=128,
        hidden_dim=32,
        num_layers=2,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        cuda_backend="torch",
    ).to(device)

    sdf_checkpoint_data = torch.load(sdf_checkpoint, map_location=device)
    sdf_net.load_state_dict(sdf_checkpoint_data['model_state_dict'])

    # Freeze SDF network
    sdf_net.eval()
    for param in sdf_net.parameters():
        param.requires_grad = False

    logger.info(f"Loaded SDF network from epoch {sdf_checkpoint_data.get('epoch', 'unknown')}")
    logger.info("SDF network frozen (no gradient flow to SDF)")

    # Step 2: Create MA-IUVF model
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating MA-IUVF Model")
    logger.info("=" * 80)

    model = create_model(
        num_charts=8,
        hidden_dim=64,
        num_layers=2,
        encoder_type="bspline_hash",
        hash_num_levels=16,
        hash_features_per_level=2,
        hash_log2_size=19,
        hash_base_res=16,
        hash_max_res=2048,
        hash_cuda_backend="auto",
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        activation="softplus",
    ).to(device)

    logger.info(f"Created MA-IUVF model: {model.get_num_params():,} parameters")

    # Step 3: Create data generator
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Creating Data Generator")
    logger.info("=" * 80)

    data_gen = GPUDynamicSampleGenerator(
        mesh_constants_path=mesh_constants_path,
        batch_size=batch_size,
        sigma_ratio=0.01,
        device=device,
        seed=seed,
    )

    # Step 4: Create optimizer
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Creating Optimizer")
    logger.info("=" * 80)

    param_groups = model.get_optimizer_param_groups(
        lr=lr,
        hash_lr=hash_lr,
        hash_weight_decay=1e-6,
        mlp_weight_decay=0.0,
    )

    optimizer = optim.Adam(param_groups)

    logger.info(f"Optimizer: Adam (lr={lr}, hash_lr={hash_lr})")

    # Step 5: Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Starting Training")
    logger.info("=" * 80)

    best_cls_acc = 0.0
    steps_per_epoch = 10

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_start = time.time()

        epoch_loss = 0.0
        epoch_metric_loss = 0.0
        epoch_normal_reg_loss = 0.0
        epoch_anchor_loss = 0.0
        epoch_cls_loss = 0.0
        total_cls_correct = 0
        total_cls_samples = 0
        num_batches = 0

        for step in range(steps_per_epoch):
            # Generate batch
            batch = data_gen.next_batch()

            # Clone positions and enable gradients
            pos = batch['pos'].clone().detach()  # [B, 3]
            pos.requires_grad_(True)

            j_3d_gt = batch['j_3d_gt']
            uv_anchor = batch['uv_anchor']
            chart_id = batch['chart_id']

            # Get SDF normals (frozen, no gradient to SDF)
            with torch.enable_grad():
                sdf_vals = sdf_net(pos)
                sdf_normals = torch.autograd.grad(
                    outputs=sdf_vals.sum(),
                    inputs=pos,
                    create_graph=False,  # Don't need graph for normals
                    retain_graph=True,
                )[0]  # [B, 3]

                # Normalize (should already be ~1.0 due to Eikonal)
                sdf_normals = torch.nn.functional.normalize(sdf_normals, dim=-1, eps=1e-6)
                sdf_normals = sdf_normals.detach()  # Detach to prevent gradient to SDF

            # Forward pass MA-IUVF
            model_output = model(pos)

            # Compute UV Jacobian
            selected_uv = gather_chart_uvs(model_output.uv_preds, chart_id)  # [B, 2]
            j_pred = compute_uv_jacobian(selected_uv, pos)  # [B, 2, 3]

            # Tangent space projection with SDF normals
            j_pred_tangent = project_to_tangent_space(j_pred, sdf_normals)
            j_gt_tangent = project_to_tangent_space(j_3d_gt, sdf_normals)

            # Metric loss in tangent space
            metric_loss = torch.mean((j_pred_tangent - j_gt_tangent) ** 2)

            # Normal gradient regularization (prevent D_normal explosion)
            j_normal = torch.bmm(j_pred, sdf_normals.unsqueeze(2))  # [B, 2, 1]
            normal_reg_loss = torch.mean(j_normal ** 2)

            # Other losses
            anchor_loss = compute_anchor_loss(selected_uv, uv_anchor)
            com_loss = compute_chart_com_loss(selected_uv, uv_anchor, chart_id, 8)
            cls_loss = compute_classification_loss(model_output.logits, chart_id)

            # Total loss
            total_loss = (
                metric_weight * metric_loss +
                normal_reg_weight * normal_reg_loss +
                anchor_weight * anchor_loss +
                com_weight * com_loss +
                cls_weight * cls_loss
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistics
            pred_chart_id = model_output.logits.argmax(-1)
            cls_correct = (pred_chart_id == chart_id).sum().item()

            epoch_loss += total_loss.item()
            epoch_metric_loss += metric_loss.item()
            epoch_normal_reg_loss += normal_reg_loss.item()
            epoch_anchor_loss += anchor_loss.item()
            epoch_cls_loss += cls_loss.item()
            total_cls_correct += cls_correct
            total_cls_samples += pos.shape[0]
            num_batches += 1

        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_metric_loss = epoch_metric_loss / num_batches
        avg_normal_reg_loss = epoch_normal_reg_loss / num_batches
        avg_anchor_loss = epoch_anchor_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_cls_acc = total_cls_correct / total_cls_samples
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{num_epochs}: "
            f"Loss={avg_loss:.4f}, ClsAcc={avg_cls_acc:.4f}, "
            f"Time={epoch_time:.2f}s"
        )
        logger.info(
            f"  Losses: metric={avg_metric_loss:.6f}, "
            f"normal_reg={avg_normal_reg_loss:.6f}, "
            f"anchor={avg_anchor_loss:.6f}, "
            f"cls={avg_cls_loss:.6f}"
        )

        # Save best checkpoint
        if avg_cls_acc > best_cls_acc:
            best_cls_acc = avg_cls_acc
            logger.info(f"  New best cls acc: {best_cls_acc:.4f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cls_acc': best_cls_acc,
                'config': {
                    'metric_weight': metric_weight,
                    'anchor_weight': anchor_weight,
                    'cls_weight': cls_weight,
                    'normal_reg_weight': normal_reg_weight,
                },
            }, Path(output_dir) / 'best.pt')

            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, Path(output_dir) / 'latest.pt')

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Completed!")
    logger.info("=" * 80)
    logger.info(f"Best classification accuracy: {best_cls_acc:.4f} ({best_cls_acc*100:.2f}%)")
    logger.info(f"Baseline accuracy: 97.52%")
    logger.info(f"Mesh normals accuracy: 95.62%")
    logger.info(f"SDF normals accuracy: {best_cls_acc*100:.2f}%")
    logger.info(f"Checkpoint saved to: {Path(output_dir) / 'best.pt'}")
    logger.info("=" * 80)

    return best_cls_acc


def main():
    parser = argparse.ArgumentParser(description="Train MA-IUVF with SDF normals")
    parser.add_argument(
        "--sdf-checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained SDF network checkpoint"
    )
    parser.add_argument(
        "--mesh-constants",
        type=str,
        required=True,
        help="Path to mesh constants file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/maiuvf_with_sdf",
        help="Output directory"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
        help="Batch size"
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
        help="Learning rate for MLP"
    )
    parser.add_argument(
        "--hash-lr",
        type=float,
        default=1e-3,
        help="Learning rate for hash grid"
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

    # Train MA-IUVF with SDF normals
    train_maiuvf_with_sdf(
        sdf_checkpoint=args.sdf_checkpoint,
        mesh_constants_path=args.mesh_constants,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        metric_weight=args.metric_weight,
        anchor_weight=args.anchor_weight,
        cls_weight=args.cls_weight,
        normal_reg_weight=args.normal_reg_weight,
        lr=args.lr,
        hash_lr=args.hash_lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
