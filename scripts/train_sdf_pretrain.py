"""
SDF Network Pre-training Script

This script pre-trains a lightweight SDF network to provide C^∞ continuous
normals for tangent space projection in MA-IUVF.

Training strategy:
- Surface samples: SDF = 0 (on mesh surface)
- Off-surface samples: Eikonal constraint (||∇SDF|| = 1)
- Decoupled training: SDF is trained independently before MA-IUVF

Expected convergence: < 10 epochs
Expected surface loss: < 1e-3
Expected eikonal loss: < 1e-4
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

from src.models.sdf_network import create_sdf_network
from src.data.sdf_data_generator import create_sdf_data_generator
from src.training.sdf_losses import compute_sdf_loss, validate_sdf_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_sdf_network(
    mesh_constants_path: str,
    output_dir: str,
    num_epochs: int = 5,
    surface_batch_size: int = 16384,
    off_surface_batch_size: int = 16384,
    off_surface_sigma_ratio: float = 0.02,
    lr: float = 1e-3,
    lambda_eikonal: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
    use_siren: bool = False,
    hidden_dim: int = 128,
    num_layers: int = 5,
    omega_0: float = 30.0,
):
    """
    Pre-train SDF network

    Args:
        mesh_constants_path: Path to mesh constants file
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs (default: 5)
        surface_batch_size: Surface samples per batch (default: 16384)
        off_surface_batch_size: Off-surface samples per batch (default: 16384)
        off_surface_sigma_ratio: Gaussian offset ratio (default: 0.02)
        lr: Learning rate (default: 1e-3)
        lambda_eikonal: Eikonal loss weight (default: 0.1)
        device: Computing device (default: "cuda")
        seed: Random seed (default: 42)
        use_siren: Use SIREN architecture (default: False)
        hidden_dim: Hidden dim for SIREN (default: 128)
        num_layers: Number of layers for SIREN (default: 5)
        omega_0: Omega_0 frequency for SIREN (default: 30.0)
    """
    logger.info("=" * 80)
    logger.info("SDF Network Pre-training")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  mesh_constants: {mesh_constants_path}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  num_epochs: {num_epochs}")
    logger.info(f"  surface_batch_size: {surface_batch_size}")
    logger.info(f"  off_surface_batch_size: {off_surface_batch_size}")
    logger.info(f"  off_surface_sigma_ratio: {off_surface_sigma_ratio}")
    logger.info(f"  lr: {lr}")
    logger.info(f"  lambda_eikonal: {lambda_eikonal}")
    logger.info(f"  device: {device}")
    logger.info(f"  seed: {seed}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create SDF network
    logger.info("\nInitializing SDF network...")

    if use_siren:
        # Use SIREN architecture
        from src.models.sdf_network import create_sdf_network_siren
        sdf_net = create_sdf_network_siren(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            omega_0=omega_0,
        ).to(device)
        logger.info(f"Using SIREN architecture: {sdf_net.get_num_params():,} parameters")
    else:
        # Use B-Spline Hash Grid architecture (original)
        sdf_net = create_sdf_network(
            num_levels=8,
            log2_hashmap_size=12,
            base_res=8,
            max_res=128,
            hidden_dim=32,
            num_layers=2,
            cuda_backend="torch",
        ).to(device)
        logger.info(f"Using Hash Grid architecture: {sdf_net.get_num_params():,} parameters")

    # Create data generator
    logger.info("Initializing SDF data generator...")
    data_gen = create_sdf_data_generator(
        mesh_constants_path=mesh_constants_path,
        surface_batch_size=surface_batch_size,
        off_surface_batch_size=off_surface_batch_size,
        off_surface_sigma_ratio=off_surface_sigma_ratio,
        device=device,
        seed=seed,
    )

    # Optimizer (Adam)
    optimizer = optim.Adam(sdf_net.parameters(), lr=lr)

    # Training loop
    logger.info("\nStarting training...")
    best_loss = float('inf')
    steps_per_epoch = 10

    for epoch in range(1, num_epochs + 1):
        sdf_net.train()
        epoch_start = time.time()

        epoch_loss = 0.0
        epoch_surface_loss = 0.0
        epoch_eikonal_loss = 0.0
        num_batches = 0

        for step in range(steps_per_epoch):
            # Generate batch
            batch = data_gen.next_batch()

            surface_pos = batch["surface_pos"]  # [B_surf, 3]
            off_surface_pos = batch["off_surface_pos"]  # [B_off, 3]

            # Forward pass
            surface_pos.requires_grad_(False)
            off_surface_pos.requires_grad_(True)

            sdf_surface = sdf_net(surface_pos)  # [B_surf]
            sdf_off = sdf_net(off_surface_pos)  # [B_off]

            # Compute gradients for Eikonal loss
            grad_off = torch.autograd.grad(
                outputs=sdf_off.sum(),
                inputs=off_surface_pos,
                create_graph=True,
                retain_graph=True,
            )[0]  # [B_off, 3]

            # Compute loss
            loss_dict = compute_sdf_loss(
                sdf_pred_surface=sdf_surface,
                sdf_pred_off_surface=sdf_off,
                grad_off_surface=grad_off,
                lambda_eikonal=lambda_eikonal,
            )

            loss = loss_dict["total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            epoch_surface_loss += loss_dict["surface"].item()
            epoch_eikonal_loss += loss_dict["eikonal"].item()
            num_batches += 1

        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_surface_loss = epoch_surface_loss / num_batches
        avg_eikonal_loss = epoch_eikonal_loss / num_batches
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{num_epochs}: "
            f"Loss={avg_loss:.6f} (surf={avg_surface_loss:.6f}, eik={avg_eikonal_loss:.6f}), "
            f"Time={epoch_time:.2f}s"
        )

        # Validation
        if epoch % 2 == 0 or epoch == num_epochs:
            logger.info("  Validating...")
            # Generate validation batch
            val_batch = data_gen.next_batch()
            val_surface_pos = val_batch["surface_pos"]
            val_off_surface_pos = val_batch["off_surface_pos"]

            # Enable gradients for off-surface positions
            val_off_surface_pos_req = val_off_surface_pos.clone().detach()
            val_off_surface_pos_req.requires_grad_(True)

            val_sdf_surface = sdf_net(val_surface_pos)
            val_sdf_off = sdf_net(val_off_surface_pos_req)

            val_grad_off = torch.autograd.grad(
                outputs=val_sdf_off.sum(),
                inputs=val_off_surface_pos_req,
                create_graph=False,
            )[0]

            with torch.no_grad():
                val_metrics = validate_sdf_loss(val_sdf_surface, val_grad_off)

                logger.info(
                    f"    Surface: mean={val_metrics['surface_mean']:.6f}, "
                    f"max={val_metrics['surface_max']:.6f}"
                )
                logger.info(
                    f"    Gradient: mean={val_metrics['grad_mean']:.6f}, "
                    f"std={val_metrics['grad_std']:.6f}"
                )
                logger.info(f"    Eikonal error: {val_metrics['eikonal_error']:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info(f"  New best loss: {best_loss:.6f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': sdf_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': {
                    'num_levels': 8,
                    'log2_hashmap_size': 12,
                    'base_res': 8,
                    'max_res': 128,
                    'hidden_dim': 32,
                    'num_layers': 2,
                    'lambda_eikonal': lambda_eikonal,
                },
            }, Path(output_dir) / 'best.pt')

            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': sdf_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, Path(output_dir) / 'latest.pt')

    logger.info("\n" + "=" * 80)
    logger.info("SDF Pre-training Completed!")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Checkpoint saved to: {Path(output_dir) / 'best.pt'}")
    logger.info("=" * 80)

    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Pre-train SDF network")
    parser.add_argument(
        "--mesh-constants",
        type=str,
        required=True,
        help="Path to mesh constants file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sdf_pretrain",
        help="Output directory"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--surface-batch-size",
        type=int,
        default=16384,
        help="Surface samples per batch"
    )
    parser.add_argument(
        "--off-surface-batch-size",
        type=int,
        default=16384,
        help="Off-surface samples per batch"
    )
    parser.add_argument(
        "--off-surface-sigma-ratio",
        type=float,
        default=0.02,
        help="Gaussian offset ratio"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--lambda-eikonal",
        type=float,
        default=0.1,
        help="Eikonal loss weight"
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
    parser.add_argument(
        "--use-siren",
        action="store_true",
        help="Use SIREN architecture instead of Hash Grid"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer dimension for SIREN (default: 128)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Number of hidden layers for SIREN (default: 5)"
    )
    parser.add_argument(
        "--omega-0",
        type=float,
        default=30.0,
        help="Omega_0 frequency parameter for SIREN (default: 30.0)"
    )

    args = parser.parse_args()

    # Train SDF network
    train_sdf_network(
        mesh_constants_path=args.mesh_constants,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        surface_batch_size=args.surface_batch_size,
        off_surface_batch_size=args.off_surface_batch_size,
        off_surface_sigma_ratio=args.off_surface_sigma_ratio,
        lr=args.lr,
        lambda_eikonal=args.lambda_eikonal,
        device=args.device,
        seed=args.seed,
        use_siren=args.use_siren,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=args.omega_0,
    )


if __name__ == "__main__":
    main()
