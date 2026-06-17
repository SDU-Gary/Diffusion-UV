"""
SIREN Unit Sphere Test

This script validates SIREN architecture on simple geometry (unit sphere)
before testing on complex meshes.

Test: SDF(x) = ||x|| - 1.0 (unit sphere)

Expected results:
- Cosine similarity > 0.95 (gradient direction matches radial direction)
- Gradient norm ≈ 1.0 (Eikonal satisfied)
- Training converges in < 20 epochs
"""

import torch
import torch.optim as optim
from pathlib import Path
import logging
import sys
import numpy as np

sys.path.append('/home/kyrie/Diffusion-UV')

from src.models.sdf_network import create_sdf_network_siren

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_unit_sphere_data(num_samples, device="cuda"):
    """Generate unit sphere training data"""
    # Sample in [-1.5, 1.5] range
    positions = torch.rand(num_samples, 3, device=device) * 3.0 - 1.5

    # GT SDF: ||x|| - 1.0
    distances = torch.norm(positions, dim=-1)
    sdf_gt = distances - 1.0

    # GT Normal: x / ||x||
    normals_gt = positions / (distances.unsqueeze(-1) + 1e-8)

    return positions, sdf_gt, normals_gt


def validate_gradient_properties(sdf_net, num_samples=100000, device="cuda"):
    """Validate gradient properties"""
    logger.info("=" * 80)
    logger.info("验证梯度属性")
    logger.info("=" * 80)

    sdf_net.eval()

    # Generate test points near surface
    positions = torch.rand(num_samples, 3, device=device) * 2.0 - 1.0
    positions = positions / torch.norm(positions, dim=-1, keepdim=True)
    positions = positions + torch.randn_like(positions) * 0.01

    positions_req = positions.clone().detach()
    positions_req.requires_grad_(True)

    # Compute SDF and gradients
    sdf_pred = sdf_net(positions_req)
    grad = torch.autograd.grad(
        outputs=sdf_pred.sum(),
        inputs=positions_req,
        create_graph=False,
    )[0]

    # Gradient norm
    grad_norm = torch.norm(grad, dim=-1)

    # Cosine similarity with position vectors
    positions_normalized = positions / (torch.norm(positions, dim=-1, keepdim=True) + 1e-8)
    grad_normalized = torch.nn.functional.normalize(grad, dim=-1, eps=1e-8)
    cosine_sim = torch.sum(grad_normalized * positions_normalized, dim=-1)

    results = {
        'grad_norm_mean': grad_norm.mean().item(),
        'grad_norm_std': grad_norm.std().item(),
        'cosine_mean': cosine_sim.mean().item(),
        'cosine_std': cosine_sim.std().item(),
    }

    logger.info(f"Gradient norm: {results['grad_norm_mean']:.6f} ± {results['grad_norm_std']:.6f}")
    logger.info(f"Cosine similarity: {results['cosine_mean']:.6f} ± {results['cosine_std']:.6f}")

    logger.info("=" * 80)
    logger.info("Validation Results")
    logger.info("=" * 80)

    if abs(results['grad_norm_mean'] - 1.0) < 0.01:
        logger.info(f"✅ Eikonal satisfied: norm = {results['grad_norm_mean']:.6f} ≈ 1.0")
    else:
        logger.warning(f"⚠️  Eikonal deviation: norm = {results['grad_norm_mean']:.6f} (expect 1.0)")

    if results['cosine_mean'] > 0.95:
        logger.info(f"✅ Direction correct: cosine = {results['cosine_mean']:.6f} > 0.95")
    else:
        logger.error(f"❌ Direction error: cosine = {results['cosine_mean']:.6f} (expect > 0.95)")

    return results


def train_unit_sphere(
    output_dir,
    num_epochs=20,
    batch_size=16384,
    lr=1e-3,
    hidden_dim=128,
    num_layers=5,
    omega_0=30.0,
    device="cuda",
):
    """Train SIREN network to fit unit sphere"""
    logger.info("=" * 80)
    logger.info("SIREN Unit Sphere Test")
    logger.info("=" * 80)
    logger.info(f"Target: SDF(x) = ||x|| - 1.0")
    logger.info(f"Expected: ∇SDF norm=1, direction matches x (cosine=1)")
    logger.info(f"Config: hidden_dim={hidden_dim}, num_layers={num_layers}, omega_0={omega_0}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create SIREN network
    sdf_net = create_sdf_network_siren(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        omega_0=omega_0,
    ).to(device)

    optimizer = optim.Adam(sdf_net.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        sdf_net.train()

        epoch_loss = 0.0
        num_batches = 10

        for step in range(num_batches):
            positions, sdf_gt, _ = generate_unit_sphere_data(batch_size, device)

            # Forward
            sdf_pred = sdf_net(positions)

            # MSE Loss
            loss = torch.mean((sdf_pred - sdf_gt) ** 2)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches

        logger.info(f"Epoch {epoch}/{num_epochs}: Loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info(f"  New best loss: {best_loss:.6f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': sdf_net.state_dict(),
                'best_loss': best_loss,
                'config': {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'omega_0': omega_0,
                },
            }, Path(output_dir) / 'best.pt')

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info("=" * 80)

    return best_loss


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test SIREN on unit sphere")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of layers")
    parser.add_argument("--omega-0", type=float, default=30.0, help="Omega_0 parameter")
    parser.add_argument("--output-dir", type=str, default="outputs/siren_unit_sphere_test")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Train
    best_loss = train_unit_sphere(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=16384,
        lr=1e-3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=args.omega_0,
        device=args.device,
    )

    # Validate
    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading best model for validation")
    logger.info("=" * 80)

    checkpoint = torch.load(f"{args.output_dir}/best.pt")
    sdf_net = create_sdf_network_siren(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=args.omega_0,
    ).to(args.device)

    sdf_net.load_state_dict(checkpoint['model_state_dict'])
    sdf_net.eval()

    results = validate_gradient_properties(sdf_net, num_samples=100000, device=args.device)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Unit Sphere Test Conclusion")
    logger.info("=" * 80)

    success = (
        abs(results['grad_norm_mean'] - 1.0) < 0.01 and
        results['cosine_mean'] > 0.95
    )

    if success:
        logger.info("✅ Unit sphere test PASSED: SIREN architecture works correctly")
        logger.info(f"   Eikonal: {results['grad_norm_mean']:.6f} ≈ 1.0")
        logger.info(f"   Cosine: {results['cosine_mean']:.6f} > 0.95")
        logger.info("")
        logger.info("Can proceed to Stanford Bunny test")
    else:
        logger.error("❌ Unit sphere test FAILED: SIREN architecture needs adjustment")
        logger.error(f"   Eikonal: {results['grad_norm_mean']:.6f} (expect 1.0)")
        logger.error(f"   Cosine: {results['cosine_mean']:.6f} (expect > 0.95)")
        logger.error("")
        logger.error("Suggested fixes:")
        logger.error("   1. Adjust omega_0 (current {:.0f} → try 10-50)".format(args.omega_0))
        logger.error("   2. Increase hidden_dim (current {} → try 256)".format(args.hidden_dim))
        logger.error("   3. Increase num_layers (current {} → try 6-7)".format(args.num_layers))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
