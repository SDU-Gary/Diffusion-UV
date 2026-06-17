"""
实验2：直接法向监督（Normal Injection）

目的：验证网络是否被Eikonal Loss困在局部极小值

操作：
1. 降低 λ_eikonal 到极低（0.01）
2. 引入真实法线监督：L_normal = 1.0 - CosineSimilarity(∇SDF_surf, n_gt)
3. 训练50 epochs

预期：
如果加入L_normal后，表面余弦相似度飙升到0.99+，说明：
- 网络架构没问题
- 之前的失败纯粹是Eikonal Loss导致的局部极小值
"""

import torch
import torch.optim as optim
from pathlib import Path
import logging
import sys

sys.path.append('/home/kyrie/Diffusion-UV')

from src.models.sdf_network import SDFNetwork
from src.data.sdf_data_generator import SDFDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_loss_with_normal_supervision(
    sdf_pred_surface,
    sdf_pred_off_surface,
    grad_off_surface,
    grad_surface,
    normals_gt,
    lambda_eikonal=0.01,
    lambda_normal=1.0,
):
    """
    计算带法线监督的SDF Loss

    Args:
        sdf_pred_surface: [B_surf] 表面SDF预测
        sdf_pred_off_surface: [B_off] 离面SDF预测
        grad_off_surface: [B_off, 3] 离面梯度
        grad_surface: [B_surf, 3] 表面梯度
        normals_gt: [B_surf, 3] GT法线
        lambda_eikonal: Eikonal loss权重（极低）
        lambda_normal: 法线监督权重（高）

    Returns:
        loss_dict: Loss字典
    """
    # Surface loss: |SDF - 0|
    loss_surface = torch.mean(torch.abs(sdf_pred_surface))

    # Eikonal loss: ||∇SDF|| - 1|^2
    grad_norm = torch.norm(grad_off_surface, dim=-1)
    loss_eikonal = torch.mean((grad_norm - 1.0) ** 2)

    # Normal supervision loss: 1 - cosine_similarity
    grad_surface_norm = torch.nn.functional.normalize(grad_surface, dim=-1, eps=1e-6)
    normals_gt_norm = torch.nn.functional.normalize(normals_gt, dim=-1, eps=1e-6)
    cosine_sim = torch.sum(grad_surface_norm * normals_gt_norm, dim=-1)
    loss_normal = torch.mean(1.0 - cosine_sim)

    # Total loss
    total_loss = loss_surface + lambda_eikonal * loss_eikonal + lambda_normal * loss_normal

    return {
        "total": total_loss,
        "surface": loss_surface,
        "eikonal": loss_eikonal,
        "normal": loss_normal,
    }


def validate_surface_normals(
    sdf_net,
    data_gen,
    num_samples=100000,
    device="cuda",
):
    """
    验证表面法线余弦相似度
    """
    from src.data.gpu_constant_baker import load_mesh_constants

    logger.info("Validating surface normals...")

    # Load mesh constants
    constants, metadata = load_mesh_constants("data/models/bunny_mesh_constants.pt", map_location=device)
    face_vertices = constants["face_vertices"]
    face_normals = constants["face_normals"]
    face_probs = constants["face_probs"]

    # Normalize face probabilities
    if face_probs.sum() <= 0:
        raise ValueError("face_probs sum must be > 0")
    face_probs = face_probs / face_probs.sum()

    # Sample surface points
    face_idx = torch.multinomial(face_probs, num_samples, replacement=True)
    sel_verts = face_vertices[face_idx]
    sel_normals = face_normals[face_idx]

    # Barycentric interpolation
    u = torch.rand(num_samples, device=device)
    v = torch.rand(num_samples, device=device)
    is_over = (u + v) > 1.0
    u = torch.where(is_over, 1.0 - u, u)
    v = torch.where(is_over, 1.0 - v, v)
    w = 1.0 - u - v
    bary = torch.stack([u, v, w], dim=-1)

    surface_pos = torch.bmm(bary.unsqueeze(1), sel_verts).squeeze(1)
    mesh_normals = torch.bmm(bary.unsqueeze(1), sel_normals).squeeze(1)
    mesh_normals = torch.nn.functional.normalize(mesh_normals, dim=-1, eps=1e-6)

    # Get SDF normals
    sdf_net.eval()
    surface_pos_req = surface_pos.clone().detach()
    surface_pos_req.requires_grad_(True)

    sdf_vals = sdf_net(surface_pos_req)
    sdf_normals = torch.autograd.grad(
        outputs=sdf_vals.sum(),
        inputs=surface_pos_req,
        create_graph=False,
    )[0]

    sdf_normals = torch.nn.functional.normalize(sdf_normals, dim=-1, eps=1e-6)

    # Cosine similarity
    cosine_sim = torch.sum(sdf_normals * mesh_normals, dim=-1)

    logger.info(f"  Surface Cosine Similarity: {cosine_sim.mean():.4f} ± {cosine_sim.std():.4f}")
    logger.info(f"  Min: {cosine_sim.min():.4f}, Max: {cosine_sim.max():.4f}")

    # Gradient norm
    grad_norm = torch.norm(sdf_normals, dim=-1)
    logger.info(f"  SDF Gradient Norm: {grad_norm.mean():.4f} ± {grad_norm.std():.4f}")

    return cosine_sim.mean().item()


def train_with_normal_supervision(
    mesh_constants_path: str,
    output_dir: str,
    num_epochs: int = 50,
    surface_batch_size: int = 16384,
    off_surface_batch_size: int = 16384,
    off_surface_sigma_ratio: float = 0.02,
    lr: float = 1e-3,
    lambda_eikonal: float = 0.01,
    lambda_normal: float = 1.0,
    device: str = "cuda",
    seed: int = 42,
):
    """
    训练SDF网络with法线监督
    """
    logger.info("=" * 80)
    logger.info("实验2：直接法向监督")
    logger.info("=" * 80)
    logger.info(f"λ_eikonal: {lambda_eikonal} (极低)")
    logger.info(f"λ_normal: {lambda_normal} (高权重)")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create SDF network
    sdf_net = SDFNetwork(
        num_levels=8,
        log2_hashmap_size=12,
        base_res=8,
        max_res=128,
        hidden_dim=32,
        num_layers=2,
        cuda_backend="torch",
    ).to(device)

    # Data generator
    data_gen = SDFDataGenerator(
        mesh_constants_path=mesh_constants_path,
        surface_batch_size=surface_batch_size,
        off_surface_batch_size=off_surface_batch_size,
        off_surface_sigma_ratio=off_surface_sigma_ratio,
        device=device,
        seed=seed,
    )

    # Optimizer
    optimizer = optim.Adam(sdf_net.parameters(), lr=lr)

    best_cosine = -1.0

    for epoch in range(1, num_epochs + 1):
        sdf_net.train()

        epoch_loss = 0.0
        epoch_surface_loss = 0.0
        epoch_eikonal_loss = 0.0
        epoch_normal_loss = 0.0
        num_batches = 10

        for step in range(num_batches):
            # Generate batch
            batch = data_gen.next_batch()

            surface_pos = batch["surface_pos"]
            surface_normals_gt = batch["surface_normals"]
            off_surface_pos = batch["off_surface_pos"]

            # Forward
            surface_pos_req = surface_pos.clone().detach()
            surface_pos_req.requires_grad_(True)
            off_surface_pos_req = off_surface_pos.clone().detach()
            off_surface_pos_req.requires_grad_(True)

            sdf_surface = sdf_net(surface_pos_req)
            sdf_off = sdf_net(off_surface_pos_req)

            # Compute gradients
            grad_surface = torch.autograd.grad(
                outputs=sdf_surface.sum(),
                inputs=surface_pos_req,
                create_graph=True,
                retain_graph=True,
            )[0]

            grad_off = torch.autograd.grad(
                outputs=sdf_off.sum(),
                inputs=off_surface_pos_req,
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute loss with GT normals
            loss_dict = compute_loss_with_normal_supervision(
                sdf_pred_surface=sdf_surface,
                sdf_pred_off_surface=sdf_off,
                grad_off_surface=grad_off,
                grad_surface=grad_surface,
                normals_gt=surface_normals_gt,
                lambda_eikonal=lambda_eikonal,
                lambda_normal=lambda_normal,
            )

            loss = loss_dict["total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_surface_loss += loss_dict["surface"].item()
            epoch_eikonal_loss += loss_dict["eikonal"].item()
            epoch_normal_loss += loss_dict["normal"].item()

        avg_loss = epoch_loss / num_batches
        avg_surface_loss = epoch_surface_loss / num_batches
        avg_eikonal_loss = epoch_eikonal_loss / num_batches
        avg_normal_loss = epoch_normal_loss / num_batches

        logger.info(
            f"Epoch {epoch}/{num_epochs}: "
            f"Loss={avg_loss:.4f} (surf={avg_surface_loss:.4f}, eik={avg_eikonal_loss:.4f}, normal={avg_normal_loss:.4f})"
        )

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            logger.info("  Validating...")
            cosine_sim = validate_surface_normals(
                sdf_net, data_gen, num_samples=10000, device=device
            )

            if cosine_sim > best_cosine:
                best_cosine = cosine_sim
                logger.info(f"  New best cosine: {best_cosine:.4f}")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': sdf_net.state_dict(),
                    'best_cosine': best_cosine,
                }, Path(output_dir) / 'best.pt')

    logger.info("\n" + "=" * 80)
    logger.info("实验2完成")
    logger.info(f"Best surface cosine: {best_cosine:.4f}")
    logger.info("=" * 80)

    return best_cosine


def main():
    best_cosine = train_with_normal_supervision(
        mesh_constants_path="data/models/bunny_mesh_constants.pt",
        output_dir="outputs/exp2_normal_supervision",
        num_epochs=50,
        surface_batch_size=16384,
        off_surface_batch_size=16384,
        off_surface_sigma_ratio=0.02,
        lr=1e-3,
        lambda_eikonal=0.01,
        lambda_normal=1.0,
        device="cuda",
        seed=42,
    )

    logger.info("\n" + "=" * 80)
    logger.info("实验2结论")
    logger.info("=" * 80)

    if best_cosine > 0.99:
        logger.info("✅ 实验2成功：表面余弦相似度 > 0.99")
        logger.info("   确认假设二：哈希网格高频噪声陷阱")
        logger.info("   网络架构没问题，被Eikonal Loss困在局部极小值")
    else:
        logger.warning(f"⚠️ 实验2部分成功：余弦相似度 = {best_cosine:.4f}")
        logger.warning("   需要更多训练或调整λ_normal")


if __name__ == "__main__":
    main()
