"""
实验1：单位球体解析梯度测试

目的：验证自动微分逻辑和坐标空间映射是否正确

原理：
- 让网络拟合解析式：SDF(x) = ||x|| - 1.0（单位球）
- 预期：∇SDF 的模长 = 1，方向与 x 一致（cosine = 1）
- 如果失败：说明自动微分或坐标归一化有bug
"""

import torch
import torch.optim as optim
from pathlib import Path
import logging
import sys
import numpy as np

# Add project root to path
sys.path.append('/home/kyrie/Diffusion-UV')

from src.models.sdf_network import SDFNetwork

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_unit_sphere_data(num_samples: int, device: str = "cuda"):
    """
    生成单位球训练数据

    Args:
        num_samples: 样本数量
        device: 设备

    Returns:
        positions: [B, 3] 随机位置（在[-1.5, 1.5]范围内）
        sdf_gt: [B] GT SDF值 = ||x|| - 1.0
        normals_gt: [B, 3] GT法线 = x / ||x||
    """
    # 在[-1.5, 1.5]范围内随机采样
    positions = torch.rand(num_samples, 3, device=device) * 3.0 - 1.5

    # GT SDF: ||x|| - 1.0
    distances = torch.norm(positions, dim=-1)
    sdf_gt = distances - 1.0

    # GT Normal: x / ||x||（只在表面有定义，内部点用此近似）
    normals_gt = positions / (distances.unsqueeze(-1) + 1e-8)

    return positions, sdf_gt, normals_gt


def validate_gradient_properties(
    sdf_net: SDFNetwork,
    num_samples: int = 10000,
    device: str = "cuda",
):
    """
    验证梯度属性

    检查：
    1. 梯度模长是否接近1（Eikonal）
    2. 梯度方向是否与位置向量一致（cosine similarity）

    Args:
        sdf_net: SDF网络
        num_samples: 验证样本数
        device: 设备

    Returns:
        results: 验证结果字典
    """
    logger.info("\n" + "=" * 80)
    logger.info("验证梯度属性")
    logger.info("=" * 80)

    sdf_net.eval()

    # 生成测试点（在表面附近）
    positions = torch.rand(num_samples, 3, device=device) * 2.0 - 1.0
    positions = positions / torch.norm(positions, dim=-1, keepdim=True)  # 归一化到单位球表面
    positions = positions + torch.randn_like(positions) * 0.01  # 添加小扰动

    positions_req = positions.clone().detach()
    positions_req.requires_grad_(True)

    # 计算SDF和梯度
    sdf_pred = sdf_net(positions_req)
    grad = torch.autograd.grad(
        outputs=sdf_pred.sum(),
        inputs=positions_req,
        create_graph=False,
    )[0]  # [B, 3]

    # 梯度模长
    grad_norm = torch.norm(grad, dim=-1)

    # 与位置向量的余弦相似度
    positions_normalized = positions / (torch.norm(positions, dim=-1, keepdim=True) + 1e-8)
    grad_normalized = torch.nn.functional.normalize(grad, dim=-1, eps=1e-8)
    cosine_sim = torch.sum(grad_normalized * positions_normalized, dim=-1)

    results = {
        'grad_norm_mean': grad_norm.mean().item(),
        'grad_norm_std': grad_norm.std().item(),
        'grad_norm_min': grad_norm.min().item(),
        'grad_norm_max': grad_norm.max().item(),
        'cosine_mean': cosine_sim.mean().item(),
        'cosine_std': cosine_sim.std().item(),
        'cosine_min': cosine_sim.min().item(),
        'cosine_max': cosine_sim.max().item(),
    }

    logger.info(f"梯度模长: {results['grad_norm_mean']:.6f} ± {results['grad_norm_std']:.6f}")
    logger.info(f"  范围: [{results['grad_norm_min']:.6f}, {results['grad_norm_max']:.6f}]")
    logger.info(f"与位置向量余弦相似度: {results['cosine_mean']:.6f} ± {results['cosine_std']:.6f}")
    logger.info(f"  范围: [{results['cosine_min']:.6f}, {results['cosine_max']:.6f}]")

    # 判断
    logger.info("\n" + "=" * 80)
    logger.info("验证结果")
    logger.info("=" * 80)

    # Eikonal: 模长应该接近1
    if abs(results['grad_norm_mean'] - 1.0) < 0.01:
        logger.info(f"✅ Eikonal属性满足: 模长 = {results['grad_norm_mean']:.6f} ≈ 1.0")
    else:
        logger.error(f"❌ Eikonal属性失败: 模长 = {results['grad_norm_mean']:.6f} ≠ 1.0")

    # 方向: 应该与位置向量一致（单位球，法线=径向）
    if abs(results['cosine_mean'] - 1.0) < 0.01:
        logger.info(f"✅ 方向正确: 余弦 = {results['cosine_mean']:.6f} ≈ 1.0")
    else:
        logger.error(f"❌ 方向错误: 余弦 = {results['cosine_mean']:.6f} ≠ 1.0")

    return results


def train_unit_sphere(
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 16384,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """
    训练SDF网络拟合单位球

    Args:
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        device: 设备
    """
    logger.info("=" * 80)
    logger.info("实验1：单位球体解析梯度测试")
    logger.info("=" * 80)
    logger.info(f"目标：SDF(x) = ||x|| - 1.0")
    logger.info(f"预期：∇SDF 模长=1，方向与x一致（cosine=1）")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 创建SDF网络（使用[-1.5, 1.5]作为bbox）
    sdf_net = SDFNetwork(
        num_levels=8,
        log2_hashmap_size=12,
        base_res=8,
        max_res=128,
        hidden_dim=32,
        num_layers=2,
        bbox_min=(-1.5, -1.5, -1.5),
        bbox_max=(1.5, 1.5, 1.5),
        cuda_backend="torch",
    ).to(device)

    logger.info(f"\nSDF网络创建成功: {sdf_net.get_num_params():,} 参数")

    optimizer = optim.Adam(sdf_net.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        sdf_net.train()

        epoch_loss = 0.0
        num_batches = 10

        for step in range(num_batches):
            # 生成训练数据
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
            }, Path(output_dir) / 'best.pt')

    logger.info("\n" + "=" * 80)
    logger.info("训练完成！")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info("=" * 80)

    return best_loss


def main():
    # 训练
    output_dir = "outputs/exp1_unit_sphere"

    train_unit_sphere(
        output_dir=output_dir,
        num_epochs=20,
        batch_size=16384,
        lr=1e-3,
        device="cuda",
    )

    # 验证
    logger.info("\n" + "=" * 80)
    logger.info("加载最佳模型进行验证")
    logger.info("=" * 80)

    checkpoint = torch.load(f"{output_dir}/best.pt")
    sdf_net = SDFNetwork(
        num_levels=8,
        log2_hashmap_size=12,
        base_res=8,
        max_res=128,
        hidden_dim=32,
        num_layers=2,
        bbox_min=(-1.5, -1.5, -1.5),
        bbox_max=(1.5, 1.5, 1.5),
        cuda_backend="torch",
    ).to("cuda")

    sdf_net.load_state_dict(checkpoint['model_state_dict'])
    sdf_net.eval()

    results = validate_gradient_properties(sdf_net, num_samples=100000, device="cuda")

    logger.info("\n" + "=" * 80)
    logger.info("实验1结论")
    logger.info("=" * 80)

    if abs(results['grad_norm_mean'] - 1.0) < 0.01 and abs(results['cosine_mean'] - 1.0) < 0.01:
        logger.info("✅ 实验1通过：自动微分和坐标归一化正确")
        logger.info("   可以排除假设一（坐标缩放断裂bug）")
    else:
        logger.error("❌ 实验1失败：自动微分或坐标归一化存在bug")
        logger.error("   确认假设一：坐标缩放断裂导致梯度计算错误")


if __name__ == "__main__":
    main()
