#!/usr/bin/env python3
"""
训练脚本：纹理采样场

独立于 G/D/R 主训练流程的新实验方向。

训练目标：
- 学习一个查询算子 P_theta(x, rho) -> {u_i, w_i, sigma_i}
- 输入：高模邻域内的查询点 x 和尺度 rho
- 输出：纹理空间的采样分布
- 最终颜色来自原始纹理的采样

使用方法:
    python scripts/train_texture_sampler_field.py \\
        --mesh data/models/stanford-bunny.obj \\
        --output-dir outputs/texture_sampler_exp \\
        --epochs 100 \\
        --num-mixtures 8 \\
        --batch-size 1024
"""

import argparse
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.texture_sampler_field import TextureSamplerField, TextureSamplerFieldOutput, sample_texture
from src.data.tubular_texture_dataset import TubularTextureDataset, create_tubular_texture_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TextureSamplerFieldTrainer:
    """纹理采样场训练器"""

    def __init__(
        self,
        model: TextureSamplerField,
        device: torch.device,
        learning_rate: float = 1e-3,
        uv_distance_weight: float = 0.01,
        entropy_weight: float = 0.001,
    ):
        """
        Args:
            model: TextureSamplerField 模型
            device: 训练设备
            learning_rate: 学习率
            uv_distance_weight: UV 距离损失权重
            entropy_weight: 熵正则化权重
        """
        self.model = model.to(device)
        self.device = device
        self.uv_distance_weight = uv_distance_weight
        self.entropy_weight = entropy_weight

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 损失函数
        self.mse_loss = nn.MSELoss()

    def compute_loss(
        self,
        batch: dict,
        model_output: TextureSamplerFieldOutput,
        texture: torch.Tensor
    ) -> dict:
        """
        计算损失

        Args:
            batch: 数据 batch
            model_output: 模型输出
            texture: 纹理张量 (3, H, W)

        Returns:
            损失字典
        """
        # 提取数据
        positions = batch['position']  # (B, 3)
        target_uv = batch['target_uv']  # (B, 2)
        target_color = batch['target_color']  # (B, 3)
        scale = batch['scale']  # (B, 1)

        B = positions.shape[0]

        # 扩展维度
        target_uv_expanded = target_uv.unsqueeze(1).expand(B, model_output.uvs.shape[1], 2)  # (B, K, 2)

        # 1. 主要损失：颜色 MSE
        # flip_v=True 与 UVTextureSampler 的 V 方向一致
        pred_color = sample_texture(texture, model_output.uvs, model_output.weights, flip_v=True)
        color_loss = self.mse_loss(pred_color, target_color)

        # 2. UV 距离损失（让某个 mixture 靠近 target_uv）
        uv_distances = torch.norm(
            model_output.uvs - target_uv_expanded,
            dim=-1
        )  # (B, K)
        min_uv_distance = uv_distances.min(dim=-1)[0]  # (B,)
        uv_loss = min_uv_distance.mean()  # 标量

        # 3. 熵正则化（避免权重完全塌缩到单个 mixture）
        # 计算熵，使用负熵损失（最小化负熵 = 最大化熵 = 鼓励平坦分布）
        entropy = -(model_output.weights * torch.log(model_output.weights + 1e-8)).sum(dim=-1)  # (B,)
        entropy_loss = -entropy.mean()  # 负熵，最小化总损失会最大化熵

        # 4. Sigma 正则化（实验性设计）
        # 注意：此监督策略是实验性的，需要进一步验证
        # - scale 是 3D 查询点邻域的扰动尺度（几何空间）
        # - sigma 理论上可代表纹理采样的 mipmap/filter 尺度
        # - 当前简单让 sigma 接近 scale，但两者是否应直接对齐需实验验证
        # 使用 log-space 监督避免数值范围差异问题
        sigmas_squeezed = model_output.sigmas.squeeze(-1)  # (B, K)
        sigma_mean = sigmas_squeezed.mean(dim=-1, keepdim=True)  # (B, 1)

        # 确保 scale 也是 (B, 1)
        if scale.dim() == 1:
            scale = scale.unsqueeze(-1)  # (B,) -> (B, 1)

        # 在 log-space 监督，避免大数值问题
        eps = 1e-6
        log_sigma = torch.log(sigma_mean + eps)
        log_scale = torch.log(scale + eps)
        sigma_loss = F.mse_loss(log_sigma, log_scale)

        # 总损失（添加 sigma 正则化，权重设为较小值）
        sigma_weight = 0.01
        total_loss = color_loss + self.uv_distance_weight * uv_loss + self.entropy_weight * entropy_loss + sigma_weight * sigma_loss

        return {
            'total': total_loss,
            'color': color_loss,
            'uv_distance': uv_loss,
            'entropy': entropy_loss,
            'sigma': sigma_loss,
        }

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        texture: torch.Tensor,
        epoch: int
    ) -> dict:
        """训练一个 epoch"""
        self.model.train()

        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            model_output = self.model(batch['position'], scale=batch['scale'])

            # 计算损失
            losses = self.compute_loss(batch, model_output, texture)
            total_loss = losses['total']

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'color': f"{losses['color'].item():.4f}",
                'uv': f"{losses['uv_distance'].item():.4f}",
                'ent': f"{losses['entropy'].item():.4f}",
                'sigma': f"{losses['sigma'].item():.4f}",
            })

            epoch_losses.append({
                'total': total_loss.item(),
                'color': losses['color'].item(),
                'uv_distance': losses['uv_distance'].item(),
                'entropy': losses['entropy'].item(),
                'sigma': losses['sigma'].item(),
            })

        # 计算 epoch 平均损失
        avg_losses = {
            k: np.mean([loss[k] for loss in epoch_losses])
            for k in epoch_losses[0].keys()
        }

        return avg_losses

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        metadata: dict
    ):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **metadata
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="训练纹理采样场",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础训练
  python scripts/train_texture_sampler_field.py \\
      --mesh data/models/stanford-bunny.obj \\
      --output-dir outputs/texture_field_exp \\
      --epochs 50 \\
      --num-mixtures 8

  # 使用程序化纹理
  python scripts/train_texture_sampler_field.py \\
      --mesh data/models/stanford-bunny.obj \\
      --output-dir outputs/texture_field_exp \\
      --epochs 100 \\
      --num-mixtures 16 \\
      --lr 0.001
        """
    )

    # 必需参数
    parser.add_argument("--mesh", required=True, help="高模 mesh 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--texture", help="纹理路径（可选，默认使用程序化纹理）")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1024, help="批量大小")
    parser.add_argument("--num-samples", type=int, default=100000, help="训练样本数")
    parser.add_argument("--num-mixtures", type=int, default=8, help="混合成分数量 K")

    # 模型参数
    parser.add_argument("--hidden-dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=4, help="MLP 层数")
    parser.add_argument("--positional-enc-freqs", type=int, default=6, help="位置编码频率数量")

    # 优化参数
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--uv-distance-weight", type=float, default=0.01, help="UV 距离损失权重")
    parser.add_argument("--entropy-weight", type=float, default=0.001, help="熵正则化权重")

    # 其他参数
    parser.add_argument("--device", default="cuda", help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log-interval", type=int, default=10, help="日志间隔（epoch）")

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据
    logger.info(f"加载数据: {args.mesh}")
    dataset = create_tubular_texture_dataset(
        mesh_path=args.mesh,
        texture_path=args.texture,
        num_samples=args.num_samples,
        min_scale=0.001,
        max_scale=0.05,
        seed=args.seed,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    logger.info(f"Dataset 创建完成: {len(dataset)} 样本")

    # 创建模型
    logger.info(f"创建模型: K={args.num_mixtures}")
    model = TextureSamplerField(
        num_mixtures=args.num_mixtures,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        positional_encoding_freqs=args.positional_enc_freqs,
        use_scale_input=True,
        dropout_rate=0.0,
    )

    # 创建纹理张量（用于 sample_texture）
    # TextureData.image 是 (H, W, 3)，需要转换为 (1, 3, H, W)
    texture_tensor = torch.from_numpy(dataset.texture_data.image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    texture_tensor = texture_tensor.to(device)  # 移动到训练设备
    logger.info(f"纹理张量形状: {texture_tensor.shape}, 设备: {device}")

    # 创建训练器
    trainer = TextureSamplerFieldTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        uv_distance_weight=args.uv_distance_weight,
        entropy_weight=args.entropy_weight,
    )

    # 训练
    logger.info(f"开始训练: {args.epochs} epochs")
    best_loss = float('inf')

    # 准备 metadata（用于所有 checkpoint）
    bbox_info = dataset.get_bbox_info()
    metadata = {
        'mesh_path': args.mesh,
        'texture_path': args.texture if args.texture else 'procedural',
        'num_mixtures': args.num_mixtures,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'positional_encoding_freqs': args.positional_enc_freqs,
        'use_scale_input': True,  # 模型配置
        'min_scale': 0.001,
        'max_scale': 0.05,
        'scale_sampling': dataset.scale_sampling,  # 数据集配置
        'bbox_min': bbox_info['bbox_min'],
        'bbox_max': bbox_info['bbox_max'],
        'learning_rate': args.lr,
        'uv_distance_weight': args.uv_distance_weight,
        'entropy_weight': args.entropy_weight,
        'sigma_weight': 0.01,  # 训练配置
    }

    for epoch in range(args.epochs):
        avg_losses = trainer.train_epoch(dataloader, texture_tensor, epoch)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {avg_losses['total']:.4f} "
            f"(color: {avg_losses['color']:.4f}, "
            f"uv: {avg_losses['uv_distance']:.4f}, "
            f"ent: {avg_losses['entropy']:.4f}, "
            f"sigma: {avg_losses['sigma']:.4f})"
        )

        # 定期保存 checkpoint
        if (epoch + 1) % args.log_interval == 0:
            save_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(save_path, epoch, metadata)

        # 保存最佳模型（包含完整 metadata）
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            best_save_path = output_dir / "best.pt"
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': best_loss,
                'best_loss': best_loss,  # 与 final.pt 统一 schema
                **metadata,
            }
            torch.save(best_checkpoint, best_save_path)
            logger.info(f"保存最佳模型: {best_save_path}")

    # 保存最终模型（包含完整 metadata）
    final_save_path = output_dir / "final.pt"
    final_loss = avg_losses['total']  # 使用最终 epoch 的 loss
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'loss': final_loss,  # 最终 epoch 的 loss
        'best_loss': best_loss,  # 额外保存最佳损失
        **metadata,
    }
    torch.save(final_checkpoint, final_save_path)
    logger.info(f"训练完成! 最终模型保存到: {final_save_path}")
    logger.info(f"最终损失: {final_loss:.4f}, 最佳损失: {best_loss:.4f}")


if __name__ == "__main__":
    main()
