"""
可视化MA-IUVF训练曲线
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curves(csv_path: Path, output_path: Path):
    """绘制训练曲线"""

    # 读取训练数据
    df = pd.read_csv(csv_path)

    # 创建图表 - 4x2布局
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('MA-IUVF Training Curves - Dynamic Classification Loss Decay',
                 fontsize=16, fontweight='bold')

    # 1. 总损失
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. 分类准确率（重点）
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['cls_acc'] * 100, 'g-', linewidth=2, label='Classification Accuracy')
    ax.axhline(y=99, color='r', linestyle='--', label='99% Threshold (Decay Trigger)')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Decay Activation)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy (Dynamic Decay at 99%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([95, 100])

    # 3. 损失权重变化（重点）
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['cls_weight'], 'r-', linewidth=2, label='Cls Weight')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Decay: 1.0→0.01)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_title('Classification Weight (Exponential Decay)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 4. Metric Loss
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['metric'], 'purple', linewidth=2, label='Metric Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Metric Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. Anchor Loss
    ax = axes[2, 0]
    ax.plot(df['epoch'], df['anchor'], 'orange', linewidth=2, label='Anchor Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Anchor Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 6. Classification Loss
    ax = axes[2, 1]
    ax.plot(df['epoch'], df['cls'], 'brown', linewidth=2, label='Classification Loss')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Weight Decay)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Classification Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 7. CoM Loss
    ax = axes[3, 0]
    ax.plot(df['epoch'], df['com'], 'pink', linewidth=2, label='CoM Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Center of Mass Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 8. 权重综合视图
    ax = axes[3, 1]
    ax.plot(df['epoch'], df['metric_weight'], 'purple', linewidth=2, label='Metric Weight')
    ax.plot(df['epoch'], df['anchor_weight'], 'orange', linewidth=2, label='Anchor Weight')
    ax.plot(df['epoch'], df['cls_weight'], 'r-', linewidth=2, label='Cls Weight')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Decay)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_title('All Loss Weights')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {output_path}")

    # 保存关键统计数据
    stats = {
        '总Epoch数': len(df),
        '最佳损失': df['loss'].min(),
        '最终损失': df['loss'].iloc[-1],
        '最佳分类准确率': f"{df['cls_acc'].max() * 100:.2f}%",
        '最终分类准确率': f"{df['cls_acc'].iloc[-1] * 100:.2f}%",
        '衰减激活Epoch': 21,
        '衰减前分类权重': 1.0,
        '衰减后分类权重': 0.01,
    }

    print("\n📊 训练统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="可视化MA-IUVF训练曲线")
    parser.add_argument("--csv", required=True, help="训练损失CSV路径")
    parser.add_argument("--output", default="training_curves.png", help="输出图像路径")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if not csv_path.exists():
        print(f"❌ CSV文件不存在: {csv_path}")
        return

    plot_training_curves(csv_path, output_path)

if __name__ == "__main__":
    main()
