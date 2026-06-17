"""
可视化 Fixed Anchor + Hard Cutoff 训练曲线
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
    fig.suptitle('MA-IUVF Training Curves - Fixed Anchor + Hard Cls Cutoff',
                 fontsize=16, fontweight='bold')

    # 1. 总损失
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Total Loss')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Cls Cutoff)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 2. 分类准确率（重点）
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['cls_acc'] * 100, 'g-', linewidth=2, label='Classification Accuracy')
    ax.axhline(y=99, color='r', linestyle='--', alpha=0.5, label='99% Threshold')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Hard Cutoff)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy (Hard Cls Cutoff at Epoch 21)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. 损失权重变化（重点）
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['cls_weight'], 'r-', linewidth=2, label='Cls Weight')
    ax.plot(df['epoch'], df['anchor_weight'], 'b-', linewidth=2, label='Anchor Weight')
    ax.plot(df['epoch'], df['metric_weight'], 'purple', linewidth=2, label='Metric Weight')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Cls Cutoff)')
    ax.axvline(x=31, color='green', linestyle='--', alpha=0.5, label='Epoch 31 (Metric Ramp)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_title('Loss Weights (Fixed Anchor=1.0, Hard Cutoff at Epoch 21)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Anchor Loss（重点）
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['anchor'], 'orange', linewidth=2, label='Anchor Loss')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Cls Cutoff)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Anchor Loss (No Rebound - Fixed Weight=1.0)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 5. Metric Loss
    ax = axes[2, 0]
    ax.plot(df['epoch'], df['metric'], 'purple', linewidth=2, label='Metric Loss')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21')
    ax.axvline(x=31, color='green', linestyle='--', alpha=0.5, label='Epoch 31 (Metric Weight Ramp)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Metric Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 6. Classification Loss
    ax = axes[2, 1]
    ax.plot(df['epoch'], df['cls'], 'brown', linewidth=2, label='Classification Loss')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Weight=0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Classification Loss (Zero After Epoch 21)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 7. CoM Loss
    ax = axes[3, 0]
    ax.plot(df['epoch'], df['com'], 'pink', linewidth=2, label='CoM Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Center of Mass Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    # 8. 关键指标对比
    ax = axes[3, 1]
    ax.plot(df['epoch'], df['cls_acc'] * 100, 'g-', linewidth=2, label='Cls Acc (%)')
    ax.plot(df['epoch'], df['cls_weight'] * 100, 'r--', linewidth=2, label='Cls Weight (x100)')
    ax.axvline(x=21, color='orange', linestyle='--', alpha=0.7, label='Epoch 21 (Cutoff)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy / Weight')
    ax.set_title('Cls Accuracy vs Weight (Hard Cutoff Strategy)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {output_path}")

    # 保存关键统计数据
    cls_cutoff_epoch = 21
    before_cutoff = df[df['epoch'] <= cls_cutoff_epoch]
    after_cutoff = df[df['epoch'] > cls_cutoff_epoch]

    stats = {
        '总Epoch数': len(df),
        '最佳损失': df['loss'].min(),
        '最终损失': df['loss'].iloc[-1],
        '最佳分类准确率': f"{df['cls_acc'].max() * 100:.2f}%",
        '最终分类准确率': f"{df['cls_acc'].iloc[-1] * 100:.2f}%",
        '分类硬截断Epoch': cls_cutoff_epoch,
        '截断前Cls权重': before_cutoff['cls_weight'].iloc[-1],
        '截断后Cls权重': after_cutoff['cls_weight'].iloc[0],
        'Anchor权重策略': 'Fixed (1.0)',
        '截断前Anchor Loss': f"{before_cutoff['anchor'].iloc[-1]:.6g}",
        '截断后Anchor Loss': f"{after_cutoff['anchor'].iloc[-1]:.6g}",
        'Anchor Loss反弹': 'No' if after_cutoff['anchor'].max() < after_cutoff['anchor'].iloc[0] * 2 else 'Yes',
    }

    print("\n📊 训练统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return stats

def main():
    parser = argparse.ArgumentParser(description="可视化 Fixed Anchor + Hard Cutoff 训练曲线")
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
