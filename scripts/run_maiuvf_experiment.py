"""
MA-IUVF 端到端实验脚本

完整流程：bake -> train -> infer/render -> save metrics/images

Supports YAML configuration files with CLI override capability.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import shutil
import numpy as np

# NEW: Import configuration system
from src.maiuvf_config_loader import MAIUVFConfigLoader
from src.maiuvf_config_utils import compute_config_diff, save_differences
from src.maiuvf_config import MAIUVFConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """运行命令并记录"""
    logger.info(f"运行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time

        logger.info(f"✓ {description} 完成 (耗时: {elapsed:.1f}s)")

        if result.stdout:
            logger.debug(f"stdout: {result.stdout[-500:]}")  # 最后 500 字符

        return True, elapsed

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        logger.error(f"✗ {description} 失败 (耗时: {elapsed:.1f}s)")
        logger.error(f"stderr: {e.stderr[-1000:]}")  # 最后 1000 字符

        return False, elapsed


def bake_data(
    input_mesh: str,
    output_dir: Path,
    num_samples: int,
    chart_mode: str,
    texture_path: str = None,
    use_dynamic_sampling: bool = False,
    virtual_epoch_size: int = 300000,
    sigma_ratio: float = 0.01,
) -> bool:
    """步骤 1: 烘载数据"""
    logger.info("\n" + "="*60)
    logger.info("步骤 1/5: 数据准备")
    logger.info("="*60)

    if use_dynamic_sampling:
        # 动态采样模式：直接使用已有的 mesh_constants 文件
        logger.info("使用动态采样模式，跳过烘焙步骤")
        logger.info(f"需要预先准备 mesh_constants 文件: data/models/bunny_mesh_constants.pt")

        # 检查 mesh_constants 文件是否存在
        mesh_constants_path = Path("data/models/bunny_mesh_constants.pt")
        if not mesh_constants_path.exists():
            logger.error(f"✗ mesh_constants 文件不存在: {mesh_constants_path}")
            logger.error("请先运行烘焙脚本生成 mesh_constants")
            return False

        logger.info(f"✓ 找到 mesh_constants 文件: {mesh_constants_path}")

        # 复制到输出目录（用于实验归档）
        output_path = output_dir / "mesh_constants.pt"
        import shutil
        shutil.copy(mesh_constants_path, output_path)
        logger.info(f"✓ 复制 mesh_constants 到: {output_path}")

        return True
    else:
        # 静态采样模式：烘焙点云数据
        output_path = output_dir / "baked.pt"

        cmd = [
            "python", "scripts/bake_metric_aligned_iuv_data.py",
            "--mesh", input_mesh,
            "--output", str(output_path),
            "--num-samples", str(num_samples),
            "--chart-mode", chart_mode,
            "--seed", "42",
        ]

        if texture_path:
            cmd.extend(["--texture", texture_path])

        success, elapsed = run_command(cmd, "数据烘焙")

        if success:
            # 验证输出
            if output_path.exists():
                logger.info(f"✓ 烘焙数据已保存: {output_path}")
                return True
            else:
                logger.error(f"✗ 烘焙数据未生成: {output_path}")
                return False

        return False


def train_model(
    baked_data: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    metric_weight: float,
    anchor_weight: float,
    cls_weight: float,
    encoder_type: str,
    activation: str,
    com_weight: float = 0.0,
    hash_lr: float = None,
    hash_num_levels: int = 16,
    hash_features_per_level: int = 2,
    hash_log2_size: int = 19,
    hash_base_res: int = 16,
    hash_max_res: int = 2048,
    hash_cuda_backend: str = "auto",
    hash_weight_decay: float = 1e-6,
    mlp_weight_decay: float = 0.0,
    loss_schedule: str = "two_stage",
    phase_a_epochs: int = 30,
    target_metric_weight: float = 1.0,
    target_anchor_weight: float = 0.01,
    target_cls_weight: float = 0.1,
    schedule_ramp: str = "cosine",
    # NEW: Hard Classification Cutoff Parameters
    cls_cutoff_epoch: int = 20,
    cls_cutoff_value: float = 0.0,
    keep_anchor_constant: bool = True,
    # NEW: Unified Local Loss Parameters
    unified_weight: float = 0.0,
    unified_num_neighbors: int = 4,
    unified_epsilon: float = 0.01,
    # ========================
    # DEPRECATED: Dynamic Classification Decay
    # dynamic_cls_decay: float = 0.01,
    # cls_decay_epoch_threshold: int = 20,
    # cls_acc_threshold: float = 0.99,
    # ========================
    use_dynamic_sampling: bool = False,
    virtual_epoch_size: int = 300000,
    sigma_ratio: float = 0.01,
) -> bool:
    """步骤 2: 训练模型"""
    logger.info("\n" + "="*60)
    logger.info("步骤 2/5: 训练模型")
    logger.info("="*60)

    cmd = [
        "python", "scripts/train_metric_aligned_iuv_field.py",
        "--data", str(baked_data),
        "--output-dir", str(output_dir / "train"),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--device", device,
        "--metric-weight", str(metric_weight),
        "--anchor-weight", str(anchor_weight),
        "--com-weight", str(com_weight),
        "--cls-weight", str(cls_weight),
        "--encoder-type", encoder_type,
        "--activation", activation,
        "--hash-num-levels", str(hash_num_levels),
        "--hash-features-per-level", str(hash_features_per_level),
        "--hash-log2-size", str(hash_log2_size),
        "--hash-base-res", str(hash_base_res),
        "--hash-max-res", str(hash_max_res),
        "--hash-cuda-backend", hash_cuda_backend,
        "--hash-weight-decay", str(hash_weight_decay),
        "--mlp-weight-decay", str(mlp_weight_decay),
        "--loss-schedule", loss_schedule,
        "--phase-a-epochs", str(phase_a_epochs),
        "--target-metric-weight", str(target_metric_weight),
        "--target-anchor-weight", str(target_anchor_weight),
        "--target-cls-weight", str(target_cls_weight),
        "--schedule-ramp", schedule_ramp,
        # NEW: Hard Classification Cutoff & Improved Strategy
        "--cls-cutoff-epoch", str(cls_cutoff_epoch),
        "--cls-cutoff-value", str(cls_cutoff_value),
        "--keep-anchor-constant" if keep_anchor_constant else "--no-keep-anchor-constant",
        # NEW: Unified Local Loss Parameters
        "--unified-weight", str(unified_weight),
        "--unified-num-neighbors", str(unified_num_neighbors),
        "--unified-epsilon", str(unified_epsilon),
        # ========================
        # DEPRECATED: Dynamic Classification Decay (replaced by hard cutoff)
        # "--dynamic-cls-decay", str(dynamic_cls_decay),
        # "--cls-decay-epoch-threshold", str(cls_decay_epoch_threshold),
        # "--cls-acc-threshold", str(cls_acc_threshold),
        # ========================
    ]

    # 添加可选参数
    if hash_lr is not None:
        cmd.extend(["--hash-lr", str(hash_lr)])

    if use_dynamic_sampling:
        cmd.extend([
            "--virtual-epoch-size", str(virtual_epoch_size),
            "--sigma-ratio", str(sigma_ratio),
        ])

    success, elapsed = run_command(cmd, "模型训练")

    if success:
        # 验证输出
        best_checkpoint = output_dir / "train" / "best.pt"
        final_checkpoint = output_dir / "train" / "final.pt"
        loss_csv = output_dir / "train" / "train_loss.csv"

        files_exist = all([
            best_checkpoint.exists(),
            final_checkpoint.exists(),
            loss_csv.exists(),
        ])

        if files_exist:
            logger.info(f"✓ 训练完成:")
            logger.info(f"  - Best checkpoint: {best_checkpoint}")
            logger.info(f"  - Final checkpoint: {final_checkpoint}")
            logger.info(f"  - Training loss: {loss_csv}")

            # 分析训练曲线
            try:
                import pandas as pd
                loss_data = pd.read_csv(loss_csv)
                final_loss = loss_data['loss'].iloc[-1]
                best_loss = loss_data['loss'].min()

                logger.info(f"  - Final loss: {final_loss:.6f}")
                logger.info(f"  - Best loss: {best_loss:.6f}")

                return True
            except Exception as e:
                logger.warning(f"无法分析训练曲线: {e}")
                return True
        else:
            logger.error(f"✗ 训练输出不完整")
            return False

    return False


def infer_and_render(
    checkpoint: Path,
    input_mesh: str,
    texture_path: str,
    output_dir: Path,
    target_faces: int,
    device: str,
    render_mode: str,
    render_resolution: int,
) -> bool:
    """步骤 3: 推理和渲染"""
    logger.info("\n" + "="*60)
    logger.info("步骤 3/5: 推理和渲染")
    logger.info("="*60)

    render_dir = output_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    if render_mode == "obj":
        # 导出 textu red OBJ
        cmd = [
            "python", "scripts/infer_metric_aligned_iuv.py",
            "--checkpoint", str(checkpoint),
            "--input-mesh", input_mesh,
            "--texture", texture_path,
            "--output-dir", str(render_dir),
            "--target-faces", str(target_faces),
            "--device", device,
        ]

        success, elapsed = run_command(cmd, "OBJ 导出")

        if success:
            logger.info(f"✓ OBJ 导出完成")

    # 离屏渲染验证
    render_output = render_dir / f"render_{render_mode}.png"

    cmd = [
        "python", "scripts/render_metric_aligned_iuv_test.py",
        "--checkpoint", str(checkpoint),
        "--input-mesh", input_mesh,
        "--texture", texture_path,
        "--output-dir", str(render_dir),
        "--render-mode", render_mode,
        "--resolution", str(render_resolution),
        "--device", device,
        "--no-viewer",
    ]

    success, elapsed = run_command(cmd, f"离屏渲染 ({render_mode})")

    if success and render_output.exists():
        logger.info(f"✓ 渲染完成: {render_output}")
        return True
    else:
        logger.error(f"✗ 渲染失败")
        return False


def compute_metrics(
    output_dir: Path,
) -> dict:
    """步骤 4: 计算指标"""
    logger.info("\n" + "="*60)
    logger.info("步骤 4/5: 计算指标")
    logger.info("="*60)

    metrics = {}

    # 训练指标
    loss_csv = output_dir / "train" / "train_loss.csv"
    if loss_csv.exists():
        try:
            import pandas as pd
            loss_data = pd.read_csv(loss_csv)

            training_metrics = {
                'final_loss': float(loss_data['loss'].iloc[-1]),
                'best_loss': float(loss_data['loss'].min()),
                'initial_loss': float(loss_data['loss'].iloc[0]),
                'epochs': len(loss_data),
            }

            # 尝试获取分类准确率
            if 'cls_acc' in loss_data.columns:
                training_metrics['final_cls_acc'] = float(loss_data['cls_acc'].iloc[-1])
                training_metrics['best_cls_acc'] = float(loss_data['cls_acc'].max())
                logger.info(f"✓ 训练指标 (含分类准确率):")
                logger.info(f"  - Best classification accuracy: {training_metrics['best_cls_acc']:.2%}")
                logger.info(f"  - Final classification accuracy: {training_metrics['final_cls_acc']:.2%}")

            metrics['training'] = training_metrics

            logger.info(f"✓ 训练指标:")
            logger.info(f"  - Epochs: {metrics['training']['epochs']}")
            logger.info(f"  - Initial loss: {metrics['training']['initial_loss']:.6f}")
            logger.info(f"  - Best loss: {metrics['training']['best_loss']:.6f}")
            logger.info(f"  - Final loss: {metrics['training']['final_loss']:.6f}")

        except Exception as e:
            logger.warning(f"无法读取训练指标: {e}")

    # 渲染统计
    render_info_path = output_dir / "render" / "render_info.json"
    if render_info_path.exists():
        try:
            with open(render_info_path, 'r') as f:
                render_info = json.load(f)

            rendering_metrics = {
                'pixel_coverage': render_info.get('coverage', 0.0),
                'valid_pixels': render_info.get('valid_pixels', 0),
                'total_pixels': render_info.get('total_pixels', 0),
                'pred_num_charts': render_info.get('pred_num_charts', 0),
                'pred_chart_distribution': render_info.get('pred_chart_distribution', {}),
            }

            # 添加新的统计信息
            if 'chart_accuracy' in render_info and render_info['chart_accuracy'] is not None:
                rendering_metrics['chart_accuracy'] = render_info['chart_accuracy']
                logger.info(f"✓ 渲染统计 (含分类准确率):")
                logger.info(f"  - 分类准确率: {rendering_metrics['chart_accuracy']:.2%}")

            if 'visible_gt_num_charts' in render_info and render_info['visible_gt_num_charts'] is not None:
                rendering_metrics['visible_gt_num_charts'] = render_info['visible_gt_num_charts']
                rendering_metrics['gt_chart_distribution'] = render_info.get('gt_chart_distribution', {})
                logger.info(f"  - 可见 GT charts: {rendering_metrics['visible_gt_num_charts']}")

            # 向后兼容：如果没有新的字段，使用旧的
            if 'pred_num_charts' not in rendering_metrics or rendering_metrics['pred_num_charts'] == 0:
                rendering_metrics['num_charts_rendered'] = len(render_info.get('chart_distribution', {}))

            metrics['rendering'] = rendering_metrics

            logger.info(f"✓ 渲染指标:")
            logger.info(f"  - 像素覆盖率: {metrics['rendering']['pixel_coverage']:.2%}")

            # 根据可用字段输出不同的信息
            if 'chart_accuracy' in rendering_metrics and rendering_metrics['chart_accuracy'] is not None:
                logger.info(f"  - 分类准确率: {rendering_metrics['chart_accuracy']:.2%}")
                logger.info(f"  - 可见 GT charts: {rendering_metrics['visible_gt_num_charts']}")
                logger.info(f"  - 预测 charts 数: {rendering_metrics['pred_num_charts']}")
            elif 'num_charts_rendered' in rendering_metrics:
                logger.info(f"  - 预测 charts 数: {rendering_metrics['num_charts_rendered']} (旧格式)")

        except Exception as e:
            logger.warning(f"无法读取渲染指标: {e}")

    # 烘焙数据统计
    baked_data_path = output_dir / "baked.pt"
    if baked_data_path.exists():
        try:
            import torch
            baked_data = torch.load(baked_data_path, weights_only=False)
            metadata = baked_data.get('metadata', {})

            metrics['baking'] = {
                'num_samples': metadata.get('num_samples', 0),
                'num_charts': metadata.get('num_charts', 0),
                'chart_mode': metadata.get('chart_mode', 'unknown'),
            }

            if 'chart_stats' in metadata:
                stats = metadata['chart_stats']
                metrics['baking']['chart_sizes'] = stats.get('chart_sizes', [])
                metrics['baking']['num_uv_seams'] = stats.get('num_uv_seams', 0)

            logger.info(f"✓ 烘焙指标:")
            logger.info(f"  - 样本数: {metrics['baking']['num_samples']}")
            logger.info(f"  - Charts: {metrics['baking']['num_charts']}")
            logger.info(f"  - Chart 模式: {metrics['baking']['chart_mode']}")

            if 'num_uv_seams' in metrics['baking']:
                logger.info(f"  - UV seams: {metrics['baking']['num_uv_seams']}")

        except Exception as e:
            logger.warning(f"无法读取烘焙指标: {e}")

    return metrics


def save_results(
    output_dir: Path,
    metrics: dict,
    experiment_config: dict,
):
    """步骤 5: 保存结果"""
    logger.info("\n" + "="*60)
    logger.info("步骤 5/5: 保存结果")
    logger.info("="*60)

    # 保存指标（处理 numpy 类型）
    def convert_to_serializable(obj):
        """转换 numpy 类型为 Python 原生类型"""
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        else:
            return obj

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)

    logger.info(f"✓ 保存指标: {metrics_path}")

    # 保存实验配置
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)

    logger.info(f"✓ 保存配置: {config_path}")

    # 创建结果摘要
    summary_path = output_dir / "EXPERIMENT_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write("# MA-IUVF 实验摘要\n\n")

        f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验配置\n\n")
        for key, value in experiment_config.items():
            f.write(f"- **{key}**: {value}\n")

        f.write("\n## 主要结果\n\n")

        if 'training' in metrics:
            f.write("### 训练\n\n")
            f.write(f"- Epochs: {metrics['training']['epochs']}\n")
            f.write(f"- Initial loss: {metrics['training']['initial_loss']:.6f}\n")
            f.write(f"- Best loss: {metrics['training']['best_loss']:.6f}\n")
            f.write(f"- Final loss: {metrics['training']['final_loss']:.6f}\n")
            if 'best_cls_acc' in metrics['training']:
                f.write(f"- Best classification accuracy: {metrics['training']['best_cls_acc']:.2%}\n")
            if 'final_cls_acc' in metrics['training']:
                f.write(f"- Final classification accuracy: {metrics['training']['final_cls_acc']:.2%}\n")
            f.write("\n")

        if 'baking' in metrics:
            f.write("### 烘焙数据验证\n\n")
            f.write(f"- 样本数: {metrics['baking']['num_samples']}\n")
            f.write(f"- Charts: {metrics['baking']['num_charts']}")
            if metrics['baking']['num_charts'] == 8:
                f.write(" ✅ (目标: 8 charts)\n")
            else:
                f.write(f" ⚠️ (目标: 8 charts, 当前: {metrics['baking']['num_charts']})\n")
            f.write(f"- Chart 模式: {metrics['baking']['chart_mode']}\n")
            f.write(f"- UV seams: {metrics['baking']['num_uv_seams']}\n\n")

        if 'rendering' in metrics:
            f.write("### 渲染质量评估\n\n")
            f.write(f"- 像素覆盖率: {metrics['rendering']['pixel_coverage']:.2%}\n")
            if metrics['rendering']['pixel_coverage'] >= 0.9:
                f.write("  ✅ 覆盖率 >= 90%\n")
            elif metrics['rendering']['pixel_coverage'] >= 0.6:
                f.write("  ⚠️ 覆盖率 60-90% (单视角几何限制)\n")
            else:
                f.write(f"  ❌ 覆盖率 < 60% (可能需要更多训练)\n")

            if 'chart_accuracy' in metrics['rendering'] and metrics['rendering']['chart_accuracy'] is not None:
                f.write(f"- 分类准确率: {metrics['rendering']['chart_accuracy']:.2%}\n")
                if metrics['rendering']['chart_accuracy'] >= 0.8:
                    f.write("  ✅ 分类准确率 >= 80%\n")
                elif metrics['rendering']['chart_accuracy'] >= 0.6:
                    f.write("  ⚠️ 分类准确率 60-80%\n")
                else:
                    f.write("  ❌ 分类准确率 < 60%\n")

            if 'visible_gt_num_charts' in metrics['rendering'] and metrics['rendering']['visible_gt_num_charts'] is not None:
                f.write(f"- 可见 GT charts: {metrics['rendering']['visible_gt_num_charts']}\n")
                f.write(f"- 预测 charts 数: {metrics['rendering']['pred_num_charts']}\n")
            f.write("\n")

        f.write("## 输出文件\n\n")
        f.write("```\n")
        f.write("baked.pt              # 烘焙的训练数据\n")
        f.write("train/\n")
        f.write("  ├─ best.pt         # 最佳模型\n")
        f.write("  ├─ final.pt        # 最终模型\n")
        f.write("  └─ train_loss.csv  # 训练曲线\n")
        f.write("render/\n")
        f.write("  ├─ render_obj.png  # OBJ 渲染结果\n")
        f.write("  ├─ render_cpu.png  # CPU 离屏渲染结果\n")
        f.write("  └─ render_info.json # 渲染统计\n")
        f.write("metrics.json         # 所有指标\n")
        f.write("experiment_config.json # 实验配置\n")
        f.write("```\n")

    logger.info(f"✓ 保存摘要: {summary_path}")

    # 打印总结
    logger.info("\n" + "="*60)
    logger.info("实验完成总结")
    logger.info("="*60)

    if 'training' in metrics:
        train_summary = f"训练: {metrics['training']['epochs']} epochs, loss {metrics['training']['initial_loss']:.4f} -> {metrics['training']['final_loss']:.4f}"
        if 'best_cls_acc' in metrics['training']:
            train_summary += f", 分类准确率 {metrics['training']['best_cls_acc']:.2%}"
        logger.info(train_summary)

    if 'baking' in metrics:
        logger.info(f"烘焙数据: {metrics['baking']['num_charts']} charts")
        if metrics['baking']['num_charts'] == 8:
            logger.info(f"  ✅ 符合目标 (8 charts)")
        else:
            logger.info(f"  ⚠️ 目标是 8 charts")

    if 'rendering' in metrics:
        render_summary = f"渲染: {metrics['rendering']['pixel_coverage']:.1%} 像素覆盖率"
        if 'chart_accuracy' in metrics['rendering'] and metrics['rendering']['chart_accuracy'] is not None:
            render_summary += f", 分类准确率 {metrics['rendering']['chart_accuracy']:.2%}"
        logger.info(render_summary)

    logger.info(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="MA-IUVF 端到端实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整实验（立方体）
  python scripts/run_maiuvf_experiment.py \\
      --input-mesh test_data/uv_seam_cube.obj \\
      --texture test_data/test.png \\
      --output-dir outputs/cube_experiment \\
      --num-samples 10000 \\
      --epochs 50

  # 快速测试
  python scripts/run_maiuvf_experiment.py \\
      --input-mesh test_data/uv_seam_cube.obj \\
      --texture test_data/test.png \\
      --output-dir outputs/quick_test \\
      --num-samples 1000 \\
      --epochs 10 \\
      --quick-test
        """
    )

    # NEW: YAML configuration file argument
    parser.add_argument("--config", type=str, help="YAML 配置文件路径")

    # 输入数据（no longer required when using --config）
    parser.add_argument("--input-mesh", help="输入高模路径")
    parser.add_argument("--texture", help="纹理路径")
    parser.add_argument("--output-dir", help="输出目录")

    # 烘焙参数
    parser.add_argument("--num-samples", type=int, default=10000, help="烘焙样本数")
    parser.add_argument("--chart-mode", default="uv_islands", choices=["face_component", "uv_islands"], help="Chart 分配模式")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8192, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--metric-weight", type=float, default=0.01, help="metric loss权重")
    parser.add_argument("--anchor-weight", type=float, default=1.0, help="anchor loss权重")
    parser.add_argument("--cls-weight", type=float, default=1.0, help="分类loss权重")
    parser.add_argument("--com-weight", type=float, default=0.0, help="质心对齐loss权重")
    parser.add_argument("--encoder-type", default="bspline_hash", choices=["fourier", "bspline_hash"], help="空间编码器")
    parser.add_argument("--activation", default="silu", choices=["softplus", "silu", "relu"], help="MLP激活函数")

    # Hash Grid参数
    parser.add_argument("--hash-lr", type=float, default=None, help="Hash Grid学习率（默认同--lr）")
    parser.add_argument("--hash-num-levels", type=int, default=16, help="Hash Grid层级数")
    parser.add_argument("--hash-features-per-level", type=int, default=2, help="每层Hash特征维度")
    parser.add_argument("--hash-log2-size", type=int, default=19, help="每层Hash表大小log2")
    parser.add_argument("--hash-base-res", type=int, default=16, help="Hash Grid基础分辨率")
    parser.add_argument("--hash-max-res", type=int, default=2048, help="Hash Grid最高分辨率")
    parser.add_argument("--hash-cuda-backend", default="auto", choices=["auto", "torch", "cuda"], help="Hash Grid CUDA后端")
    parser.add_argument("--hash-weight-decay", type=float, default=1e-6, help="Hash Grid权重衰减")
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0, help="MLP/head权重衰减")

    # Loss调度参数
    parser.add_argument("--loss-schedule", default="two_stage", choices=["fixed", "two_stage"], help="损失调度策略")
    parser.add_argument("--phase-a-epochs", type=int, default=30, help="Phase A训练轮数")
    parser.add_argument("--target-metric-weight", type=float, default=1.0, help="目标metric loss权重")
    parser.add_argument("--target-anchor-weight", type=float, default=0.01, help="目标anchor loss权重")
    parser.add_argument("--target-cls-weight", type=float, default=0.1, help="目标分类loss权重")
    parser.add_argument("--schedule-ramp", default="cosine", choices=["cosine", "linear"], help="调度插值方式")

    # NEW: Hard Classification Cutoff & Improved Strategy Parameters
    parser.add_argument("--cls-cutoff-epoch", type=int, default=20, help="分类权重硬截断epoch（超过此epoch后cls_weight设为cls_cutoff_value）")
    parser.add_argument("--cls-cutoff-value", type=float, default=0.0, help="分类权重硬截断值（推荐0.0或1e-4）")
    parser.add_argument("--keep-anchor-constant", action="store_true", default=True, help="保持anchor权重恒定（禁用two_stage衰减）")
    parser.add_argument("--no-keep-anchor-constant", action="store_false", dest="keep_anchor_constant", help="允许anchor权重参与two_stage衰减")
    parser.add_argument("--unified-weight", type=float, default=0.0, help="统一局部损失权重（0.0=禁用）")
    parser.add_argument("--unified-num-neighbors", type=int, default=4, help="统一局部损失：邻域点数量")
    parser.add_argument("--unified-epsilon", type=float, default=0.01, help="统一局部损失：邻域扰动幅度")

    # DEPRECATED: Dynamic Classification Decay (replaced by hard cutoff)
    # parser.add_argument("--dynamic-cls-decay", type=float, default=0.01, help="分类权重指数衰减因子（当cls_acc > 0.99时）")
    # parser.add_argument("--cls-decay-epoch-threshold", type=int, default=20, help="开始检查分类准确率的epoch阈值")
    # parser.add_argument("--cls-acc-threshold", type=float, default=0.99, help="触发指数衰减的分类准确率阈值")

    # 动态采样参数
    parser.add_argument("--use-dynamic-sampling", action="store_true", help="使用GPU动态采样（需要mesh_constants数据）")
    parser.add_argument("--virtual-epoch-size", type=int, default=300000, help="动态采样：每epoch虚拟样本数")
    parser.add_argument("--sigma-ratio", type=float, default=0.01, help="动态采样：sigma/bbox_diagonal")

    # 推理渲染参数
    parser.add_argument("--target-faces", type=int, default=500, help="低模目标面数")
    parser.add_argument("--render-mode", default="cpu", choices=["obj", "cpu"], help="渲染模式")
    parser.add_argument("--render-resolution", type=int, default=512, help="渲染分辨率")

    # 其他
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--quick-test", action="store_true", help="快速测试模式（减少样本和轮数）")

    args = parser.parse_args()

    # === NEW: YAML Configuration System ===
    loader = MAIUVFConfigLoader()
    flat_config = None

    if args.config:
        # Load from YAML configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = loader.load_config(args.config)

        # Apply CLI overrides
        config = loader.apply_cli_overrides(args)

        # Get flat config for existing functions
        flat_config = loader.get_flat_config()

        # Update args with loaded config for compatibility
        for key, value in flat_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

        # Create output directory
        output_dir = Path(flat_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final configuration (YAML + CLI overrides)
        final_config_path = output_dir / "final_config.yaml"
        config.to_yaml(str(final_config_path))
        logger.info(f"✓ 保存最终配置: {final_config_path}")

        # Save CLI overrides if any
        base_config = MAIUVFConfig.from_yaml(args.config)
        differences = compute_config_diff(base_config, config)
        if any(differences.values()):
            diff_path = output_dir / "config_overrides.yaml"
            save_differences(differences, diff_path)
            logger.info(f"✓ 保存配置覆盖: {diff_path}")

        logger.info(f"✓ 配置加载完成，实验名称: {config.experiment.name}")

    else:
        # Traditional CLI mode (fully backward compatible)
        logger.info("使用传统CLI模式（未指定--config）")

        # Check required parameters
        if not all([args.input_mesh, args.texture, args.output_dir]):
            parser.error("当不使用 --config 时，必须指定 --input-mesh, --texture, 和 --output-dir")

        # Convert CLI args to flat config dict
        flat_config = {
            'input_mesh': args.input_mesh,
            'texture': args.texture,
            'output_dir': args.output_dir,
            'num_samples': args.num_samples,
            'chart_mode': args.chart_mode,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': args.device,
            'encoder_type': args.encoder_type,
            'activation': args.activation,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'positional_enc_freqs': args.positional_enc_freqs,
            'hash_lr': args.hash_lr,
            'hash_num_levels': args.hash_num_levels,
            'hash_features_per_level': args.hash_features_per_level,
            'hash_log2_size': args.hash_log2_size,
            'hash_base_res': args.hash_base_res,
            'hash_max_res': args.hash_max_res,
            'hash_cuda_backend': args.hash_cuda_backend,
            'hash_weight_decay': args.hash_weight_decay,
            'mlp_weight_decay': args.mlp_weight_decay,
            'metric_weight': args.metric_weight,
            'anchor_weight': args.anchor_weight,
            'com_weight': args.com_weight,
            'cls_weight': args.cls_weight,
            'loss_schedule': args.loss_schedule,
            'phase_a_epochs': args.phase_a_epochs,
            'target_metric_weight': args.target_metric_weight,
            'target_anchor_weight': args.target_anchor_weight,
            'target_cls_weight': args.target_cls_weight,
            'schedule_ramp': args.schedule_ramp,
            'cls_cutoff_epoch': args.cls_cutoff_epoch,
            'cls_cutoff_value': args.cls_cutoff_value,
            'keep_anchor_constant': args.keep_anchor_constant,
            'unified_weight': args.unified_weight,
            'unified_num_neighbors': args.unified_num_neighbors,
            'unified_epsilon': args.unified_epsilon,
            'use_dynamic_sampling': args.use_dynamic_sampling,
            'virtual_epoch_size': args.virtual_epoch_size,
            'sigma_ratio': args.sigma_ratio,
            'target_faces': args.target_faces,
            'render_mode': args.render_mode,
            'render_resolution': args.render_resolution,
            'quick_test': args.quick_test,
            'seed': 42,
        }

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)



    # 快速测试模式
    if args.quick_test:
        logger.info("快速测试模式")
        args.num_samples = 1000
        args.epochs = 10
        args.render_resolution = 256

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 记录实验配置
    experiment_config = {
        'input_mesh': args.input_mesh,
        'texture': args.texture,
        'num_samples': args.num_samples,
        'chart_mode': args.chart_mode,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'metric_weight': args.metric_weight,
        'anchor_weight': args.anchor_weight,
        'com_weight': args.com_weight,
        'cls_weight': args.cls_weight,
        'encoder_type': args.encoder_type,
        'activation': args.activation,
        'hash_lr': args.hash_lr,
        'hash_num_levels': args.hash_num_levels,
        'hash_features_per_level': args.hash_features_per_level,
        'hash_log2_size': args.hash_log2_size,
        'hash_base_res': args.hash_base_res,
        'hash_max_res': args.hash_max_res,
        'hash_cuda_backend': args.hash_cuda_backend,
        'hash_weight_decay': args.hash_weight_decay,
        'mlp_weight_decay': args.mlp_weight_decay,
        'loss_schedule': args.loss_schedule,
        'phase_a_epochs': args.phase_a_epochs,
        'target_metric_weight': args.target_metric_weight,
        'target_anchor_weight': args.target_anchor_weight,
        'target_cls_weight': args.target_cls_weight,
        'schedule_ramp': args.schedule_ramp,
        # NEW: Hard Classification Cutoff & Improved Strategy
        'cls_cutoff_epoch': args.cls_cutoff_epoch,
        'cls_cutoff_value': args.cls_cutoff_value,
        'keep_anchor_constant': args.keep_anchor_constant,
        'unified_weight': args.unified_weight,
        'unified_num_neighbors': args.unified_num_neighbors,
        'unified_epsilon': args.unified_epsilon,
        # ========================
        # DEPRECATED: Dynamic Classification Decay (replaced by hard cutoff)
        # 'dynamic_cls_decay': args.dynamic_cls_decay,
        # 'cls_decay_epoch_threshold': args.cls_decay_epoch_threshold,
        # 'cls_acc_threshold': args.cls_acc_threshold,
        # ========================
        'use_dynamic_sampling': args.use_dynamic_sampling,
        'virtual_epoch_size': args.virtual_epoch_size,
        'sigma_ratio': args.sigma_ratio,
        'target_faces': args.target_faces,
        'render_mode': args.render_mode,
        'render_resolution': args.render_resolution,
        'device': args.device,
        'quick_test': args.quick_test,
        'timestamp': datetime.now().isoformat(),
    }

    logger.info(f"实验输出目录: {output_dir}")
    logger.info(f"实验配置: {experiment_config}")

    # 开始实验
    start_time = time.time()

    try:
        # 步骤 1: 数据准备
        success = bake_data(
            input_mesh=args.input_mesh,
            output_dir=output_dir,
            num_samples=args.num_samples,
            chart_mode=args.chart_mode,
            texture_path=args.texture,
            use_dynamic_sampling=args.use_dynamic_sampling,
            virtual_epoch_size=args.virtual_epoch_size,
            sigma_ratio=args.sigma_ratio,
        )

        if not success:
            logger.error("数据准备失败，实验终止")
            return 1

        # 确定数据路径
        if args.use_dynamic_sampling:
            baked_data_path = output_dir / "mesh_constants.pt"
        else:
            baked_data_path = output_dir / "baked.pt"

        # 步骤 2: 训练模型
        success = train_model(
            baked_data=baked_data_path,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            metric_weight=args.metric_weight,
            anchor_weight=args.anchor_weight,
            cls_weight=args.cls_weight,
            encoder_type=args.encoder_type,
            activation=args.activation,
            com_weight=args.com_weight,
            hash_lr=args.hash_lr,
            hash_num_levels=args.hash_num_levels,
            hash_features_per_level=args.hash_features_per_level,
            hash_log2_size=args.hash_log2_size,
            hash_base_res=args.hash_base_res,
            hash_max_res=args.hash_max_res,
            hash_cuda_backend=args.hash_cuda_backend,
            hash_weight_decay=args.hash_weight_decay,
            mlp_weight_decay=args.mlp_weight_decay,
            loss_schedule=args.loss_schedule,
            phase_a_epochs=args.phase_a_epochs,
            target_metric_weight=args.target_metric_weight,
            target_anchor_weight=args.target_anchor_weight,
            target_cls_weight=args.target_cls_weight,
            schedule_ramp=args.schedule_ramp,
            # NEW: Hard Classification Cutoff & Improved Strategy
            cls_cutoff_epoch=args.cls_cutoff_epoch,
            cls_cutoff_value=args.cls_cutoff_value,
            keep_anchor_constant=args.keep_anchor_constant,
            unified_weight=args.unified_weight,
            unified_num_neighbors=args.unified_num_neighbors,
            unified_epsilon=args.unified_epsilon,
            # ========================
            # DEPRECATED: Dynamic Classification Decay
            # dynamic_cls_decay=args.dynamic_cls_decay,
            # cls_decay_epoch_threshold=args.cls_decay_epoch_threshold,
            # cls_acc_threshold=args.cls_acc_threshold,
            # ========================
            use_dynamic_sampling=args.use_dynamic_sampling,
            virtual_epoch_size=args.virtual_epoch_size,
            sigma_ratio=args.sigma_ratio,
        )

        if not success:
            logger.error("模型训练失败，实验终止")
            return 1

        best_checkpoint = output_dir / "train" / "best.pt"

        # 步骤 3: 推理和渲染
        success = infer_and_render(
            checkpoint=best_checkpoint,
            input_mesh=args.input_mesh,
            texture_path=args.texture,
            output_dir=output_dir,
            target_faces=args.target_faces,
            device=args.device,
            render_mode=args.render_mode,
            render_resolution=args.render_resolution,
        )

        if not success:
            logger.error("推理渲染失败，实验终止")
            return 1

        # 步骤 4: 计算指标
        metrics = compute_metrics(output_dir)

        # 检查渲染覆盖率
        if 'rendering' in metrics:
            coverage = metrics['rendering'].get('pixel_coverage', 0.0)

            # 覆盖率阈值：至少 5%
            min_coverage_threshold = 0.05

            if coverage < min_coverage_threshold:
                logger.error(f"渲染覆盖率过低: {coverage:.2%} < {min_coverage_threshold:.2%}，实验终止")
                logger.error("这通常意味着渲染器的坐标投影有问题")

                # 标记实验失败
                metrics['experiment_success'] = False
                metrics['failure_reason'] = f"Low render coverage: {coverage:.2%}"

                # 保存失败的指标
                save_results(output_dir, metrics, experiment_config)

                return 1
            else:
                logger.info(f"✓ 渲染覆盖率正常: {coverage:.2%}")

        # 步骤 5: 保存结果
        save_results(output_dir, metrics, experiment_config)

        total_elapsed = time.time() - start_time

        logger.info(f"\n{'='*60}")
        logger.info(f"实验完成! 总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
        logger.info(f"{'='*60}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n实验被用户中断")
        return 1
    except Exception as e:
        logger.error(f"\n实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
