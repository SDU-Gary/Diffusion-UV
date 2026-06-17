"""
MA-IUVF 数据烘焙脚本

从带UV的高模mesh烘焙训练数据
"""

import argparse
import sys
from pathlib import Path
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="MA-IUVF数据烘焙脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mesh", required=True, help="输入高模路径（必须包含UV）")
    parser.add_argument("--output", required=True, help="输出烘焙数据路径 (.pt)")
    parser.add_argument("--num-samples", type=int, default=100000, help="总样本数")
    parser.add_argument("--extrusion-sigma-ratio", type=float, default=0.01,
                       help="挤出标准差相对于bbox diagonal的比例")
    parser.add_argument("--chart-mode", default="face_component",
                       choices=["face_component", "uv_islands"],
                       help="Chart分配模式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--texture", help="纹理路径（可选，保存到metadata）")

    args = parser.parse_args()

    # 创建烘焙器（强制使用OBJ解析器）
    baker = MetricAlignedIUVBaker(
        mesh_path=args.mesh,
        seed=args.seed,
        use_obj_parser=True,  # 强制使用OBJ解析器以正确处理UV seams
    )

    # 烘焙数据
    data = baker.bake(
        num_samples=args.num_samples,
        extrusion_sigma_ratio=args.extrusion_sigma_ratio,
        chart_mode=args.chart_mode,
    )

    # 保存
    baker.save(
        data=data,
        output_path=args.output,
        chart_mode=args.chart_mode,
        extrusion_sigma_ratio=args.extrusion_sigma_ratio,
        texture_path=args.texture,
    )

    logger.info(f"烘焙完成! 输出: {args.output}")


if __name__ == "__main__":
    main()
