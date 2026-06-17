"""
MA-IUVF 离线渲染测试脚本

实现简单的CPU rasterizer验证训练结果
"""

import argparse
import sys
import subprocess
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def simple_cpu_rasterizer(
    checkpoint_path: str,
    input_mesh: str,
    texture_path: str,
    output_path: str,
    resolution: int = 512,
    device: str = "cuda",
):
    """
    简单CPU rasterizer（使用新的离屏渲染器）

    流程：
    1. 光栅化网格得到像素3D坐标
    2. 批量调用MA-IUVF得到UV
    3. 从纹理采样
    4. 保存结果

    Args:
        checkpoint_path: 模型checkpoint
        input_mesh: 输入mesh
        texture_path: 纹理路径
        output_path: 输出图像路径
        resolution: 图像分辨率
        device: 设备
    """
    from src.inference.metric_aligned_iuv_inference import MetricAlignedIUVInference
    from src.inference.offline_renderer import OfflineRenderer
    from src.data.obj_parser import parse_obj_file
    import torch

    logger.info("启动离屏渲染器...")

    # 加载模型
    logger.info(f"加载模型: {checkpoint_path}")
    inference = MetricAlignedIUVInference(
        checkpoint_path=checkpoint_path,
        texture_path=texture_path,
        device=device,
    )

    # 加载网格
    logger.info(f"加载网格: {input_mesh}")

    if input_mesh.endswith('.obj'):
        # 使用 OBJ 解析器
        obj_data = parse_obj_file(input_mesh)
        vertices = obj_data['vertices']
        faces = obj_data['faces']
    else:
        # 使用 trimesh
        import trimesh
        mesh = trimesh.load(input_mesh)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        vertices = mesh.vertices
        faces = mesh.faces

    # 加载纹理
    logger.info(f"加载纹理: {texture_path}")
    texture_img = Image.open(texture_path)
    texture = np.array(texture_img)

    # 创建渲染器
    renderer = OfflineRenderer(
        mesh_vertices=vertices,
        mesh_faces=faces,
        texture_image=texture,
        resolution=(resolution, resolution),
    )

    # 使用 MA-IUVF 渲染
    logger.info("开始渲染...")

    # 获取 baker_metadata
    baker_metadata = inference.metadata.get('baker_metadata', None)

    rendered_image, render_info = renderer.render_with_maiuvf(
        model=inference.model,
        device=torch.device(device),
        baker_metadata=baker_metadata,
    )

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer.save_render(rendered_image, str(output_path))
    buffer_paths = renderer.save_prediction_buffers(str(output_path.parent), prefix="pred")
    if buffer_paths:
        render_info['outputs'] = {
            **render_info.get('outputs', {}),
            **buffer_paths,
        }

    # 打印统计信息
    logger.info(f"渲染统计:")
    logger.info(f"  - 有效像素: {render_info['valid_pixels']} / {render_info['total_pixels']}")
    logger.info(f"  - 覆盖率: {render_info['coverage']:.2%}")
    logger.info(f"  - 预测 chart 分布: {render_info['pred_chart_distribution']}")
    logger.info(f"  - 预测 charts 数: {render_info['pred_num_charts']}")

    if render_info['chart_accuracy'] is not None:
        logger.info(f"  - 分类准确率: {render_info['chart_accuracy']:.2%}")
        logger.info(f"  - 可见 GT charts: {render_info['visible_gt_num_charts']}")
        logger.info(f"  - GT chart 分布: {render_info['gt_chart_distribution']}")

    # 保存渲染信息
    import json
    info_path = output_path.parent / "render_info.json"
    with open(info_path, 'w') as f:
        json.dump(render_info, f, indent=2)
    logger.info(f"保存渲染信息: {info_path}")


def render_textured_obj(
    checkpoint_path: str,
    input_mesh: str,
    texture_path: str,
    output_dir: str,
    target_faces: int = 500,
    device: str = "cuda",
):
    """
    渲染textured OBJ

    调用推理脚本生成textured OBJ
    """
    logger.info("渲染textured OBJ...")

    # 调用推理脚本
    cmd = [
        "python", "scripts/infer_metric_aligned_iuv.py",
        "--checkpoint", checkpoint_path,
        "--input-mesh", input_mesh,
        "--texture", texture_path,
        "--output-dir", output_dir,
        "--target-faces", str(target_faces),
        "--device", device,
        "--export-npz",
    ]

    logger.info(f"运行: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)

    logger.info(f"渲染完成: {output_dir}")
    logger.info(f"主要文件: {Path(output_dir) / 'low_maiuvf_textured.obj'}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="MA-IUVF离线渲染测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 渲染textured OBJ
  python scripts/render_metric_aligned_iuv_test.py \\
      --checkpoint outputs/maiuvf_bunny/train_small/best.pt \\
      --input-mesh data/models/stanford_bunny_textured.obj \\
      --texture data/textures/bunny_texture.png \\
      --output-dir outputs/maiuvf_bunny/render_test \\
      --target-faces 500

  # 离屏渲染验证
  python scripts/render_metric_aligned_iuv_test.py \\
      --checkpoint outputs/maiuvf_bunny/train_small/best.pt \\
      --input-mesh data/models/stanford_bunny_textured.obj \\
      --texture data/textures/bunny_texture.png \\
      --output-dir outputs/maiuvf_bunny/render_test \\
      --render-mode cpu \\
      --resolution 256

  # 渲染后用viewer查看
  python scripts/viewer_3d.py outputs/maiuvf_bunny/render_test/low_maiuvf_textured.obj
        """
    )

    parser.add_argument("--checkpoint", required=True, help="Checkpoint路径")
    parser.add_argument("--input-mesh", required=True, help="输入高模路径")
    parser.add_argument("--texture", required=True, help="纹理路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--target-faces", type=int, default=500, help="目标面数")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--render-mode", default="obj", choices=["obj", "cpu"], help="渲染模式")
    parser.add_argument("--resolution", type=int, default=256, help="CPU渲染分辨率")
    parser.add_argument("--no-viewer", action="store_true", help="不自动启动viewer")

    args = parser.parse_args()

    if args.render_mode == "obj":
        # 渲染textured OBJ
        success = render_textured_obj(
            checkpoint_path=args.checkpoint,
            input_mesh=args.input_mesh,
            texture_path=args.texture,
            output_dir=args.output_dir,
            target_faces=args.target_faces,
            device=args.device,
        )
    elif args.render_mode == "cpu":
        # CPU离屏渲染
        output_path = Path(args.output_dir) / "render_cpu.png"
        simple_cpu_rasterizer(
            checkpoint_path=args.checkpoint,
            input_mesh=args.input_mesh,
            texture_path=args.texture,
            output_path=str(output_path),
            resolution=args.resolution,
            device=args.device,
        )
        success = True
    else:
        success = False

    if not success:
        logger.error("渲染失败")
        return 1

    # 启动viewer
    if not args.no_viewer and args.render_mode == "obj":
        obj_path = Path(args.output_dir) / "low_maiuvf_textured.obj"

        if obj_path.exists():
            logger.info(f"启动viewer查看: {obj_path}")

            viewer_cmd = ["python", "scripts/viewer_3d.py", str(obj_path)]
            logger.info(f"运行: {' '.join(viewer_cmd)}")

            try:
                subprocess.run(viewer_cmd, check=True)
            except subprocess.CalledProcessError:
                logger.warning("Viewer启动失败或已关闭")
        else:
            logger.error(f"OBJ文件不存在: {obj_path}")
            return 1

    logger.info("渲染测试完成!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
