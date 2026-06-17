"""
MA-IUVF 推理脚本

从训练好的checkpoint出发，将UV场应用到低模
"""

import argparse
import sys
import os
import torch
import numpy as np
import trimesh
from pathlib import Path
import logging
import json
from typing import Dict, Tuple

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.metric_aligned_iuv_inference import MetricAlignedIUVInference
from src.inference.mesh_simplification import MeshSimplifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MetricAlignedIUVDemoPipeline:
    """MA-IUVF推理管线"""

    def __init__(
        self,
        checkpoint_path: str,
        texture_path: str = None,
        device: str = "cuda",
    ):
        """
        初始化推理管线

        Args:
            checkpoint_path: 训练好的checkpoint路径
            texture_path: 纹理路径（可选）
            device: 推理设备
        """
        # 加载推理器
        self.inference = MetricAlignedIUVInference(
            checkpoint_path=checkpoint_path,
            texture_path=texture_path,
            device=device,
        )

        self.checkpoint_path = checkpoint_path

    def prepare_low_mesh(
        self,
        input_mesh_path: str,
        low_mesh_path: str = None,
        target_faces: int = None,
        face_ratio: float = None,
        simplify: bool = True,
    ) -> trimesh.Trimesh:
        """
        准备低模

        Args:
            input_mesh_path: 原始高模路径
            low_mesh_path: 如果提供，直接使用此低模
            target_faces: 目标面数
            face_ratio: 面数比例
            simplify: 是否减面

        Returns:
            低模mesh
        """
        logger.info("准备低模...")

        # 如果直接提供低模
        if low_mesh_path:
            logger.info(f"直接使用低模: {low_mesh_path}")
            low_mesh = trimesh.load(low_mesh_path)
            if isinstance(low_mesh, trimesh.Scene):
                low_mesh = list(low_mesh.geometry.values())[0]
            return low_mesh

        # 加载高模
        high_mesh = trimesh.load(input_mesh_path)
        if isinstance(high_mesh, trimesh.Scene):
            high_mesh = list(high_mesh.geometry.values())[0]

        # 如果不减面
        if not simplify:
            logger.info(f"不减面，直接使用高模: {input_mesh_path}")
            return high_mesh

        # 减面
        logger.info(f"从高模减面: {input_mesh_path}")

        # 创建临时文件
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_mesh_path = temp_dir / "temp_mesh.obj"
        high_mesh.export(str(temp_mesh_path))

        try:
            simplifier = MeshSimplifier(str(temp_mesh_path))

            # 确定目标面数
            if target_faces:
                target = target_faces
                logger.info(f"减面到目标面数: {target}")
            elif face_ratio:
                target = int(len(high_mesh.faces) * face_ratio)
                logger.info(f"减面比例: {face_ratio:.1%} → {target}面")
            else:
                target = int(len(high_mesh.faces) * 0.05)
                logger.info(f"使用默认减面比例5% → {target}面")

            # 执行减面
            low_mesh = simplifier.simplify_by_count(target, method="quadric", aggression=10)

        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir)

        return low_mesh

    def predict_corner_uvs(
        self,
        mesh: trimesh.Trimesh,
        batch_size: int = 8192,
        uv_mode: str = "argmax",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        为mesh的每个face corner预测UV

        Args:
            mesh: 输入mesh
            batch_size: 批处理大小
            uv_mode: UV选择模式

        Returns:
            corner_uvs: [F*3, 2] 每个corner的UV
            chart_ids: [F*3] 每个corner的chart ID
            logits: [F*3, C] chart分类logits
            stats: 统计信息
        """
        vertices = mesh.vertices  # [V, 3]
        faces = mesh.faces  # [F, 3]

        # 获取所有corner的位置
        corner_positions = vertices[faces].reshape(-1, 3)  # [F*3, 3]

        logger.info(f"预测{len(corner_positions)}个corner的UV...")

        # 批量预测
        logits, uv_preds = self.inference.predict(corner_positions, batch_size)

        # 选择UV
        selected_uvs, chart_ids = self.inference.select_uvs(logits, uv_preds, mode=uv_mode)

        # 统计信息
        num_charts = self.inference.metadata['num_charts']
        chart_distribution = np.bincount(chart_ids, minlength=num_charts)

        stats = {
            'num_corners': len(corner_positions),
            'uv_mode': uv_mode,
            'num_charts': num_charts,
            'chart_distribution': chart_distribution.tolist(),
        }

        logger.info(f"预测完成")
        logger.info(f"  UV模式: {uv_mode}")
        logger.info(f"  Charts: {num_charts}")
        logger.info(f"  Chart分布: {chart_distribution}")

        return selected_uvs, chart_ids, logits, stats

    def export_obj_with_uv(
        self,
        mesh: trimesh.Trimesh,
        corner_uvs: np.ndarray,
        output_obj_path: str,
        copy_texture: bool = True,
    ):
        """
        导出带UV的OBJ文件

        Args:
            mesh: 输入mesh
            corner_uvs: [F*3, 2] 每个corner的UV
            output_obj_path: 输出OBJ路径
            copy_texture: 是否复制纹理到输出目录
        """
        vertices = mesh.vertices  # [V, 3]
        faces = mesh.faces  # [F, 3]

        # 准备输出路径
        output_path = Path(output_obj_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        obj_path = output_path
        mtl_path = output_path.with_suffix('.mtl')

        # 确定纹理路径
        if self.inference.texture_path:
            if copy_texture:
                # 复制纹理到输出目录
                texture_abs = Path(self.inference.texture_path).resolve()
                texture_filename = texture_abs.name
                texture_output = obj_path.parent / texture_filename

                if texture_abs != texture_output:
                    import shutil
                    shutil.copy2(texture_abs, texture_output)
                    logger.info(f"复制纹理: {texture_output}")

                texture_rel = texture_filename
            else:
                # 使用相对路径
                texture_abs = Path(self.inference.texture_path).resolve()
                obj_dir = obj_path.parent.resolve()

                try:
                    texture_rel = os.path.relpath(texture_abs, obj_dir)
                except (ValueError, OSError):
                    texture_rel = str(texture_abs)
        else:
            logger.warning("没有纹理，MTL将不包含map_Kd")
            texture_rel = None

        # 写OBJ文件
        with open(obj_path, 'w') as f:
            # 写mtllib
            f.write(f"mtllib {mtl_path.name}\n")

            # 写顶点
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # 写UV（每个corner一个vt）
            for uv in corner_uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # 写usemtl
            f.write("usemtl material_0\n")

            # 写面（每个corner都有不同的vt）
            vt_idx = 1  # vt从1开始计数
            for face in faces:
                f.write("f")
                for j, v_idx in enumerate(face):
                    f.write(f" {v_idx + 1}/{vt_idx + j}")
                f.write("\n")
                vt_idx += 3

        # 写MTL文件
        with open(mtl_path, 'w') as f:
            f.write("newmtl material_0\n")
            f.write("Ka 1.0 1.0 1.0\n")
            f.write("Kd 1.0 1.0 1.0\n")
            if texture_rel:
                f.write(f"map_Kd {texture_rel}\n")

        logger.info(f"导出OBJ: {obj_path}")
        logger.info(f"  顶点数: {len(vertices)}")
        logger.info(f"  面数: {len(faces)}")
        logger.info(f"  UV数: {len(corner_uvs)}")

    def save_predictions_npz(
        self,
        output_path: str,
        corner_uvs: np.ndarray,
        chart_ids: np.ndarray,
        logits: np.ndarray,
        mesh: trimesh.Trimesh,
    ):
        """
        保存预测结果为NPZ

        Args:
            output_path: 输出路径
            corner_uvs: [F*3, 2] UV
            chart_ids: [F*3] chart ID
            logits: [F*3, C] logits
            mesh: 输入mesh
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vertices = mesh.vertices
        faces = mesh.faces
        corner_positions = vertices[faces].reshape(-1, 3)

        np.savez_compressed(
            output_path,
            corner_positions=corner_positions,
            corner_uvs=corner_uvs,
            chart_ids=chart_ids,
            logits=logits,
            faces=faces,
        )

        logger.info(f"保存预测结果: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MA-IUVF推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必需参数
    parser.add_argument("--checkpoint", required=True, help="Checkpoint路径")
    parser.add_argument("--input-mesh", required=True, help="输入高模路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")

    # 低模参数
    parser.add_argument("--low-mesh", help="直接使用低模（不减面）")
    parser.add_argument("--target-faces", type=int, help="目标面数")
    parser.add_argument("--face-ratio", type=float, help="面数比例（0.0-1.0）")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="不减面")
    parser.set_defaults(simplify=True)

    # 纹理参数
    parser.add_argument("--texture", help="覆盖checkpoint的纹理路径")
    parser.add_argument("--no-copy-texture", dest="copy_texture", action="store_false",
                       help="不复制纹理到输出目录")
    parser.set_defaults(copy_texture=True)

    # UV选择
    parser.add_argument("--uv-mode", default="argmax", choices=["argmax", "sample"],
                       help="UV选择模式（默认: argmax）")

    # 其他参数
    parser.add_argument("--batch-size", type=int, default=8192, help="批处理大小")
    parser.add_argument("--device", default="cuda", help="推理设备")

    # 导出选项
    parser.add_argument("--export-npz", action="store_true", help="导出预测结果NPZ")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建推理管线
    pipeline = MetricAlignedIUVDemoPipeline(
        checkpoint_path=args.checkpoint,
        texture_path=args.texture,
        device=args.device,
    )

    # 准备低模
    low_mesh = pipeline.prepare_low_mesh(
        input_mesh_path=args.input_mesh,
        low_mesh_path=args.low_mesh,
        target_faces=args.target_faces,
        face_ratio=args.face_ratio,
        simplify=args.simplify,
    )

    logger.info(f"低模统计: {len(low_mesh.vertices)}顶点, {len(low_mesh.faces)}面")

    # 预测UV
    corner_uvs, chart_ids, logits, stats = pipeline.predict_corner_uvs(
        mesh=low_mesh,
        batch_size=args.batch_size,
        uv_mode=args.uv_mode,
    )

    # 导出OBJ with UV
    obj_output = output_dir / f"low_maiuvf_textured.obj"
    pipeline.export_obj_with_uv(
        mesh=low_mesh,
        corner_uvs=corner_uvs,
        output_obj_path=str(obj_output),
        copy_texture=args.copy_texture,
    )

    # 可选：导出NPZ
    if args.export_npz:
        npz_output = output_dir / "low_maiuvf_predictions.npz"
        pipeline.save_predictions_npz(
            output_path=str(npz_output),
            corner_uvs=corner_uvs,
            chart_ids=chart_ids,
            logits=logits,
            mesh=low_mesh,
        )

    # 保存metadata
    metadata = {
        "checkpoint_path": args.checkpoint,
        "input_mesh_path": args.input_mesh,
        "low_mesh_path": str(args.low_mesh) if args.low_mesh else None,
        "low_mesh_stats": {
            "num_vertices": int(len(low_mesh.vertices)),
            "num_faces": int(len(low_mesh.faces)),
        },
        "texture_path": pipeline.inference.texture_path,
        "uv_mode": args.uv_mode,
        **stats,
        "output_files": {
            "obj": str(obj_output),
            "mtl": str(obj_output.with_suffix('.mtl')),
            "predictions_npz": str(output_dir / "low_maiuvf_predictions.npz") if args.export_npz else None,
        },
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"保存metadata: {metadata_path}")

    logger.info("推理完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"主要文件: {obj_output}")


if __name__ == "__main__":
    main()
