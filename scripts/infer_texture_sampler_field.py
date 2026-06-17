"""
TextureSamplerField 推理脚本

从训练好的 checkpoint 出发，将纹理采样场应用到低模上。
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
from typing import Dict, Tuple, Optional, Union

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.texture_sampler_inference import TextureSamplerFieldInference
from src.inference.mesh_simplification import MeshSimplifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TextureSamplerFieldDemoPipeline:
    """
    完整的推理管线

    包括：模型加载、低模准备、scale 估计、UV 预测、OBJ 导出
    """

    def __init__(
        self,
        checkpoint_path: str,
        texture_path: str = None,
        device: str = "cuda",
    ):
        """
        初始化推理管线

        Args:
            checkpoint_path: 训练好的 checkpoint 路径
            texture_path: 纹理路径（可选，覆盖 checkpoint）
            device: 推理设备
        """
        # 加载推理器
        self.inference = TextureSamplerFieldInference(
            checkpoint_path=checkpoint_path,
            texture_path=texture_path,
            device=device,
        )

        # 保存 checkpoint 路径
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
            simplify: 是否减面（默认 True）

        Returns:
            低模 mesh
        """
        logger.info(f"准备低模...")

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

        # 如果不减面，直接返回高模
        if not simplify:
            logger.info(f"不减面，直接使用高模: {input_mesh_path}")
            return high_mesh

        # 否则从高模减面
        logger.info(f"从高模减面: {input_mesh_path}")
        if isinstance(high_mesh, trimesh.Scene):
            high_mesh = list(high_mesh.geometry.values())[0]

        # 创建临时文件用于减面
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
                logger.info(f"减面比例: {face_ratio:.1%} → {target} 面")
            else:
                target = int(len(high_mesh.faces) * 0.05)
                logger.info(f"使用默认减面比例 5% → {target} 面")

            # 执行减面
            low_mesh = simplifier.simplify_by_count(target, method="quadric", aggression=10)

        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir)

        return low_mesh

    def estimate_scales(
        self,
        mesh: trimesh.Trimesh,
        mode: str = "edge_mean",
        fixed_scale: float = None,
        scale_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        为每个顶点估计 scale

        Args:
            mesh: 输入 mesh
            mode: 估计模式
                - "edge_mean": 顶点邻接边平均长度
                - "face_area": 面积相关（暂未实现）
                - "fixed": 使用固定值
            fixed_scale: 固定 scale 值（mode="fixed" 时使用）
            scale_multiplier: scale 倍数

        Returns:
            scales: (V,) 每个顶点的 scale
        """
        vertices = mesh.vertices  # (V, 3)
        faces = mesh.faces  # (F, 3)

        if mode == "fixed":
            if fixed_scale is None:
                raise ValueError("mode='fixed' 时必须提供 --fixed-scale 参数")
            scales = np.full(len(vertices), fixed_scale, dtype=np.float32)
            logger.info(f"使用固定 scale: {fixed_scale}")

        elif mode == "edge_mean":
            # 计算每个顶点的邻接边平均长度
            # 使用 trimesh 的 edges_unique 和 edges_unique_length 获取正确的邻接关系
            edges = mesh.edges_unique  # (E, 2) 每条边的两个顶点索引
            edge_lengths = mesh.edges_unique_length  # (E,) 每条边的长度

            # 为每个顶点收集邻接边长度
            vertex_edge_lengths = [[] for _ in range(len(vertices))]
            for (v0_idx, v1_idx), edge_len in zip(edges, edge_lengths):
                vertex_edge_lengths[v0_idx].append(edge_len)
                vertex_edge_lengths[v1_idx].append(edge_len)

            # 计算平均边长
            scales = np.array([
                np.mean(edges) if edges else 0.01
                for edges in vertex_edge_lengths
            ], dtype=np.float32)

            logger.info(f"边长统计: min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}")

        else:
            raise ValueError(f"未知 scale 模式: {mode}")

        # 应用倍数
        scales = scales * scale_multiplier

        # Clip 到 checkpoint 的范围
        min_scale = self.inference.metadata.get('min_scale', 0.001)
        max_scale = self.inference.metadata.get('max_scale', 0.05)
        scales = np.clip(scales, min_scale, max_scale)

        logger.info(f"Scale 统计（clip 后）: min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}")
        logger.info(f"  (范围: [{min_scale}, {max_scale}])")

        return scales

    def predict_corner_uvs(
        self,
        mesh: trimesh.Trimesh,
        uv_mode: str = "argmax",
        scale_mode: str = "edge_mean",
        fixed_scale: float = None,
        scale_multiplier: float = 1.0,
        batch_size: int = 8192,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        为 mesh 的每个 face corner 预测 UV

        Args:
            mesh: 输入 mesh
            uv_mode: UV 选择模式（"argmax" 或 "weighted"）
            scale_mode: scale 估计模式
            fixed_scale: 固定 scale
            scale_multiplier: scale 倍数
            batch_size: 批处理大小

        Returns:
            corner_uvs: (F*3, 2) 每个 corner 的 UV
            corner_weights: (F*3, K) 每个 corner 的权重分布
            corner_sigmas: (F*3, K, 1) 每个 corner 的网络输出 sigma
            stats: 统计信息字典
        """
        vertices = mesh.vertices  # (V, 3)
        faces = mesh.faces  # (F, 3)

        # 获取所有 corner 的位置和 scale
        corner_positions = vertices[faces].reshape(-1, 3)  # (F*3, 3)
        corner_vertex_indices = faces.reshape(-1)  # (F*3,)

        # 估计 scale
        vertex_scales = self.estimate_scales(
            mesh, mode=scale_mode, fixed_scale=fixed_scale, scale_multiplier=scale_multiplier
        )
        corner_scales = vertex_scales[corner_vertex_indices]  # (F*3,)

        logger.info(f"预测 {len(corner_positions)} 个 corner 的 UV 分布...")

        # 批量预测
        uvs, weights, sigmas = self.inference.predict_distribution(
            corner_positions,
            corner_scales,
            batch_size=batch_size
        )

        # 选择 UV
        selected_uvs = self.inference.select_uvs(uvs, weights, mode=uv_mode)

        # 统计信息
        stats = {
            'num_corners': len(corner_positions),
            'uv_mode': uv_mode,
            'scale_mode': scale_mode,
            'scale_min': float(corner_scales.min()),
            'scale_max': float(corner_scales.max()),
            'scale_mean': float(corner_scales.mean()),
            'weight_max': float(weights.max()),
            'weight_mean': float(weights.mean()),
            'entropy_mean': float(-(weights * np.log(weights + 1e-8)).sum(axis=-1).mean()),
            'sigma_min': float(sigmas.min()),
            'sigma_max': float(sigmas.max()),
            'sigma_mean': float(sigmas.mean()),
        }

        logger.info(f"预测完成")
        logger.info(f"  UV 模式: {uv_mode}")
        logger.info(f"  权重统计: max={stats['weight_max']:.4f}, mean={stats['weight_mean']:.4f}")
        logger.info(f"  熵平均: {stats['entropy_mean']:.4f}")

        return selected_uvs, weights, sigmas, stats

    def export_obj_with_uv(
        self,
        mesh: trimesh.Trimesh,
        corner_uvs: np.ndarray,
        output_obj_path: str,
        texture_relative_path: str = None,
        copy_texture: bool = True,
    ):
        """
        导出带 UV 的 OBJ 文件

        Args:
            mesh: 输入 mesh
            corner_uvs: (F*3, 2) 每个 corner 的 UV
            output_obj_path: 输出 OBJ 路径
            texture_relative_path: 纹理相对路径（相对于 OBJ）
            copy_texture: 是否将纹理复制到输出目录（默认 True，保证便携性）
        """
        vertices = mesh.vertices  # (V, 3)
        faces = mesh.faces  # (F, 3)

        # 准备输出路径
        output_path = Path(output_obj_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        obj_path = output_path
        mtl_path = output_path.with_suffix('.mtl')

        # 确定纹理路径
        if texture_relative_path is None:
            # 默认将纹理复制到输出目录（保证便携性）
            if copy_texture:
                texture_abs = Path(self.inference.texture_path).resolve()
                texture_filename = texture_abs.name
                texture_output = obj_path.parent / texture_filename

                # 复制纹理文件（仅当不在同一位置时）
                if texture_abs != texture_output:
                    import shutil
                    shutil.copy2(texture_abs, texture_output)
                    logger.info(f"复制纹理到输出目录: {texture_output}")
                texture_rel = texture_filename
            else:
                # 不复制，使用相对路径
                texture_abs = Path(self.inference.texture_path).resolve()
                obj_dir = obj_path.parent.resolve()

                # 计算相对路径（允许包含 ..，这是 OBJ/MTL 标准做法）
                try:
                    texture_rel = os.path.relpath(texture_abs, obj_dir)
                except (ValueError, OSError):
                    # 跨驱动器或其他错误，使用绝对路径
                    texture_rel = str(texture_abs)
        else:
            texture_rel = texture_relative_path

        # 写 OBJ 文件
        with open(obj_path, 'w') as f:
            # 写 mtllib
            f.write(f"mtllib {mtl_path.name}\n")

            # 写顶点
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # 写 UV（每个 corner 一个 vt）
            for uv in corner_uvs:
                # 注意：这里不翻转 V，OBJ 的 VT 坐标原点在左下角
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # 写 usemtl
            f.write(f"usemtl material_0\n")

            # 写面（每个 corner 都有不同的 vt）
            vt_idx = 1  # vt 从 1 开始计数
            for face in faces:
                # 每个面有 3 个 corner，每个 corner 有自己的 vt
                f.write("f")
                for j, v_idx in enumerate(face):
                    f.write(f" {v_idx + 1}/{vt_idx + j}")
                f.write("\n")
                vt_idx += 3

        # 写 MTL 文件
        with open(mtl_path, 'w') as f:
            f.write("newmtl material_0\n")
            f.write(f"Ka 1.0 1.0 1.0\n")
            f.write(f"Kd 1.0 1.0 1.0\n")
            f.write(f"map_Kd {texture_rel}\n")

        logger.info(f"导出 OBJ: {obj_path}")
        logger.info(f"  顶点数: {len(vertices)}")
        logger.info(f"  面数: {len(faces)}")
        logger.info(f"  UV 数: {len(corner_uvs)}")
        logger.info(f"  纹理: {texture_rel}")

    def export_vertex_colors_debug(
        self,
        mesh: trimesh.Trimesh,
        output_obj_path: str,
        scale_mode: str = "edge_mean",
        fixed_scale: float = None,
        scale_multiplier: float = 1.0,
        batch_size: int = 8192,
    ):
        """
        导出带顶点颜色的 OBJ（仅用于调试）

        为每个顶点预测颜色并写入 OBJ。
        注意：这不是最终方法，只是用于调试 viewer。

        Args:
            mesh: 输入 mesh
            output_obj_path: 输出 OBJ 路径
            scale_mode: scale 估计模式
            fixed_scale: 固定 scale
            scale_multiplier: scale 倍数
            batch_size: 批处理大小
        """
        vertices = mesh.vertices  # (V, 3)

        # 估计 scale
        vertex_scales = self.estimate_scales(
            mesh, mode=scale_mode, fixed_scale=fixed_scale, scale_multiplier=scale_multiplier
        )

        # 预测颜色
        logger.info(f"预测 {len(vertices)} 个顶点的颜色...")
        colors = self.inference.predict_colors(
            vertices,
            vertex_scales,
            batch_size=batch_size
        )

        # 转换为 uint8
        colors_uint8 = (colors * 255).astype(np.uint8)

        # 创建带顶点色的 mesh
        mesh.visual.vertex_colors = colors_uint8

        # 导出
        output_path = Path(output_obj_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

        logger.info(f"导出调试 OBJ: {output_path}")

    def save_predictions_npz(
        self,
        output_path: str,
        corner_uvs: np.ndarray,
        weights: np.ndarray,
        sigmas: np.ndarray,
        mesh: trimesh.Trimesh,
    ):
        """
        保存预测结果为 NPZ（用于调试）

        Args:
            output_path: 输出路径
            corner_uvs: (F*3, 2) UV
            weights: (F*3, K) 权重
            sigmas: (F*3, K, 1) 尺度
            mesh: 输入 mesh
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vertices = mesh.vertices
        faces = mesh.faces
        corner_positions = vertices[faces].reshape(-1, 3)

        # 也可以计算预测颜色
        # 这里我们保存分布信息，不采样颜色

        np.savez_compressed(
            output_path,
            corner_positions=corner_positions,
            corner_uvs=corner_uvs,
            weights=weights,
            sigmas=sigmas,
            faces=faces,
        )

        logger.info(f"保存预测结果: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TextureSamplerField 推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础推理（减面到 5%）
  python scripts/infer_texture_sampler_field.py \\
      --checkpoint outputs/texture_sampler_bunny_full/best.pt \\
      --input-mesh data/models/stanford_bunny_textured.obj \\
      --output-dir outputs/texture_sampler_bunny_demo \\
      --face-ratio 0.05

  # 使用已有低模
  python scripts/infer_texture_sampler_field.py \\
      --checkpoint outputs/texture_sampler_bunny_full/best.pt \\
      --input-mesh data/models/stanford_bunny_textured.obj \\
      --low-mesh data/models/low_poly.obj \\
      --output-dir outputs/demo

  # 导出调试信息
  python scripts/infer_texture_sampler_field.py \\
      --checkpoint outputs/texture_sampler_bunny_full/best.pt \\
      --input-mesh data/models/stanford_bunny_textured.obj \\
      --output-dir outputs/demo \\
      --export-vertex-colors \\
      --export-npz
        """
    )

    # 必需参数
    parser.add_argument("--checkpoint", required=True, help="Checkpoint 路径")
    parser.add_argument("--input-mesh", required=True, help="输入高模路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")

    # 低模参数
    parser.add_argument("--low-mesh", help="直接使用低模（不减面）")
    parser.add_argument("--target-faces", type=int, help="目标面数")
    parser.add_argument("--face-ratio", type=float, help="面数比例（0.0-1.0）")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="不减面")
    parser.set_defaults(simplify=True)

    # 纹理参数
    parser.add_argument("--texture", help="覆盖 checkpoint 的纹理路径")
    parser.add_argument("--no-copy-texture", dest="copy_texture", action="store_false",
                       help="不复制纹理到输出目录（使用相对路径）")
    parser.set_defaults(copy_texture=True)

    # UV 选择
    parser.add_argument("--uv-mode", default="argmax", choices=["argmax", "weighted"],
                       help="UV 选择模式（默认: argmax）")

    # Scale 估计
    parser.add_argument("--scale-mode", default="edge_mean", choices=["edge_mean", "fixed"],
                       help="Scale 估计模式（默认: edge_mean）")
    parser.add_argument("--fixed-scale", type=float, help="固定 scale 值（mode=fixed 时使用）")
    parser.add_argument("--scale-multiplier", type=float, default=1.0,
                       help="Scale 倍数（默认: 1.0）")

    # 其他参数
    parser.add_argument("--batch-size", type=int, default=8192, help="批处理大小")
    parser.add_argument("--device", default="cuda", help="推理设备")

    # 导出选项
    parser.add_argument("--export-vertex-colors", action="store_true",
                       help="导出顶点颜色调试 OBJ")
    parser.add_argument("--export-npz", action="store_true",
                       help="导出预测结果 NPZ")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建推理管线
    pipeline = TextureSamplerFieldDemoPipeline(
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

    logger.info(f"低模统计: {len(low_mesh.vertices)} 顶点, {len(low_mesh.faces)} 面")

    # 预测 UV
    corner_uvs, weights, corner_sigmas, stats = pipeline.predict_corner_uvs(
        mesh=low_mesh,
        uv_mode=args.uv_mode,
        scale_mode=args.scale_mode,
        fixed_scale=args.fixed_scale,
        scale_multiplier=args.scale_multiplier,
        batch_size=args.batch_size,
    )

    # 导出 OBJ with UV
    obj_output = output_dir / f"low_textured_{args.uv_mode}.obj"
    pipeline.export_obj_with_uv(
        mesh=low_mesh,
        corner_uvs=corner_uvs,
        output_obj_path=str(obj_output),
        copy_texture=args.copy_texture,
    )

    # 可选：导出顶点颜色调试
    if args.export_vertex_colors:
        vertex_colors_output = output_dir / "low_vertex_colors.obj"
        pipeline.export_vertex_colors_debug(
            mesh=low_mesh,
            output_obj_path=str(vertex_colors_output),
            scale_mode=args.scale_mode,
            fixed_scale=args.fixed_scale,
            scale_multiplier=args.scale_multiplier,
            batch_size=args.batch_size,
        )

    # 可选：导出 NPZ
    if args.export_npz:
        npz_output = output_dir / "low_predictions.npz"
        pipeline.save_predictions_npz(
            output_path=str(npz_output),
            corner_uvs=corner_uvs,
            weights=weights,
            sigmas=corner_sigmas,
            mesh=low_mesh,
        )

    # 保存 metadata
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
        "scale_mode": args.scale_mode,
        "scale_multiplier": args.scale_multiplier,
        **stats,
        "output_files": {
            "obj": str(obj_output),
            "mtl": str(obj_output.with_suffix('.mtl')),
            "vertex_colors_obj": str(output_dir / "low_vertex_colors.obj") if args.export_vertex_colors else None,
            "predictions_npz": str(output_dir / "low_predictions.npz") if args.export_npz else None,
        },
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"保存 metadata: {metadata_path}")

    logger.info("推理完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"主要文件: {obj_output}")


if __name__ == "__main__":
    main()
