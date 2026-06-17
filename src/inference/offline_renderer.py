"""
离屏渲染验证

实现：
1. 光栅化低模得到每个像素的世界坐标
2. 将像素 3D 坐标送入 MA-IUVF
3. 用 argmax(logits) 选 chart
4. 用对应 UV 采样原始纹理
5. 保存 result.png
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# 尝试导入 OpenGL G-Buffer 渲染器
try:
    from .opengl_renderer import render_with_opengl_gbuffer
    OPENGL_AVAILABLE = True
    logger.info("OpenGL G-Buffer 渲染器可用")
except ImportError as e:
    OPENGL_AVAILABLE = False
    logger.warning(f"OpenGL G-Buffer 渲染器不可用: {e}，将使用 CPU 渲染器")
except Exception as e:
    OPENGL_AVAILABLE = False
    logger.warning(f"OpenGL G-Buffer 渲染器导入失败: {e}，将使用 CPU 渲染器")


class OfflineRenderer:
    """
    离屏渲染器

    光栅化低模网格，使用 MA-IUVF 预测 UV，采样纹理渲染
    """

    def __init__(
        self,
        mesh_vertices: np.ndarray,  # [V, 3]
        mesh_faces: np.ndarray,     # [F, 3]
        texture_image: np.ndarray,   # [H, W, 3] 纹理图像
        resolution: Tuple[int, int] = (512, 512),
        view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        初始化离屏渲染器

        Args:
            mesh_vertices: 网格顶点
            mesh_faces: 网格面（顶点索引）
            texture_image: 纹理图像
            resolution: 渲染分辨率 (width, height)
        """
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces
        self.texture_image = texture_image
        self.resolution = resolution
        self.view_bounds = view_bounds
        self.last_prediction_buffers = {}

        logger.info(f"初始化离屏渲染器: {resolution}, 纹理 {texture_image.shape}")

    def rasterize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        光栅化网格

        Returns:
            pixel_coords: [H, W, 3] 每个像素的 3D 坐标
            face_ids: [H, W] 每个像素所属的面 ID (-1 表示背景)
            bary_coords: [H, W, 3] 每个像素的重心坐标
        """
        if OPENGL_AVAILABLE:
            # 尝试使用 OpenGL G-Buffer 渲染器
            try:
                logger.info("尝试使用 OpenGL G-Buffer 渲染器")
                world_pos, face_ids, bary_coords = render_with_opengl_gbuffer(
                    self.mesh_vertices,
                    self.mesh_faces,
                    self.resolution,
                    view_bounds=self.view_bounds,
                )
                logger.info(f"OpenGL G-Buffer 渲染成功")
                return world_pos, face_ids, bary_coords
            except Exception as e:
                logger.warning(f"OpenGL G-Buffer 渲染失败，回退到 CPU 渲染器: {e}")
        else:
            logger.info("OpenGL 不可用，使用 CPU 渲染器")

        # 回退到 CPU 渲染器
        logger.info("使用 CPU 渲染器")
        cpu_texturizer = CPUTexturizer(
            self.mesh_vertices,
            self.mesh_faces,
            self.resolution,
            view_bounds=self.view_bounds,
        )

        # 光栅化
        pixel_coords, face_ids, bary_coords = cpu_texturizer.rasterize()

        logger.info(f"CPU 光栅化完成: {np.sum(face_ids >= 0)} / {face_ids.size} 像素覆盖")

        return pixel_coords, face_ids, bary_coords

    def render_with_maiuvf(
        self,
        model,
        device: torch.device,
        baker_metadata: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        使用 MA-IUVF 渲染

        Args:
            model: 训练好的 MA-IUVF 模型
            device: 计算设备
            baker_metadata: 烘焙元数据（包含 GT face_chart_id）

        Returns:
            rendered_image: [H, W, 3] 渲染图像
            render_info: dict 渲染信息
        """
        # 1. 光栅化
        logger.info("步骤 1/5: 光栅化网格")
        pixel_coords, face_ids, bary_coords = self.rasterize()

        height, width = self.resolution[1], self.resolution[0]

        # 2. 收集需要预测的像素（非背景）
        logger.info("步骤 2/5: 收集有效像素")
        valid_mask = face_ids >= 0
        valid_coords = pixel_coords[valid_mask]  # [N, 3]

        logger.info(f"有效像素: {len(valid_coords)} / {width * height}")

        if len(valid_coords) == 0:
            logger.error("没有有效像素，渲染失败")
            return np.zeros((height, width, 3), dtype=np.uint8), {}

        # 3. MA-IUVF 预测
        logger.info("步骤 3/5: MA-IUVF 预测 UV")
        with torch.no_grad():
            coords_tensor = torch.from_numpy(valid_coords.astype(np.float32)).to(device)
            coords_tensor = coords_tensor.unsqueeze(0)  # [1, N, 3]

            # 模型预测
            output = model(coords_tensor.squeeze(0))  # [N, C], [N, C, 2]

            logits = output.logits.cpu().numpy()      # [N, C]
            uv_preds = output.uv_preds.cpu().numpy()  # [N, C, 2]

        # 4. 选择 UV（argmax chart）
        logger.info("步骤 4/5: 选择 UV 并采样纹理")
        chart_ids = logits.argmax(axis=-1)  # [N]

        # 从对应的 chart 收集 UV
        batch_indices = np.arange(len(chart_ids))
        selected_uvs = uv_preds[batch_indices, chart_ids]  # [N, 2]

        pred_uv_image = np.zeros((height, width, 2), dtype=np.float32)
        pred_chart_image = np.full((height, width), -1, dtype=np.int32)
        pred_face_image = face_ids.astype(np.int32).copy()
        pred_uv_image[valid_mask] = selected_uvs.astype(np.float32)
        pred_chart_image[valid_mask] = chart_ids.astype(np.int32)
        self.last_prediction_buffers = {
            "pred_uv": pred_uv_image,
            "pred_chart_id": pred_chart_image,
            "pred_face_id": pred_face_image,
            "pred_valid_mask": valid_mask.astype(np.uint8),
        }

        # 5. 纹理采样
        logger.info("步骤 5/5: 纹理采样")
        rendered_image = self._sample_texture(selected_uvs, valid_mask)

        # 收集统计信息
        render_info = {
            'valid_pixels': int(valid_mask.sum()),
            'total_pixels': width * height,
            'coverage': float(valid_mask.sum() / (width * height)),
            'pred_chart_distribution': {},
            'gt_chart_distribution': {},
            'chart_accuracy': None,
            'visible_gt_num_charts': None,
            'pred_num_charts': None,
        }

        # 统计预测 chart 分布
        unique_charts, counts = np.unique(chart_ids, return_counts=True)
        for chart_id, count in zip(unique_charts, counts):
            render_info['pred_chart_distribution'][int(chart_id)] = int(count)
        render_info['pred_num_charts'] = len(unique_charts)

        # 如果有 baker_metadata，计算 GT 统计
        if baker_metadata is not None and 'face_chart_id' in baker_metadata:
            try:
                face_chart_id = np.array(baker_metadata['face_chart_id'])

                # 检查长度匹配
                if len(face_chart_id) == len(self.mesh_faces):
                    # 获取可见像素对应的 face_id
                    visible_face_ids = face_ids[valid_mask]  # [N]

                    # 获取这些 face 的 GT chart_id
                    gt_chart_ids = face_chart_id[visible_face_ids]  # [N]

                    # 统计 GT chart 分布
                    unique_gt_charts, gt_counts = np.unique(gt_chart_ids, return_counts=True)
                    for chart_id, count in zip(unique_gt_charts, gt_counts):
                        render_info['gt_chart_distribution'][int(chart_id)] = int(count)
                    render_info['visible_gt_num_charts'] = len(unique_gt_charts)

                    # 计算分类准确率
                    correct_preds = (chart_ids == gt_chart_ids).sum()
                    total_preds = len(chart_ids)
                    render_info['chart_accuracy'] = float(correct_preds) / total_preds if total_preds > 0 else 0.0

                    logger.info(f"分类准确率: {render_info['chart_accuracy']:.2%}")
                    logger.info(f"可见 GT charts: {render_info['visible_gt_num_charts']}")
                else:
                    logger.warning("face_chart_id 长度与 mesh_faces 不匹配，跳过 GT 统计")

            except Exception as e:
                logger.warning(f"计算 GT 统计失败: {e}")

        logger.info(f"渲染完成: 覆盖率 {render_info['coverage']:.2%}")
        logger.info(f"预测 chart 分布: {render_info['pred_chart_distribution']}")

        return rendered_image, render_info

    def _sample_texture(
        self,
        uv_coords: np.ndarray,  # [N, 2]
        valid_mask: np.ndarray,  # [H, W]
    ) -> np.ndarray:
        """
        从纹理采样

        Args:
            uv_coords: UV 坐标（范围 [0, 1]）
            valid_mask: 有效像素掩码

        Returns:
            sampled_image: [H, W, 3] 采样结果
        """
        height, width = self.resolution[1], self.resolution[0]
        tex_h, tex_w = self.texture_image.shape[:2]

        # 初始化输出图像
        output = np.zeros((height, width, 3), dtype=np.uint8)

        # 双线性纹理采样，保持与 render_high_uv_reference.py 一致的
        # bottom-left UV 约定：UV v=1 对应图像 y=0。
        u = np.clip(uv_coords[:, 0], 0.0, 1.0)
        v = np.clip(uv_coords[:, 1], 0.0, 1.0)

        x = u * (tex_w - 1)
        y = (1.0 - v) * (tex_h - 1)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, tex_w - 1)
        y1 = np.clip(y0 + 1, 0, tex_h - 1)

        wx = (x - x0).reshape(-1, 1)
        wy = (y - y0).reshape(-1, 1)

        # Handle both RGB (3 channels) and RGBA (4 channels) textures
        if self.texture_image.shape[2] == 4:
            c00 = self.texture_image[y0, x0, :3].astype(np.float32)
            c10 = self.texture_image[y0, x1, :3].astype(np.float32)
            c01 = self.texture_image[y1, x0, :3].astype(np.float32)
            c11 = self.texture_image[y1, x1, :3].astype(np.float32)
        else:
            c00 = self.texture_image[y0, x0].astype(np.float32)
            c10 = self.texture_image[y0, x1].astype(np.float32)
            c01 = self.texture_image[y1, x0].astype(np.float32)
            c11 = self.texture_image[y1, x1].astype(np.float32)

        c0 = c00 * (1.0 - wx) + c10 * wx
        c1 = c01 * (1.0 - wx) + c11 * wx
        sampled_colors = np.clip(c0 * (1.0 - wy) + c1 * wy, 0, 255).astype(np.uint8)

        # 填充到输出图像
        output[valid_mask] = sampled_colors

        return output

    def save_render(self, image: np.ndarray, output_path: str):
        """保存渲染结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存图像
        img = Image.fromarray(image)
        img.save(output_path)

        logger.info(f"保存渲染结果: {output_path}")

    def save_prediction_buffers(self, output_dir: str, prefix: str = "pred"):
        """Save per-pixel prediction buffers produced by the last render."""
        if not self.last_prediction_buffers:
            logger.warning("没有可保存的预测buffer，请先调用 render_with_maiuvf")
            return {}

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, array in self.last_prediction_buffers.items():
            path = output / f"{prefix}_{name.replace('pred_', '')}.npy"
            np.save(path, array)
            paths[name] = str(path)
        logger.info("保存预测buffer: %s", paths)
        return paths


class CPUTexturizer:
    """
    CPU 光栅化器（简化版）

    使用 z-buffer 和重心坐标插值
    """

    def __init__(
        self,
        vertices: np.ndarray,  # [V, 3]
        faces: np.ndarray,     # [F, 3]
        resolution: Tuple[int, int] = (512, 512),
        view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        初始化 CPU 光栅化器

        Args:
            vertices: 顶点坐标
            faces: 面索引
            resolution: 分辨率 (width, height)
        """
        self.vertices = vertices
        self.faces = faces
        self.width, self.height = resolution
        self.view_bounds = view_bounds

        # 计算等比例正交视口投影坐标
        self._compute_viewport_projection()

        # 计算每个面的 bbox
        self._compute_face_bboxes()

    def _compute_viewport_projection(self):
        """Project world XY into an aspect-preserving orthographic viewport."""
        if self.view_bounds is None:
            bbox_min = self.vertices.min(axis=0)
            bbox_max = self.vertices.max(axis=0)
        else:
            bbox_min = np.asarray(self.view_bounds[0], dtype=np.float64)
            bbox_max = np.asarray(self.view_bounds[1], dtype=np.float64)

        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_size = np.maximum(bbox_max - bbox_min, 1e-8)
        bbox_diagonal = np.linalg.norm(bbox_size)
        viewport_aspect = float(self.width) / max(float(self.height), 1.0)

        # Use one uniform world-to-screen scale. The wider axis fills the
        # viewport and the other axis receives letterbox/pillarbox margins.
        view_height = max(float(bbox_size[1]), float(bbox_size[0]) / viewport_aspect)
        view_width = view_height * viewport_aspect
        if view_width < 1e-8:
            view_width = 1.0
        if view_height < 1e-8:
            view_height = 1.0

        x0 = bbox_center[0] - view_width * 0.5
        y0 = bbox_center[1] - view_height * 0.5

        sx = (self.vertices[:, 0] - x0) / view_width
        # World coordinates are Y-up, image coordinates are top-down.
        sy = 1.0 - (self.vertices[:, 1] - y0) / view_height
        sz = (self.vertices[:, 2] - bbox_min[2]) / max(float(bbox_size[2]), 1e-8)

        self.screen_coords = np.stack([sx, sy, sz], axis=1).astype(np.float32)
        self.original_vertices = self.vertices.copy()  # [V, 3] 原始坐标

        logger.info(
            "Viewport projection: center=%s, bbox_size=%s, view_size=(%.6f, %.6f), "
            "aspect=%.4f, diagonal=%.4f",
            bbox_center,
            bbox_size,
            view_width,
            view_height,
            viewport_aspect,
            bbox_diagonal,
        )

    def _compute_face_bboxes(self):
        """计算每个面的 2D bbox"""
        self.face_bboxes = []

        for face in self.faces:
            face_verts = self.vertices[face]  # [3, 3]

            # 投影到屏幕空间（简化：直接使用 XY 坐标）
            x_coords = face_verts[:, 0]
            y_coords = face_verts[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            self.face_bboxes.append((x_min, y_min, x_max, y_max))

    def rasterize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        光栅化网格

        Returns:
            pixel_coords: [H, W, 3] 像素 3D 坐标
            face_ids: [H, W] 面 ID
            bary_coords: [H, W, 3] 重心坐标
        """
        # 初始化输出
        pixel_coords = np.zeros((self.height, self.width, 3), dtype=np.float32)
        face_ids = np.full((self.height, self.width), -1, dtype=np.int32)
        bary_coords = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Z-buffer
        z_buffer = np.full((self.height, self.width), np.inf, dtype=np.float32)

        # 光栅化每个面
        for face_idx, face in enumerate(self.faces):
            # 使用归一化的屏幕坐标进行光栅化
            screen_verts = self.screen_coords[face]  # [3, 3] in [0, 1]
            original_verts = self.original_vertices[face]  # [3, 3] 原始坐标

            # 光栅化这个三角形
            self._rasterize_triangle(
                screen_verts=screen_verts,
                original_verts=original_verts,
                face_idx=face_idx,
                pixel_coords=pixel_coords,
                face_ids=face_ids,
                bary_coords=bary_coords,
                z_buffer=z_buffer,
            )

        return pixel_coords, face_ids, bary_coords

    def _rasterize_triangle(
        self,
        screen_verts: np.ndarray,    # [3, 3] 归一化屏幕坐标
        original_verts: np.ndarray,  # [3, 3] 原始 3D 坐标
        face_idx: int,
        pixel_coords: np.ndarray,
        face_ids: np.ndarray,
        bary_coords: np.ndarray,
        z_buffer: np.ndarray,
    ):
        """光栅化单个三角形"""
        # 归一化坐标转像素坐标
        v0 = screen_verts[0]
        v1 = screen_verts[1]
        v2 = screen_verts[2]

        # 像素坐标
        x0 = int(v0[0] * (self.width - 1))
        y0 = int(v0[1] * (self.height - 1))
        x1 = int(v1[0] * (self.width - 1))
        y1 = int(v1[1] * (self.height - 1))
        x2 = int(v2[0] * (self.width - 1))
        y2 = int(v2[1] * (self.height - 1))

        # 计算 bbox（像素空间）
        min_x = max(0, min(x0, x1, x2))
        max_x = min(self.width - 1, max(x0, x1, x2))
        min_y = max(0, min(y0, y1, y2))
        max_y = min(self.height - 1, max(y0, y1, y2))

        # 遍历 bbox 内的像素
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # 计算重心坐标（使用归一化坐标）
                # 像素中心偏移 0.5
                px = (x + 0.5) / (self.width - 1)
                py = (y + 0.5) / (self.height - 1)

                bary = self._compute_barycentric_2d(
                    v0[:2], v1[:2], v2[:2],  # 只使用 XY 坐标
                    px, py
                )

                # 检查是否在三角形内
                if (bary >= 0).all() and (bary <= 1).all():
                    # 使用原始 3D 坐标计算 3D 位置
                    pos_3d = bary[0] * original_verts[0] + bary[1] * original_verts[1] + bary[2] * original_verts[2]

                    # Z-test（使用 Z 坐标）
                    z = pos_3d[2]
                    if z < z_buffer[y, x]:
                        z_buffer[y, x] = z
                        pixel_coords[y, x] = pos_3d
                        face_ids[y, x] = face_idx
                        bary_coords[y, x] = bary

    def _compute_barycentric_2d(
        self,
        v0: np.ndarray,  # [2]
        v1: np.ndarray,  # [2]
        v2: np.ndarray,  # [2]
        px: float,
        py: float,
    ) -> np.ndarray:
        """
        计算重心坐标（2D 版本）

        使用叉积方法
        """
        def cross_product_2d(a, b):
            return a[0] * b[1] - a[1] * b[0]

        p = np.array([px, py], dtype=np.float32)

        area_total = cross_product_2d(v1 - v0, v2 - v0)

        if abs(area_total) < 1e-8:
            # 退化三角形
            return np.array([0.0, 0.0, 0.0])

        w0 = cross_product_2d(v1 - p, v2 - p) / area_total
        w1 = cross_product_2d(v2 - p, v0 - p) / area_total
        w2 = cross_product_2d(v0 - p, v1 - p) / area_total

        return np.array([w0, w1, w2])


if __name__ == "__main__":
    # 测试光栅化器
    import sys

    if len(sys.argv) > 1:
        obj_path = sys.argv[1]
        texture_path = sys.argv[2]

        from ..data.obj_parser import parse_obj_file

        # 加载网格
        obj_data = parse_obj_file(obj_path)
        vertices = obj_data['vertices']
        faces = obj_data['faces']

        # 加载纹理
        texture = np.array(Image.open(texture_path))

        # 创建光栅化器
        renderer = OfflineRenderer(vertices, faces, texture)

        # 光栅化
        pixel_coords, face_ids, bary_coords = renderer.rasterize()

        print(f"光栅化完成: {np.sum(face_ids >= 0)} / {face_ids.size} 像素覆盖")
