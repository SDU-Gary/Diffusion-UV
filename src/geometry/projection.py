"""
最近点投影 - 将表面法线场扩展到空间邻域

实现空间点到网格表面的最近点投影，用于将表面法线场扩展到3D空间。
这是"影子投影"的核心实现，允许空间中的任意点继承其投影点的平滑法线。

核心概念：
- 狭窄邻域：网格表面附近的薄层区域
- 最近点映射：π(x) = argmin_{p∈S} ||x - p||
- 法线继承：n(x) = n_smooth(π(x))

优势：
- 避免中轴奇点
- 保持连续性
- 高效查询
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class ClosestPointProjection:
    """
    最近点投影器

    将3D空间点投影到网格表面，实现表面法线场到空间的扩展。
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, smooth_normals: np.ndarray):
        """
        初始化最近点投影器

        Args:
            vertices: [V, 3] 顶点坐标
            faces: [F, 3] 面索引
            smooth_normals: [V, 3] 平滑后的顶点法线
        """
        self.vertices = vertices.astype(np.float32)
        self.faces = faces.astype(np.int32)
        self.smooth_normals = smooth_normals.astype(np.float32)

        self.num_vertices = len(vertices)
        self.num_faces = len(faces)

        # 构建KD-Tree用于快速最近点查询
        self.vertex_kdtree = cKDTree(vertices)

        # 预计算面的几何信息（用于精确投影）
        self.face_bboxes = self._compute_face_bboxes()
        self.face_normals = self._compute_face_normals()

        logger.info(f"初始化最近点投影: {self.num_vertices}顶点, {self.num_faces}面")

    def _compute_face_bboxes(self) -> np.ndarray:
        """
        计算每个面的包围盒

        Returns:
            [F, 2, 3] 面包围盒 (min, max)
        """
        face_bboxes = np.zeros((self.num_faces, 2, 3), dtype=np.float32)

        for i, face in enumerate(self.faces):
            face_verts = self.vertices[face]
            face_bboxes[i, 0] = np.min(face_verts, axis=0)  # min
            face_bboxes[i, 1] = np.max(face_verts, axis=0)  # max

        return face_bboxes

    def _compute_face_normals(self) -> np.ndarray:
        """
        计算面法线

        Returns:
            [F, 3] 面法线
        """
        face_normals = np.zeros((self.num_faces, 3), dtype=np.float32)

        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            face_normals[i] = normal / (np.linalg.norm(normal) + 1e-10)

        return face_normals

    def project_to_surface(
        self,
        points: np.ndarray,
        max_distance: float = 0.1,
        use_vertex_projection: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        将空间点投影到网格表面

        Args:
            points: [N, 3] 空间点坐标
            max_distance: 最大投影距离（超出此距离的点无效）
            use_vertex_projection: 是否使用顶点投影（快速）vs 面投影（精确）

        Returns:
            projected_points: [N, 3] 投影后的表面点
            face_ids: [N] 投影到的面ID (-1表示无效)
            barycentric_coords: [N, 3] 投影点的重心坐标
            vertex_indices: [N] 投影到的顶点索引 (-1表示无效)
        """
        num_points = len(points)
        projected_points = np.zeros_like(points, dtype=np.float32)
        face_ids = np.full(num_points, -1, dtype=np.int32)
        barycentric_coords = np.zeros((num_points, 3), dtype=np.float32)
        vertex_indices = np.full(num_points, -1, dtype=np.int32)

        if use_vertex_projection:
            # 快速顶点投影
            distances, vert_indices = self.vertex_kdtree.query(points)

            # 过滤超出最大距离的点
            valid_mask = distances < max_distance

            if np.any(valid_mask):
                projected_points[valid_mask] = self.vertices[vert_indices[valid_mask]]
                face_ids[valid_mask] = 0  # 顶点投影标记为0
                barycentric_coords[valid_mask] = np.array([1.0, 0.0, 0.0])
                vertex_indices[valid_mask] = vert_indices[valid_mask]

            invalid_count = np.sum(~valid_mask)
            if invalid_count > 0:
                logger.warning(f"{invalid_count}/{num_points} 点超出投影距离")

        else:
            # 精确面投影（迭代最近点）
            for i, point in enumerate(points):
                proj_point, face_id, bary_coords = self._project_point_to_face(point, max_distance)

                if face_id >= 0:
                    projected_points[i] = proj_point
                    face_ids[i] = face_id
                    barycentric_coords[i] = bary_coords
                else:
                    logger.debug(f"点 {i} 投影失败，超出距离")

        return projected_points, face_ids, barycentric_coords, vertex_indices

    def _project_point_to_face(
        self,
        point: np.ndarray,
        max_distance: float
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        将单个点精确投影到最近的面

        Args:
            point: [3] 点坐标
            max_distance: 最大投影距离

        Returns:
            projected_point: [3] 投影点
            face_id: int 面ID (-1表示失败)
            barycentric: [3] 重心坐标
        """
        # 1. 找到最近的顶点
        dist, vertex_idx = self.vertex_kdtree.query(point)

        if dist > max_distance:
            return point, -1, np.zeros(3)

        # 2. 找到包含该顶点的所有面
        incident_faces = []
        for face_id, face in enumerate(self.faces):
            if vertex_idx in face:
                incident_faces.append(face_id)

        # 3. 在这些面中找最近点
        best_distance = float('inf')
        best_face_id = -1
        best_barycentric = np.zeros(3)
        best_projected = point.copy()

        for face_id in incident_faces:
            face = self.faces[face_id]
            face_verts = self.vertices[face]

            # 计算点到面的投影
            projected, barycentric = self._point_to_triangle_projection(point, face_verts)

            distance = np.linalg.norm(projected - point)

            if distance < best_distance:
                best_distance = distance
                best_face_id = face_id
                best_barycentric = barycentric
                best_projected = projected

        if best_distance > max_distance:
            return point, -1, np.zeros(3)

        return best_projected, best_face_id, best_barycentric

    def _point_to_triangle_projection(
        self,
        point: np.ndarray,
        triangle_verts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算点到三角面的投影

        Args:
            point: [3] 点坐标
            triangle_verts: [3, 3] 三角面顶点

        Returns:
            projected_point: [3] 投影点
            barycentric: [3] 重心坐标
        """
        v0, v1, v2 = triangle_verts

        # 构建面的局部坐标系
        edge0 = v1 - v0
        edge1 = v2 - v0

        # 面法线
        normal = np.cross(edge0, edge1)
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        # 点到面的投影距离
        point_to_plane = point - v0
        distance = np.dot(point_to_plane, normal)

        # 投影点
        projected = point - distance * normal

        # 计算重心坐标
        # 解方程: P = v0 + u * edge0 + v * edge1
        # 即: P - v0 = u * edge0 + v * edge1
        # 写成矩阵形式: [edge0, edge1] * [u, v]^T = P - v0

        diff = projected - v0

        # 构建2x2线性系统
        mat = np.array([
            [np.dot(edge0, edge0), np.dot(edge0, edge1)],
            [np.dot(edge1, edge0), np.dot(edge1, edge1)]
        ])

        rhs = np.array([np.dot(diff, edge0), np.dot(diff, edge1)])

        try:
            solution = np.linalg.solve(mat, rhs)
            u, v = solution

            # 重心坐标
            w = 1.0 - u - v
            barycentric = np.array([w, u, v])

            # 检查是否在三角形内
            if np.all(barycentric >= 0) and np.all(barycentric <= 1.0):
                return projected, barycentric
            else:
                # 投影点在三角形外，找到最近的边或顶点
                return self._project_to_triangle_edge(point, triangle_verts)

        except np.linalg.LinAlgError:
            # 退化三角形，返回最近顶点
            distances = np.linalg.norm(triangle_verts - point, axis=1)
            nearest_idx = np.argmin(distances)
            barycentric = np.zeros(3)
            barycentric[nearest_idx] = 1.0
            return triangle_verts[nearest_idx], barycentric

    def _project_to_triangle_edge(
        self,
        point: np.ndarray,
        triangle_verts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点到三角形的最近边或顶点

        Args:
            point: [3] 点坐标
            triangle_verts: [3, 3] 三角面顶点

        Returns:
            projected_point: [3] 投影点
            barycentric: [3] 重心坐标
        """
        v0, v1, v2 = triangle_verts

        # 三条边
        edges = [
            (v0, v1, [0, 1]),
            (v1, v2, [1, 2]),
            (v2, v0, [2, 0])
        ]

        best_distance = float('inf')
        best_projected = point.copy()
        best_barycentric = np.zeros(3)

        # 检查每条边
        for va, vb, indices in edges:
            projected, bary = self._project_to_line_segment(point, va, vb)
            distance = np.linalg.norm(projected - point)

            if distance < best_distance:
                best_distance = distance
                best_projected = projected

                # 构建重心坐标
                best_barycentric = np.zeros(3)
                best_barycentric[indices[0]] = bary[0]
                best_barycentric[indices[1]] = bary[1]

        # 检查顶点
        for i, vert in enumerate(triangle_verts):
            distance = np.linalg.norm(vert - point)
            if distance < best_distance:
                best_distance = distance
                best_projected = vert
                best_barycentric = np.zeros(3)
                best_barycentric[i] = 1.0

        return best_projected, best_barycentric

    def _project_to_line_segment(
        self,
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点到线段

        Args:
            point: [3] 点坐标
            line_start: [3] 线段起点
            line_end: [3] 线段终点

        Returns:
            projected_point: [3] 投影点
            barycentric: [2] 重心坐标
        """
        edge = line_end - line_start
        edge_length_sq = np.dot(edge, edge)

        if edge_length_sq < 1e-10:
            # 退化线段
            return line_start, np.array([1.0, 0.0])

        # 投影参数 t
        t = np.dot(point - line_start, edge) / edge_length_sq

        # 限制在线段内
        t = np.clip(t, 0.0, 1.0)

        # 投影点
        projected = line_start + t * edge

        # 重心坐标
        barycentric = np.array([1.0 - t, t])

        return projected, barycentric

    def interpolate_normals(
        self,
        barycentric_coords: np.ndarray,
        face_ids: np.ndarray,
        vertex_indices: np.ndarray = None,
        use_angle_aware_interpolation: bool = True
    ) -> np.ndarray:
        """
        插值表面法线到投影点（支持角度感知插值）

        Args:
            barycentric_coords: [N, 3] 重心坐标
            face_ids: [N] 面ID
            vertex_indices: [N] 顶点索引（用于顶点投影）
            use_angle_aware_interpolation: 是否使用角度感知插值

        Returns:
            [N, 3] 插值后的法线
        """
        num_points = len(barycentric_coords)
        normals = np.zeros((num_points, 3), dtype=np.float32)

        for i in range(num_points):
            face_id = face_ids[i]

            if face_id < 0:
                # 无效投影，使用零法线
                normals[i] = np.zeros(3)
                continue

            # 检查是否为顶点投影（face_id == 0 且 barycentric 为 [1,0,0]）
            if vertex_indices is not None and face_id == 0 and \
               np.allclose(barycentric_coords[i], [1.0, 0.0, 0.0]):
                # 顶点投影：直接使用该顶点的法线
                vertex_idx = vertex_indices[i]
                if vertex_idx >= 0 and vertex_idx < len(self.smooth_normals):
                    normals[i] = self.smooth_normals[vertex_idx]
                else:
                    normals[i] = np.zeros(3)
                continue

            # 面投影：使用重心坐标插值
            face = self.faces[face_id]
            bary = barycentric_coords[i]

            # 获取面的法线
            face_normals = self.smooth_normals[face]

            if use_angle_aware_interpolation:
                # 角度感知插值：考虑法线之间的角度
                # 使用SLERP（球面线性插值）的简化版本
                interpolated_normal = self._angle_aware_normal_interpolation(
                    face_normals, bary
                )
            else:
                # 标准线性插值
                interpolated_normal = bary[0] * face_normals[0] + \
                                     bary[1] * face_normals[1] + \
                                     bary[2] * face_normals[2]

            # 归一化
            normal_norm = np.linalg.norm(interpolated_normal)
            if normal_norm > 1e-10:
                normals[i] = interpolated_normal / normal_norm
            else:
                normals[i] = face_normals[0]  # fallback

        return normals

    def _angle_aware_normal_interpolation(
        self,
        normals: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        角度感知的法线插值

        使用球面线性插值的思想，考虑法线方向之间的角度，
        避免在对立法线之间插值时产生不自然的结果。

        Args:
            normals: [3, 3] 三个顶点的法线
            weights: [3] 重心坐标权重

        Returns:
            [3] 插值后的法线
        """
        # 确保所有法线都是单位向量
        unit_normals = []
        for n in normals:
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                unit_normals.append(n / n_norm)
            else:
                unit_normals.append(np.array([0.0, 0.0, 1.0]))

        # 计算参考法线（加权平均）
        reference_normal = weights[0] * unit_normals[0] + \
                           weights[1] * unit_normals[1] + \
                           weights[2] * unit_normals[2]

        ref_norm = np.linalg.norm(reference_normal)
        if ref_norm > 1e-10:
            reference_normal = reference_normal / ref_norm
        else:
            reference_normal = unit_normals[0]

        # 角度加权插值
        result_normal = np.zeros(3, dtype=np.float32)
        total_weight = 0.0

        for i in range(3):
            if weights[i] < 1e-10:
                continue

            # 计算法线角度
            dot_product = np.dot(unit_normals[i], reference_normal)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            # 角度越大，权重越小（避免对立法线的影响）
            angle_weight = weights[i] * (1.0 - angle / np.pi)

            result_normal += angle_weight * unit_normals[i]
            total_weight += angle_weight

        # 归一化
        if total_weight > 1e-10:
            result_normal = result_normal / total_weight

        return result_normal

    def compute_spatial_normal_field(
        self,
        points: np.ndarray,
        max_distance: float = 0.1,
        use_vertex_projection: bool = False,
        apply_smoothing: bool = True,
        smoothing_iterations: int = 2
    ) -> Dict[str, np.ndarray]:
        """
        计算空间点的法线场（优化版）

        这是完整的"影子投影"流程：
        1. 将空间点投影到表面
        2. 获取投影点的平滑法线
        3. 插值法线到原始点
        4. 应用法线场平滑（可选）

        Args:
            points: [N, 3] 空间点坐标
            max_distance: 最大投影距离
            use_vertex_projection: 是否使用快速顶点投影（默认False，使用精确面投影）
            apply_smoothing: 是否应用法线场平滑
            smoothing_iterations: 平滑迭代次数

        Returns:
            dict: {
                'points': [N, 3] 原始点
                'normals': [N, 3] 法线场
                'projected_points': [N, 3] 投影点
                'face_ids': [N] 投影面ID
                'barycentric_coords': [N, 3] 重心坐标
                'vertex_indices': [N] 顶点索引
            }
        """
        logger.info(f"计算空间法线场: {len(points)} 点 (精确投影: {not use_vertex_projection})")

        # 1. 投影到表面
        projected_points, face_ids, barycentric_coords, vertex_indices = \
            self.project_to_surface(points, max_distance, use_vertex_projection)

        # 2. 插值法线（使用角度感知插值）
        normals = self.interpolate_normals(
            barycentric_coords, face_ids, vertex_indices,
            use_angle_aware_interpolation=True
        )

        # 3. 应用法线场平滑（可选）
        if apply_smoothing and len(points) > 1:
            normals = self._smooth_normal_field(normals, points, smoothing_iterations)
            logger.info(f"应用法线场平滑: {smoothing_iterations} 迭代")

        return {
            'points': points,
            'normals': normals,
            'projected_points': projected_points,
            'face_ids': face_ids,
            'barycentric_coords': barycentric_coords,
            'vertex_indices': vertex_indices,
        }

    def _smooth_normal_field(
        self,
        normals: np.ndarray,
        points: np.ndarray,
        iterations: int = 2
    ) -> np.ndarray:
        """
        对法线场应用平滑处理，减少相邻点之间的突变

        使用基于距离的加权平均，在保持局部特征的同时提高连续性。

        Args:
            normals: [N, 3] 原始法线
            points: [N, 3] 对应的空间点
            iterations: 平滑迭代次数

        Returns:
            [N, 3] 平滑后的法线
        """
        smoothed_normals = normals.copy()

        for iteration in range(iterations):
            # 对每个点，考虑其k近邻进行平滑
            k = min(10, len(points))  # 使用10个最近邻

            # 构建KD树用于快速最近邻查询
            kdtree = cKDTree(points)

            new_normals = np.zeros_like(smoothed_normals)

            for i in range(len(points)):
                # 查找k近邻
                distances, indices = kdtree.query(points[i], k=k)

                # 距离加权（距离越近权重越大）
                weights = 1.0 / (distances + 1e-10)
                weights = weights / np.sum(weights)  # 归一化

                # 加权平均法线
                neighbor_normals = smoothed_normals[indices]
                weighted_normal = np.sum(neighbor_normals * weights[:, np.newaxis], axis=0)

                # 归一化
                normal_norm = np.linalg.norm(weighted_normal)
                if normal_norm > 1e-10:
                    new_normals[i] = weighted_normal / normal_norm
                else:
                    new_normals[i] = smoothed_normals[i]

            smoothed_normals = new_normals

        return smoothed_normals

    def evaluate_projection_quality(self, test_points: np.ndarray, spatial_result: Dict = None) -> Dict[str, float]:
        """
        评估投影质量（改进版）

        Args:
            test_points: [N, 3] 测试点
            spatial_result: 可选，已计算的空间法线场结果

        Returns:
            dict: 质量指标
        """
        logger.info("评估投影质量...")

        # 如果没有提供spatial_result，则计算
        if spatial_result is None:
            spatial_result = self.compute_spatial_normal_field(test_points)

        # 1. 投影成功率
        success_rate = np.sum(spatial_result['face_ids'] >= 0) / len(test_points)

        # 2. 平均投影距离
        valid_mask = spatial_result['face_ids'] >= 0
        if np.any(valid_mask):
            projection_distances = np.linalg.norm(
                spatial_result['projected_points'][valid_mask] - test_points[valid_mask],
                axis=1
            )
            avg_projection_distance = np.mean(projection_distances)
            max_projection_distance = np.max(projection_distances)
        else:
            avg_projection_distance = float('inf')
            max_projection_distance = float('inf')

        # 3. 法线连续性（改进：基于空间距离的K近邻）
        valid_normals = spatial_result['normals'][valid_mask]
        valid_points = test_points[valid_mask]

        if len(valid_normals) > 1:
            # 使用KD树找到空间上的近邻
            from scipy.spatial import cKDTree
            kdtree = cKDTree(valid_points)

            # 对每个点，找到其k个最近邻
            k = min(5, len(valid_normals) - 1)
            normal_differences = []

            for i in range(len(valid_normals)):
                # 找到k个最近邻（排除自己）
                distances, indices = kdtree.query(valid_points[i], k=k+1)
                neighbors = indices[1:]  # 排除自己

                # 计算与近邻的法线差异
                for neighbor_idx in neighbors:
                    if neighbor_idx < len(valid_normals):
                        diff = np.linalg.norm(valid_normals[i] - valid_normals[neighbor_idx])
                        normal_differences.append(diff)

            avg_normal_difference = np.mean(normal_differences) if normal_differences else 0.0
        else:
            avg_normal_difference = 0.0

        quality_metrics = {
            'projection_success_rate': success_rate,
            'avg_projection_distance': avg_projection_distance,
            'max_projection_distance': max_projection_distance,
            'avg_normal_difference': avg_normal_difference,
            'continuity_score': 1.0 / (1.0 + avg_normal_difference),
        }

        logger.info(f"投影质量评估: {quality_metrics}")

        return quality_metrics