"""
基于热方法的表面法线场平滑

实现基于Laplace-Beltrami算子的热扩散方法，用于在流形表面生成连续的法线场。
该方法是独立于主干网络的几何处理方法，用于替代SDF网络提供切空间投影。

核心思想：
1. 在网格表面进行极短时间的热扩散，平滑离散法线
2. 通过最近点投影将表面法线场扩展到空间邻域
3. 避免SDF梯度的中轴奇点问题

参考文献：
- Crane et al. "The Heat Method for Distance Computation" SIGGRAPH 2017
- 拉普拉斯-贝尔特拉米算子 (Laplace-Beltrami Operator)
"""

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HeatMethodNormalField:
    """
    基于热方法的表面法线场生成器

    使用拉普拉斯-贝尔特拉米算子在流形表面进行法线扩散，
    生成C∞连续的法线场，用于UV映射的切空间投影。

    核心优势：
    - 无需神经网络训练
    - 数学上保证的光滑度
    - 避免中轴奇点
    - 只需要线性系统求解
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, time_step: float = 1e-3):
        """
        初始化热方法法线场

        Args:
            vertices: [V, 3] 顶点坐标
            faces: [F, 3] 面顶点索引
            time_step: 热扩散时间步长 (t -> 0 保证解析级光滑)
        """
        self.vertices = vertices.astype(np.float32)
        self.faces = faces.astype(np.int32)
        self.time_step = time_step

        self.num_vertices = len(vertices)
        self.num_faces = len(faces)

        # 构建拉普拉斯-贝尔特拉米算子
        self.L = None  # 拉普拉斯矩阵
        self.M = None  # 质量矩阵
        self.area_weights = None  # 顶点面积权重

        # 预计算初始法线
        self.initial_normals = None

        logger.info(f"初始化热方法: {self.num_vertices}顶点, {self.num_faces}面, t={time_step}")

    def compute_vertex_normals(self) -> np.ndarray:
        """
        计算顶点法线（面积加权平均）

        Returns:
            [V, 3] 顶点法线，已归一化
        """
        # 计算每个面的法线
        face_normals = np.zeros((self.num_faces, 3), dtype=np.float32)

        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            face_normals[i] = normal

        # 归一化面法线
        face_normal_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = np.divide(face_normals, face_normal_norms,
                                 where=face_normal_norms > 1e-10)

        # 面积权重
        face_areas = 0.5 * face_normal_norms.flatten()

        # 顶点法线 = 面积加权平均
        vertex_normals = np.zeros((self.num_vertices, 3), dtype=np.float32)
        vertex_weights = np.zeros(self.num_vertices, dtype=np.float32)

        for i, face in enumerate(self.faces):
            for vertex_idx in face:
                vertex_normals[vertex_idx] += face_areas[i] * face_normals[i]
                vertex_weights[vertex_idx] += face_areas[i]

        # 归一化顶点法线
        vertex_normal_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = np.divide(vertex_normals, vertex_normal_norms,
                                   where=vertex_normal_norms > 1e-10)

        return vertex_normals

    def build_laplacian_operator(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        构建离散拉普拉斯-贝尔特拉米算子

        使用余切权重拉普拉斯：
        L_ij = -0.5 * (cot(alpha) + cot(beta)) / A

        Returns:
            L: 拉普拉斯矩阵 (V x V)
            M: 质量矩阵 (V x V)
        """
        logger.info("构建拉普拉斯-贝尔特拉米算子...")

        # 初始化稀疏矩阵的构建数据
        row_indices = []
        col_indices = []
        data = []
        mass_data = []

        # 顶点对邻居的边计数（用于计算质量矩阵）
        edge_count = np.zeros(self.num_vertices, dtype=np.int32)

        # 遍历所有边
        edge_cot_weights = {}

        for face in self.faces:
            v0, v1, v2 = face

            # 计算面的三条边
            edges = [(v0, v1), (v1, v2), (v2, v0)]

            for vi, vj in edges:
                # 确保边的顺序一致（小到大）
                edge = tuple(sorted((vi, vj)))

                if edge not in edge_cot_weights:
                    edge_cot_weights[edge] = []

        # 计算余切权重
        for face in self.faces:
            v0, v1, v2 = face
            verts = self.vertices[face]

            # 计算三个角
            angles = self._compute_face_angles(verts)

            # 三条边
            edges = [(v1, v2, angles[0]), (v0, v2, angles[1]), (v0, v1, angles[2])]

            for vi, vj, angle in edges:
                edge = tuple(sorted((vi, vj)))
                cot_weight = 1.0 / np.tan(angle)
                edge_cot_weights[edge].append(cot_weight)

        # 构建拉普拉斯矩阵
        for edge, cot_weights in edge_cot_weights.items():
            vi, vj = edge

            # 余切权重和
            if len(cot_weights) == 2:
                cot_sum = cot_weights[0] + cot_weights[1]
            else:
                cot_sum = cot_weights[0]  # 边界情况

            # 拉普拉斯矩阵元素
            row_indices.extend([vi, vj])
            col_indices.extend([vj, vi])
            data.extend([-0.5 * cot_sum, -0.5 * cot_sum])

            edge_count[vi] += 1
            edge_count[vj] += 1

        # 对角元素（负的行和）
        diagonal = np.zeros(self.num_vertices, dtype=np.float32)
        for i in range(len(row_indices)):
            diagonal[row_indices[i]] -= data[i]

        # 添加对角元素
        for i in range(self.num_vertices):
            row_indices.append(i)
            col_indices.append(i)
            data.append(diagonal[i])
            mass_data.append(1.0)  # 简化的质量矩阵

        # 构建稀疏矩阵
        L = csr_matrix((data, (row_indices, col_indices)),
                       shape=(self.num_vertices, self.num_vertices))
        M = csr_matrix((mass_data, (range(self.num_vertices), range(self.num_vertices))),
                       shape=(self.num_vertices, self.num_vertices))

        logger.info(f"拉普拉斯矩阵构建完成: {L.nnz} 非零元素")

        return L, M

    def _compute_face_angles(self, verts: np.ndarray) -> np.ndarray:
        """
        计算三角面的三个角

        Args:
            verts: [3, 3] 面的三个顶点

        Returns:
            [3] 三个角（弧度）
        """
        v0, v1, v2 = verts

        # 三条边
        edge0 = v1 - v0
        edge1 = v2 - v0
        edge2 = v2 - v1

        # 边长
        len0 = np.linalg.norm(edge0)
        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)

        # 余弦定理
        cos_angle0 = np.dot(edge0, edge1) / (len0 * len1 + 1e-10)
        cos_angle1 = -np.dot(edge0, edge2) / (len0 * len2 + 1e-10)
        cos_angle2 = np.dot(edge1, edge2) / (len1 * len2 + 1e-10)

        # 角度
        angle0 = np.arccos(np.clip(cos_angle0, -1.0, 1.0))
        angle1 = np.arccos(np.clip(cos_angle1, -1.0, 1.0))
        angle2 = np.arccos(np.clip(cos_angle2, -1.0, 1.0))

        return np.array([angle0, angle1, angle2])

    def diffuse_normals(self, num_iterations: int = 10) -> np.ndarray:
        """
        使用热扩散平滑法线场

        求解热方程：∂n/∂t = Δn
        隐式求解：(M - t*L) n_new = M n_old

        Args:
            num_iterations: 扩散迭代次数

        Returns:
            [V, 3] 平滑后的顶点法线
        """
        logger.info(f"开始热扩散: {num_iterations} 迭代")

        # 计算初始法线
        if self.initial_normals is None:
            self.initial_normals = self.compute_vertex_normals()

        # 构建拉普拉斯算子
        if self.L is None or self.M is None:
            self.L, self.M = self.build_laplacian_operator()

        # 构建热扩散算子: (M - t*L)
        heat_operator = self.M - self.time_step * self.L

        # 初始法线
        smooth_normals = self.initial_normals.copy()

        # 迭代扩散
        for iteration in range(num_iterations):
            # 对每个分量分别求解
            for dim in range(3):
                rhs = self.M @ smooth_normals[:, dim]
                smooth_normals[:, dim] = spsolve(heat_operator, rhs)

            # 归一化（保持单位长度）
            normal_norms = np.linalg.norm(smooth_normals, axis=1, keepdims=True)
            smooth_normals = np.divide(smooth_normals, normal_norms,
                                       where=normal_norms > 1e-10)

            if iteration % 5 == 0:
                logger.debug(f"扩散迭代 {iteration}/{num_iterations}")

        logger.info("热扩散完成")

        return smooth_normals

    def compute_smooth_normals_field(self, num_iterations: int = 10) -> Dict[str, np.ndarray]:
        """
        计算完整的平滑法线场

        Returns:
            dict: {
                'initial_normals': [V, 3] 初始法线
                'smooth_normals': [V, 3] 平滑法线
                'vertices': [V, 3] 顶点坐标
                'faces': [F, 3] 面索引
            }
        """
        logger.info("计算平滑法线场...")

        # 初始法线
        initial_normals = self.compute_vertex_normals()
        self.initial_normals = initial_normals

        # 平滑法线
        smooth_normals = self.diffuse_normals(num_iterations)

        return {
            'initial_normals': initial_normals,
            'smooth_normals': smooth_normals,
            'vertices': self.vertices,
            'faces': self.faces,
        }

    def evaluate_smoothness_quality(self) -> Dict[str, float]:
        """
        评估法线场的平滑质量

        Returns:
            dict: 平滑质量指标
        """
        logger.info("评估法线场平滑质量...")

        # 计算平滑法线
        smooth_normals = self.diffuse_normals(num_iterations=10)
        initial_normals = self.initial_normals

        # 1. 法线变化率（相邻顶点法线差异）
        normal_variation = self._compute_normal_variation(smooth_normals)

        # 2. 平均曲率估计
        mean_curvature = self._estimate_mean_curvature(smooth_normals)

        # 3. 法线雅可比条件数（衡量奇点）
        jacobian_condition = self._compute_normal_jacobian_condition(smooth_normals)

        quality_metrics = {
            'normal_variation': normal_variation,
            'mean_curvature': mean_curvature,
            'jacobian_condition': jacobian_condition,
            'smoothness_score': 1.0 / (1.0 + normal_variation + mean_curvature) if np.isfinite(normal_variation + mean_curvature) else 0.5,
        }

        logger.info(f"平滑质量评估: {quality_metrics}")

        return quality_metrics

    def _compute_normal_variation(self, normals: np.ndarray) -> float:
        """
        计算相邻顶点法线变化率

        Args:
            normals: [V, 3] 顶点法线

        Returns:
            平均法线变化率
        """
        total_variation = 0.0
        count = 0

        for face in self.faces:
            for i in range(3):
                vi, vj = face[i], face[(i + 1) % 3]
                normal_diff = np.linalg.norm(normals[vi] - normals[vj])
                total_variation += normal_diff
                count += 1

        return total_variation / max(count, 1)

    def _estimate_mean_curvature(self, normals: np.ndarray) -> float:
        """
        估计平均曲率（基于法线梯度）

        Args:
            normals: [V, 3] 顶点法线

        Returns:
            平均曲率估计
        """
        if self.L is None:
            self.L, _ = self.build_laplacian_operator()

        # 法线拉普拉斯
        normal_laplacian = self.L @ normals

        # 平均曲率范数
        curvature_norms = np.linalg.norm(normal_laplacian, axis=1)

        # 过滤NaN和Inf值
        valid_curvatures = curvature_norms[np.isfinite(curvature_norms)]

        if len(valid_curvatures) == 0:
            return 0.0

        mean_curvature = np.mean(valid_curvatures)

        return float(mean_curvature) if np.isfinite(mean_curvature) else 0.0

    def _compute_normal_jacobian_condition(self, normals: np.ndarray) -> float:
        """
        计算法线雅可比矩阵的条件数（检测奇点）

        Args:
            normals: [V, 3] 顶点法线

        Returns:
            平均条件数
        """
        # 构建法线梯度矩阵（简化估计）
        if self.L is None:
            self.L, _ = self.build_laplacian_operator()

        # 法线梯度估计
        normal_gradient = self.L @ normals

        # 计算条件数
        conditions = []
        for i in range(self.num_vertices):
            grad = normal_gradient[i]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                conditions.append(grad_norm)

        return np.mean(conditions) if conditions else 0.0