"""
ALSFD (Anisotropic Level Set Frame Diffusion) 修复版本

修复的问题：
1. 初始向量场应该是切向的，不是法向的
2. Laplacian算子构建错误
3. 过度投影导致数值衰减
4. 数值稳定性改进

核心理论：
1. 符号距离函数 f: |∇f| = 1
2. 单位法向量 n = ∇f
3. 投影矩阵 P = I - nn^T
4. 退化扩散算子 L(v) = P(∇·(P∇v))
5. 演化方程: ∂v/∂t = L(v)

定理：对于切向量场，空间退化扩散等价于曲面Bochner热流
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.sparse import csr_matrix, eye, coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class ALSFDVectorFieldDiffusionFixed:
    """
    ALSFD向量场扩散器（修复版）

    修复版本解决了原始实现中的严重数值问题：
    1. 正确构建切向Laplacian算子
    2. 使用真实的切向量场作为初始条件
    3. 避免过度投影
    4. 改进数值稳定性
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, sdf_values: Optional[np.ndarray] = None):
        """
        初始化ALSFD扩散器

        Args:
            vertices: [V, 3] 顶点坐标
            faces: [F, 3] 面索引
            sdf_values: [V] 可选的符号距离函数值
        """
        self.vertices = vertices.astype(np.float64)  # 提高精度
        self.faces = faces.astype(np.int32)
        self.num_vertices = len(vertices)

        # 如果没有提供SDF，则计算近似SDF
        if sdf_values is None:
            self.sdf_values = self._compute_approximate_sdf()
        else:
            self.sdf_values = sdf_values.astype(np.float64)

        # 计算单位法向量（梯度）
        self.normals = self._compute_sdf_normals()

        # 计算投影矩阵
        self.projection_matrices = self._compute_projection_matrices()

        logger.info(f"初始化ALSFD: {self.num_vertices}顶点")
        logger.info(f"SDF范围: [{self.sdf_values.min():.3f}, {self.sdf_values.max():.3f}]")

    def _compute_approximate_sdf(self) -> np.ndarray:
        """
        计算近似符号距离函数

        使用顶点法线作为SDF梯度的近似，
        SDF值通过到平均平面的距离计算

        Returns:
            [V] SDF值
        """
        # 计算每个顶点到网格平均平面的有向距离
        center = np.mean(self.vertices, axis=0)

        # 计算顶点法线
        vertex_normals = self._compute_vertex_normals()

        # SDF值 = (顶点 - 中心) · 法线
        sdf_values = np.sum((self.vertices - center) * vertex_normals, axis=1)

        return sdf_values

    def _compute_vertex_normals(self) -> np.ndarray:
        """
        计算顶点法线

        Returns:
            [V, 3] 单位法向量
        """
        # 计算面法线
        face_normals = np.zeros((len(self.faces), 3), dtype=np.float64)
        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            face_normals[i] = normal / (np.linalg.norm(normal) + 1e-10)

        # 面积权重
        face_areas = np.zeros(len(self.faces), dtype=np.float64)
        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_areas[i] = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

        # 顶点法线（面积加权平均）
        vertex_normals = np.zeros_like(self.vertices)
        vertex_weights = np.zeros(self.num_vertices, dtype=np.float64)

        for i, face in enumerate(self.faces):
            for j, vertex_idx in enumerate(face):
                vertex_normals[vertex_idx] += face_areas[i] * face_normals[i]
                vertex_weights[vertex_idx] += face_areas[i]

        # 归一化
        vertex_normal_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = np.divide(vertex_normals, vertex_normal_norms,
                                   where=vertex_normal_norms > 1e-10)

        return vertex_normals

    def _compute_sdf_normals(self) -> np.ndarray:
        """
        计算SDF的梯度（单位法向量）

        Returns:
            [V, 3] 单位法向量 n = ∇f
        """
        # 对于网格表面，SDF梯度就是表面法线
        return self._compute_vertex_normals()

    def _compute_projection_matrices(self) -> np.ndarray:
        """
        计算投影矩阵 P = I - nn^T

        Returns:
            [V, 3, 3] 投影矩阵
        """
        projection_matrices = np.zeros((self.num_vertices, 3, 3), dtype=np.float64)

        for i in range(self.num_vertices):
            n = self.normals[i]
            # P = I - nn^T
            P = np.eye(3, dtype=np.float64) - np.outer(n, n)
            projection_matrices[i] = P

        return projection_matrices

    def generate_tangent_vector_field(self) -> np.ndarray:
        """
        生成真实的切向量场（修复版的关键）

        生成一个在切空间内的随机向量场，作为ALSFD的初始条件
        这修复了原始版本使用法线作为初始条件的错误

        Returns:
            [V, 3] 切向量场
        """
        logger.info("生成切向量场...")

        # 生成随机向量
        np.random.seed(42)  # 可重复性
        random_vectors = np.random.randn(self.num_vertices, 3).astype(np.float64)

        # 投影到切空间
        tangent_vectors = np.zeros_like(random_vectors)
        for i in range(self.num_vertices):
            P = self.projection_matrices[i]
            tangent_vectors[i] = P @ random_vectors[i]

        # 归一化
        tangent_magnitudes = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
        tangent_vectors = np.divide(tangent_vectors, tangent_magnitudes,
                                   where=tangent_magnitudes > 1e-10)

        # 验证切向性
        avg_tangent = np.mean(tangent_magnitudes)
        logger.info(f"生成切向量场: 平均模长 {avg_tangent:.6f}")

        return tangent_vectors

    def compute_tangent_frame_field(self, initial_vectors: np.ndarray) -> np.ndarray:
        """
        计算切标架场（确保切向性）

        Args:
            initial_vectors: [V, 3] 初始向量场

        Returns:
            [V, 3] 切向投影后的向量场
        """
        tangent_vectors = np.zeros_like(initial_vectors)

        for i in range(self.num_vertices):
            P = self.projection_matrices[i]
            # 投影到切空间：v_tangent = P * v
            tangent_vectors[i] = P @ initial_vectors[i]

        return tangent_vectors

    def build_mesh_laplacian(self) -> csr_matrix:
        """
        构建网格Laplacian算子（使用余切权重）

        Returns:
            [V, V] 稀疏矩阵
        """
        logger.info("构建余切权重Laplacian...")

        # 构建邻接关系和余切权重
        from collections import defaultdict
        edge_weights = defaultdict(float)
        vertex_weights = np.zeros(self.num_vertices, dtype=np.float64)

        for face in self.faces:
            v0, v1, v2 = face

            # 顶点坐标
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]

            # 计算边向量
            e01 = p1 - p0
            e12 = p2 - p1
            e20 = p0 - p2

            # 计算角度（余切权重）
            def cotangent(edge1, edge2):
                """计算余切值"""
                cos_theta = np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2) + 1e-10)
                sin_theta = np.sqrt(1 - min(cos_theta**2, 1.0))
                return cos_theta / (sin_theta + 1e-10)

            # 计算三个角的余切
            cot_alpha = cotangent(-e20, e01)   # 在v0处的角
            cot_beta = cotangent(-e01, e12)    # 在v1处的角
            cot_gamma = cotangent(-e12, e20)   # 在v2处的角

            # 添加边权重
            edges = [(v0, v1, cot_gamma), (v1, v2, cot_alpha), (v2, v0, cot_beta)]
            for vi, vj, cot_weight in edges:
                edge = tuple(sorted((vi, vj)))
                edge_weights[edge] += cot_weight
                vertex_weights[vi] += cot_weight
                vertex_weights[vj] += cot_weight

        # 构建稀疏矩阵
        row_indices = []
        col_indices = []
        data = []

        for edge, weight in edge_weights.items():
            vi, vj = edge
            # 非对角元素
            row_indices.extend([vi, vj])
            col_indices.extend([vj, vi])
            data.extend([-weight, -weight])

        # 对角元素
        for i in range(self.num_vertices):
            row_indices.append(i)
            col_indices.append(i)
            data.append(vertex_weights[i])

        L = csr_matrix((data, (row_indices, col_indices)),
                       shape=(self.num_vertices, self.num_vertices))

        logger.info(f"Laplacian构建完成: {L.nnz} 非零元素")

        return L

    def build_alsfd_operator(self) -> csr_matrix:
        """
        构建ALSFD退化扩散算子

        修复版：使用正确的切向Laplacian

        Returns:
            [3V, 3V] 稀疏矩阵
        """
        logger.info("构建ALSFD退化扩散算子...")

        # 构建网格Laplacian
        L_mesh = self.build_mesh_laplacian()

        # 扩展到向量场（3个分量）
        size = 3 * self.num_vertices
        row_indices = []
        col_indices = []
        data = []

        # 对每个顶点的每个分量应用Laplacian和投影
        for i in range(self.num_vertices):
            P = self.projection_matrices[i]

            # 获取Laplacian的第i行（邻居关系）
            L_row_start = L_mesh.indptr[i]
            L_row_end = L_mesh.indptr[i + 1]

            for comp in range(3):  # 对每个分量
                base_idx = 3 * i + comp

                # 对邻居的Laplacian贡献
                for j in range(L_row_start, L_row_end):
                    neighbor = L_mesh.indices[j]
                    weight = L_mesh.data[j]

                    # 应用投影矩阵
                    # 对邻居的每个分量，计算对当前顶点当前分量的贡献
                    for neighbor_comp in range(3):
                        neighbor_base_idx = 3 * neighbor + neighbor_comp

                        # P[comp, neighbor_comp] * weight
                        projected_weight = P[comp, neighbor_comp] * weight

                        if abs(projected_weight) > 1e-10:
                            row_indices.append(base_idx)
                            col_indices.append(neighbor_base_idx)
                            data.append(projected_weight)

        L_alsfd = csr_matrix((data, (row_indices, col_indices)),
                             shape=(size, size))

        logger.info(f"ALSFD算子构建完成: {L_alsfd.nnz} 非零元素")

        return L_alsfd

    def diffuse_vector_field(
        self,
        initial_vectors: np.ndarray,
        time_step: float = 0.001,
        num_iterations: int = 10,
        use_projection: bool = False  # 修复版：默认不使用过度投影
    ) -> np.ndarray:
        """
        执行ALSFD向量场扩散（修复版）

        求解演化方程: ∂v/∂t = L(v), 其中L是退化扩散算子

        Args:
            initial_vectors: [V, 3] 初始向量场
            time_step: 时间步长
            num_iterations: 扩散迭代次数
            use_projection: 是否在每次迭代后投影（默认False，避免过度衰减）

        Returns:
            [V, 3] 扩散后的向量场
        """
        logger.info(f"开始ALSFD扩散: t={time_step}, iterations={num_iterations}")

        # 1. 确保初始向量场是切向的
        tangent_vectors = self.compute_tangent_frame_field(initial_vectors)

        # 2. 构建扩散算子
        L_operator = self.build_alsfd_operator()

        # 3. 隐式时间步进
        # (I - t*L) v_new = v_old
        size = 3 * self.num_vertices
        identity = eye(size, dtype=np.float64, format='csr')
        heat_operator = identity - time_step * L_operator

        # 展平向量场
        v_flat = tangent_vectors.flatten()

        # 迭代扩散
        magnitude_history = []
        for iteration in range(num_iterations):
            try:
                v_flat = spsolve(heat_operator, v_flat)
            except Exception as e:
                logger.warning(f"迭代 {iteration} 失败: {e}")
                break

            # 可选：重新投影到切空间（可能导致数值衰减）
            if use_projection:
                v_reshaped = v_flat.reshape(self.num_vertices, 3)
                v_reshaped = self.compute_tangent_frame_field(v_reshaped)
                v_flat = v_reshaped.flatten()

            # 记录数值范围
            if iteration % 5 == 0:
                v_reshaped = v_flat.reshape(self.num_vertices, 3)
                magnitudes = np.linalg.norm(v_reshaped, axis=1)
                mean_magnitude = np.mean(magnitudes)
                magnitude_history.append(mean_magnitude)
                logger.debug(f"ALSFD扩散迭代 {iteration}/{num_iterations}: 平均模长 {mean_magnitude:.6e}")

        # 4. 最终结果
        final_vectors = v_flat.reshape(self.num_vertices, 3)

        logger.info("ALSFD扩散完成")
        if magnitude_history:
            logger.info(f"最终平均模长: {magnitude_history[-1]:.6e}")

        return final_vectors

    def verify_tangent_preservation(self, vectors: np.ndarray) -> Dict[str, float]:
        """
        验证切向保持性

        检查向量场是否保持在切空间内

        Args:
            vectors: [V, 3] 向量场

        Returns:
            dict: 验证指标
        """
        tangent_components = []
        normal_components = []

        for i in range(self.num_vertices):
            v = vectors[i]
            n = self.normals[i]

            # 确保法向量是单位向量
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                n = n / n_norm

            # 切向分量大小
            v_dot_n = np.dot(v, n)
            v_tangent = v - v_dot_n * n
            tangent_mag = np.linalg.norm(v_tangent)

            # 法向分量大小
            normal_mag = abs(v_dot_n)

            tangent_components.append(tangent_mag)
            normal_components.append(normal_mag)

        # 计算切向保持性比例
        avg_tangent = np.mean(tangent_components)
        avg_normal = np.mean(normal_components)

        # 避免除零错误
        total = avg_tangent + avg_normal
        if total > 1e-10:
            preservation_ratio = avg_tangent / total
        else:
            preservation_ratio = 1.0  # 如果都很小，认为是完美的

        metrics = {
            'avg_tangent_component': float(avg_tangent),
            'avg_normal_component': float(avg_normal),
            'tangent_preservation_ratio': float(preservation_ratio),
            'max_normal_leakage': float(np.max(normal_components)),
        }

        logger.info(f"切向保持性验证: {metrics}")

        return metrics

    def compare_with_surface_bochner(self, vectors: np.ndarray) -> Dict[str, float]:
        """
        与曲面Bochner热流对比

        验证空间ALSFD扩散是否等价于曲面Bochner拉普拉斯

        Args:
            vectors: [V, 3] 向量场

        Returns:
            dict: 对比指标
        """
        # 计算曲面Bochner拉普拉斯（参考）
        surface_laplacian = self._compute_surface_bochner_laplacian(vectors)

        # 计算ALSFD拉普拉斯
        alsfd_laplacian = self._compute_alsfd_laplacian(vectors)

        # 对比差异
        difference = np.linalg.norm(surface_laplacian - alsfd_laplacian)
        relative_diff = difference / (np.linalg.norm(surface_laplacian) + 1e-10)

        comparison = {
            'absolute_difference': difference,
            'relative_difference': relative_diff,
            'equivalence_score': 1.0 / (1.0 + relative_diff),
        }

        logger.info(f"Bochner等价性验证: {comparison}")

        return comparison

    def _compute_surface_bochner_laplacian(self, vectors: np.ndarray) -> np.ndarray:
        """
        计算曲面Bochner拉普拉斯（参考实现）

        Δ_B(v) = P(Δ_𝓜 v)

        Returns:
            [V, 3] Bochner拉普拉斯结果
        """
        # 计算曲面的Laplace-Beltrami算子
        L_surface = self.build_mesh_laplacian()

        # 对每个分量应用Laplace-Beltrami，然后投影
        bochner_laplacian = np.zeros_like(vectors)

        for comp in range(3):
            # Laplace-Beltrami作用于第comp个分量
            delta_component = L_surface @ vectors[:, comp]

            # 投影到切空间
            for i in range(self.num_vertices):
                P = self.projection_matrices[i]
                # delta_component[i]是标量，需要构造向量
                projected_delta = P @ np.array([0, 0, delta_component[i]], dtype=float)
                bochner_laplacian[i, comp] += projected_delta[comp]

        return bochner_laplacian

    def _compute_alsfd_laplacian(self, vectors: np.ndarray) -> np.ndarray:
        """
        计算ALSFD拉普拉斯（空间退化扩散）

        L(v) = P(∇·(P∇v))

        Returns:
            [V, 3] ALSFD拉普拉斯结果
        """
        # 先计算切向投影
        tangent_vectors = self.compute_tangent_frame_field(vectors)

        # 计算Laplacian
        L_mesh = self.build_mesh_laplacian()
        laplacian_result = np.zeros_like(tangent_vectors)

        for comp in range(3):
            laplacian_result[:, comp] = L_mesh @ tangent_vectors[:, comp]

        # 投影到切空间
        alsfd_laplacian = np.zeros_like(laplacian_result)
        for i in range(self.num_vertices):
            P = self.projection_matrices[i]
            alsfd_laplacian[i] = P @ laplacian_result[i]

        return alsfd_laplacian

    def evaluate_diffusion_quality(self, initial_vectors: np.ndarray, final_vectors: np.ndarray) -> Dict[str, float]:
        """
        评估扩散质量

        Args:
            initial_vectors: [V, 3] 初始向量场
            final_vectors: [V, 3] 最终向量场

        Returns:
            dict: 质量指标
        """
        # 1. 切向保持性
        tangent_metrics = self.verify_tangent_preservation(final_vectors)

        # 2. 平滑度改善
        initial_roughness = self._compute_field_roughness(initial_vectors)
        final_roughness = self._compute_field_roughness(final_vectors)
        smoothness_improvement = (initial_roughness - final_roughness) / (initial_roughness + 1e-10)

        # 3. Bochner等价性
        equivalence_metrics = self.compare_with_surface_bochner(final_vectors)

        quality_metrics = {
            'tangent_preservation': tangent_metrics['tangent_preservation_ratio'],
            'smoothness_improvement': smoothness_improvement,
            'bochner_equivalence': equivalence_metrics['equivalence_score'],
            'overall_quality': (
                tangent_metrics['tangent_preservation_ratio'] *
                equivalence_metrics['equivalence_score']
            ),
        }

        logger.info(f"扩散质量评估: {quality_metrics}")

        return quality_metrics

    def _compute_field_roughness(self, vectors: np.ndarray) -> float:
        """
        计算向量场的粗糙度（相邻变化）

        Args:
            vectors: [V, 3] 向量场

        Returns:
            粗糙度值
        """
        roughness = 0.0
        count = 0

        for face in self.faces:
            for i in range(3):
                vi, vj = face[i], face[(i + 1) % 3]
                diff = np.linalg.norm(vectors[vi] - vectors[vj])
                roughness += diff
                count += 1

        return roughness / max(count, 1)
