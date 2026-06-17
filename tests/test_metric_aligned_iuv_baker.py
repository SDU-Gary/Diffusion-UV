"""
MA-IUVF Baker 测试

测试烘焙器的数学正确性和功能完整性
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import shutil

from src.data.metric_aligned_iuv_baker import (
    MetricAlignedIUVBaker,
    MetricAlignedIUVSampleData,
    load_baked_data,
)


class TestTriangleJacobian:
    """测试单三角形雅可比计算"""

    def test_simple_triangle_jacobian(self):
        """测试简单三角形：v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), uv0=(0,0), uv1=(2,0), uv2=(0,3)"""
        # 构造测试三角形
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)

        uvs = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ], dtype=torch.float32)

        # 计算雅可比
        baker = MetricAlignedIUVBaker.__new__(MetricAlignedIUVBaker)
        j_3d_gt = baker._compute_triangle_jacobian(vertices, uvs)

        # 预期结果
        expected_j = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ], dtype=torch.float32)

        # 验证雅可比
        assert j_3d_gt.shape == (2, 3), f"雅可比形状错误: {j_3d_gt.shape}"
        assert torch.allclose(j_3d_gt, expected_j, atol=1e-5), \
            f"雅可比计算错误:\n预期:\n{expected_j}\n实际:\n{j_3d_gt}"

    def test_skewed_triangle_jacobian(self):
        """测试斜三角形：非轴对齐的UV映射"""
        # 构造斜三角形（但仍在XY平面内，便于验证）
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],  # 斜边
        ], dtype=torch.float32)

        # UV映射：u=2x, v=3y（斜映射）
        uvs = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],  # u = 2x
            [0.0, 3.0],  # v = 3y
        ], dtype=torch.float32)

        # 计算雅可比
        baker = MetricAlignedIUVBaker.__new__(MetricAlignedIUVBaker)
        j_3d_gt = baker._compute_triangle_jacobian(vertices, uvs)

        # 预期结果：由于三角形几何倾斜（vertex 2在x=0.5而非x=0），雅可比应该有非对角元素
        # J_3d = [[2, -1, 0], [0, 3, 0]]
        # 解释：u方向导数为(2, -1, 0)因为在x方向增加1同时u增加2，但倾斜使得y方向补偿
        expected_j = torch.tensor([
            [2.0, -1.0, 0.0],
            [0.0, 3.0, 0.0],
        ], dtype=torch.float32)

        # 验证雅可比
        assert j_3d_gt.shape == (2, 3), f"雅可比形状错误: {j_3d_gt.shape}"
        assert torch.allclose(j_3d_gt, expected_j, atol=1e-4), \
            f"斜三角形雅可比计算错误:\n预期:\n{expected_j}\n实际:\n{j_3d_gt}"

    def test_normal_zero_gradient(self):
        """测试法向零梯度：J_3d @ normal ≈ [0, 0]"""
        # 平面三角形
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)

        uvs = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ], dtype=torch.float32)

        # 计算雅可比
        baker = MetricAlignedIUVBaker.__new__(MetricAlignedIUVBaker)
        j_3d_gt = baker._compute_triangle_jacobian(vertices, uvs)

        # 法向
        normal = torch.tensor([0.0, 0.0, 1.0])

        # J @ n
        result = j_3d_gt @ normal

        # 应该接近[0, 0]
        assert torch.allclose(result, torch.zeros(2), atol=1e-5), \
            f"法向零梯度失效: J@n = {result}"

    def test_scaled_triangle(self):
        """测试缩放三角形：验证雅可比正确缩放"""
        # 基础三角形
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)

        # UV缩放2倍
        uvs = torch.tensor([
            [0.0, 0.0],
            [4.0, 0.0],  # 2x
            [0.0, 6.0],  # 2x
        ], dtype=torch.float32)

        # 计算雅可比
        baker = MetricAlignedIUVBaker.__new__(MetricAlignedIUVBaker)
        j_3d_gt = baker._compute_triangle_jacobian(vertices, uvs)

        # 预期：雅可比也应该是2倍
        expected_j = torch.tensor([
            [4.0, 0.0, 0.0],  # 2x
            [0.0, 6.0, 0.0],  # 2x
        ], dtype=torch.float32)

        assert torch.allclose(j_3d_gt, expected_j, atol=1e-5), \
            f"缩放雅可比错误:\n预期:\n{expected_j}\n实际:\n{j_3d_gt}"


class TestBakerFunctionality:
    """测试烘焙器功能"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def dummy_mesh(self, temp_dir):
        """创建dummy mesh（带UV）"""
        import trimesh

        # 简单三角形
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        faces = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 添加UV
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)

        mesh_path = temp_dir / "dummy.obj"
        mesh.export(str(mesh_path))

        return mesh_path

    def test_sample_point_uv_anchor(self):
        """测试采样点的UV anchor计算正确性"""
        # 构造已知UV的三角形
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)

        uvs = torch.tensor([
            [0.0, 0.0],   # (0, 0)
            [1.0, 0.0],   # (1, 0)
            [0.0, 1.0],   # (0, 1)
        ], dtype=torch.float32)

        # 模拟采样：重心坐标 (0.2, 0.3, 0.5)
        bary = torch.tensor([0.2, 0.3, 0.5])

        # 计算采样点的UV（根据重心坐标插值）
        uv_anchor_sample = (uvs * bary.view(-1, 1)).sum(dim=0)

        # 预期结果：(0.2*0 + 0.3*1 + 0.5*0, 0.2*0 + 0.3*0 + 0.5*1) = (0.3, 0.5)
        expected_uv = torch.tensor([0.3, 0.5])

        assert torch.allclose(uv_anchor_sample, expected_uv, atol=1e-6), \
            f"采样点UV anchor错误: {uv_anchor_sample} vs {expected_uv}"

    def test_baker_initialization(self, dummy_mesh):
        """测试烘焙器初始化"""
        baker = MetricAlignedIUVBaker(str(dummy_mesh), seed=42)

        # 注意：当使用OBJ解析器时，baker.mesh 为None
        # 但 face_vertices 和 face_uvs 应该正常工作
        assert baker.face_vertices.shape[0] == 1  # 1个面
        assert baker.face_uvs.shape == (1, 3, 2)

    def test_face_component_chart_mode(self, dummy_mesh):
        """测试face_component chart模式"""
        baker = MetricAlignedIUVBaker(str(dummy_mesh), seed=42)

        # 烘焙
        data = baker.bake(
            num_samples=100,
            extrusion_sigma_ratio=0.01,
            chart_mode="face_component",
        )

        # 验证数据shape
        assert data.pos.shape[1] == 3, f"pos shape错误: {data.pos.shape}"
        assert data.j_3d_gt.shape == (len(data.pos), 2, 3), \
            f"j_3d_gt shape错误: {data.j_3d_gt.shape}"
        assert data.uv_anchor.shape == (len(data.pos), 2), \
            f"uv_anchor shape错误: {data.uv_anchor.shape}"
        assert data.chart_id.shape == (len(data.pos),), \
            f"chart_id shape错误: {data.chart_id.shape}"

        # face_component模式：所有样本应该chart_id=0
        assert torch.all(data.chart_id == 0), "face_component模式应该所有chart_id=0"

    def test_save_and_load(self, temp_dir, dummy_mesh):
        """测试保存和加载"""
        baker = MetricAlignedIUVBaker(str(dummy_mesh), seed=42)

        # 烘焙
        data = baker.bake(num_samples=100, extrusion_sigma_ratio=0.01)

        # 保存
        output_path = temp_dir / "baked.pt"
        baker.save(data, str(output_path))

        assert output_path.exists(), "保存失败"

        # 加载
        loaded_data, metadata = load_baked_data(str(output_path))

        # 验证数据一致性
        assert torch.allclose(loaded_data.pos, data.pos), "加载的pos不一致"
        assert torch.allclose(loaded_data.j_3d_gt, data.j_3d_gt), "加载的j_3d_gt不一致"
        assert torch.allclose(loaded_data.uv_anchor, data.uv_anchor), "加载的uv_anchor不一致"
        assert torch.all(loaded_data.chart_id == data.chart_id), "加载的chart_id不一致"

        # 验证metadata
        assert 'num_charts' in metadata
        assert 'num_samples' in metadata
        assert metadata['num_samples'] == len(data.pos)

    def test_extrusion_sigma_calculation(self, dummy_mesh):
        """测试挤出sigma计算"""
        baker = MetricAlignedIUVBaker(str(dummy_mesh), seed=42)

        # 烘焙
        extrusion_sigma_ratio = 0.01
        data = baker.bake(
            num_samples=100,
            extrusion_sigma_ratio=extrusion_sigma_ratio,
        )

        # 验证挤出效果：样本应该在表面附近扩散
        # 计算样本到原面的距离
        face_center = baker.face_vertices.mean(dim=1)[0]  # [3]
        distances = (data.pos - face_center).norm(dim=-1)

        # 距离应该有一定方差（由于挤出）
        assert distances.std() > 1e-6, "挤出似乎没有生效"

    def test_output_shapes_consistency(self, dummy_mesh):
        """测试输出shape一致性"""
        baker = MetricAlignedIUVBaker(str(dummy_mesh), seed=42)

        data = baker.bake(num_samples=100, extrusion_sigma_ratio=0.01)

        num_samples = len(data.pos)

        # 所有tensor第一维应该一致
        assert data.j_3d_gt.shape[0] == num_samples
        assert data.uv_anchor.shape[0] == num_samples
        assert data.chart_id.shape[0] == num_samples

        # 雅可比维度
        assert data.j_3d_gt.shape[1] == 2  # [u, v]
        assert data.j_3d_gt.shape[2] == 3  # [x, y, z]

        # UV维度
        assert data.uv_anchor.shape[1] == 2  # [u, v]


if __name__ == "__main__":
    # 快速自检
    pytest.main([__file__, "-v", "--tb=short"])
