"""
测试 TextureSamplerField 推理功能

测试覆盖：
- checkpoint 加载和模型恢复
- predict_distribution 输出 shape
- scale clip 到范围
- select_uvs 功能
- OBJ writer 格式
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import sys
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.texture_sampler_inference import TextureSamplerFieldInference
from src.models.texture_sampler_field import TextureSamplerField


def create_test_checkpoint(tmp_path):
    """创建测试用的 checkpoint"""
    # 创建小模型
    model = TextureSamplerField(
        num_mixtures=2,
        hidden_dim=16,
        num_layers=2,
        positional_encoding_freqs=4,
        use_scale_input=True,
    )

    # 创建假纹理
    texture_path = tmp_path / "test.png"
    from PIL import Image
    Image.new('RGB', (32, 32), color=(128, 128, 128)).save(texture_path)

    # 保存 checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'loss': 0.5,
        'best_loss': 0.5,
        'num_mixtures': 2,
        'hidden_dim': 16,
        'num_layers': 2,
        'positional_encoding_freqs': 4,
        'use_scale_input': True,
        'min_scale': 0.001,
        'max_scale': 0.05,
        'texture_path': str(texture_path),
    }
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path, texture_path


class TestTextureSamplerFieldInference:
    """测试 TextureSamplerFieldInference 类"""

    @pytest.fixture
    def fake_checkpoint(self, tmp_path):
        """创建一个假的 checkpoint 用于测试"""
        # 创建一个小模型
        model = TextureSamplerField(
            num_mixtures=4,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,
            use_scale_input=True,
        )

        # 创建假纹理
        import os
        texture_path = tmp_path / "test_texture.png"
        from PIL import Image
        texture_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
        texture_img.save(texture_path)

        # 保存 checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'loss': 0.5,
            'best_loss': 0.5,
            'num_mixtures': 4,
            'hidden_dim': 32,
            'num_layers': 2,
            'positional_encoding_freqs': 4,
            'use_scale_input': True,
            'min_scale': 0.001,
            'max_scale': 0.05,
            'texture_path': str(texture_path),
        }
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path, texture_path

    def _create_checkpoint(self, tmp_path):
        """辅助方法：创建测试用的 checkpoint"""
        # 创建小模型
        model = TextureSamplerField(
            num_mixtures=2,
            hidden_dim=16,
            num_layers=2,
            positional_encoding_freqs=4,
            use_scale_input=True,
        )

        # 创建假纹理
        texture_path = tmp_path / "test.png"
        from PIL import Image
        Image.new('RGB', (32, 32), color=(128, 128, 128)).save(texture_path)

        # 保存 checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'loss': 0.5,
            'best_loss': 0.5,
            'num_mixtures': 2,
            'hidden_dim': 16,
            'num_layers': 2,
            'positional_encoding_freqs': 4,
            'use_scale_input': True,
            'min_scale': 0.001,
            'max_scale': 0.05,
            'texture_path': str(texture_path),
        }
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path, texture_path

    def test_load_checkpoint(self, fake_checkpoint):
        """测试从 checkpoint 加载模型"""
        checkpoint_path, texture_path = fake_checkpoint

        # 创建推理器
        inference = TextureSamplerFieldInference(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 验证模型加载
        assert inference.model is not None
        assert inference.model.num_mixtures == 4
        assert inference.metadata['num_mixtures'] == 4
        assert inference.metadata['min_scale'] == 0.001
        assert inference.metadata['max_scale'] == 0.05

    def test_predict_distribution_shape(self, fake_checkpoint):
        """测试 predict_distribution 输出 shape"""
        checkpoint_path, texture_path = fake_checkpoint

        inference = TextureSamplerFieldInference(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 准备输入
        N = 10
        positions = np.random.randn(N, 3).astype(np.float32)
        scales = np.random.rand(N, 1).astype(np.float32) * 0.01

        # 预测
        uvs, weights, sigmas = inference.predict_distribution(positions, scales)

        # 验证 shape
        assert uvs.shape == (N, 4, 2), f"UVs shape 错误: {uvs.shape}"
        assert weights.shape == (N, 4), f"Weights shape 错误: {weights.shape}"
        assert sigmas.shape == (N, 4, 1), f"Sigmas shape 错误: {sigmas.shape}"

        # 验证 UV 范围
        assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0), "UV 应在 [0, 1] 范围内"

        # 验证权重和为 1
        weight_sums = weights.sum(axis=-1)
        assert np.allclose(weight_sums, 1.0, atol=1e-5), "权重和应该为 1"

        # 验证 sigma 为正
        assert np.all(sigmas > 0), "Sigma 应该为正"

    def test_predict_distribution_no_nan(self, fake_checkpoint):
        """测试预测结果不包含 NaN"""
        checkpoint_path, texture_path = fake_checkpoint

        inference = TextureSamplerFieldInference(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 使用各种输入
        positions = np.random.randn(20, 3).astype(np.float32)
        scales = np.random.rand(20, 1).astype(np.float32) * 0.02

        uvs, weights, sigmas = inference.predict_distribution(positions, scales)

        # 验证无 NaN
        assert not np.isnan(uvs).any(), "UVs 不应包含 NaN"
        assert not np.isnan(weights).any(), "Weights 不应包含 NaN"
        assert not np.isnan(sigmas).any(), "Sigmas 不应包含 NaN"

    def test_select_uvs_argmax(self, fake_checkpoint):
        """测试 select_uvs argmax 模式"""
        checkpoint_path, texture_path = fake_checkpoint

        inference = TextureSamplerFieldInference(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 准备数据
        N = 5
        K = 4
        uvs = np.random.rand(N, K, 2).astype(np.float32)
        weights = np.array([
            [0.1, 0.6, 0.2, 0.1],
            [0.2, 0.3, 0.4, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.7],
            [0.4, 0.4, 0.1, 0.1],
        ], dtype=np.float32)

        # 选择 argmax
        selected = inference.select_uvs(uvs, weights, mode="argmax")

        # 验证 shape
        assert selected.shape == (N, 2), f"Selected UVs shape 错误: {selected.shape}"

        # 验证选择了最大权重的索引
        expected_indices = weights.argmax(axis=-1)
        for i in range(N):
            expected_uv = uvs[i, expected_indices[i]]
            assert np.allclose(selected[i], expected_uv), f"Row {i} 应选择最大权重的 UV"

    def test_select_uvs_weighted(self, fake_checkpoint):
        """测试 select_uvs weighted 模式"""
        checkpoint_path, texture_path = fake_checkpoint

        inference = TextureSamplerFieldInference(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 准备数据
        N = 3
        K = 2
        uvs = np.array([
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.2, 0.3], [0.8, 0.7]],
        ], dtype=np.float32)
        weights = np.array([
            [0.7, 0.3],
            [0.5, 0.5],
            [0.4, 0.6],
        ], dtype=np.float32)

        # 选择 weighted
        selected = inference.select_uvs(uvs, weights, mode="weighted")

        # 验证 shape
        assert selected.shape == (N, 2), f"Selected UVs shape 错误: {selected.shape}"

        # 验证加权平均
        for i in range(N):
            expected = (weights[i, :, np.newaxis] * uvs[i]).sum(axis=0)
            assert np.allclose(selected[i], expected), f"Row {i} 加权平均错误"

    def test_scale_clip_to_range(self, fake_checkpoint):
        """测试 scale 被 clip 到 checkpoint 范围"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        checkpoint_path, texture_path = fake_checkpoint

        # 创建完整的管线（包含 estimate_scales 方法）
        pipeline = TextureSamplerFieldDemoPipeline(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 创建极小的 triangle mesh 用于测试
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 边长约为 0.5-1.0，但 checkpoint 范围是 [0.001, 0.05]
        # 使用 edge_mean 模式估计 scale
        scales = pipeline.estimate_scales(mesh, mode="edge_mean")

        # 验证 clip 到范围
        assert np.all(scales >= 0.001), "Scale 应该 >= min_scale"
        assert np.all(scales <= 0.05), "Scale 应该 <= max_scale"


class TestOBJWriter:
    """测试 OBJ writer 功能"""

    def test_obj_writer_format(self, tmp_path):
        """测试 OBJ writer 输出格式"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        # 创建极简 mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 假 UV
        corner_uvs = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ], dtype=np.float32)

        # 创建完整管线
        checkpoint_path, texture_path = create_test_checkpoint(tmp_path)
        pipeline = TextureSamplerFieldDemoPipeline(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 导出
        output_path = tmp_path / "test_output.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 验证文件存在
        assert output_path.exists(), "OBJ 文件应该存在"
        assert output_path.with_suffix('.mtl').exists(), "MTL 文件应该存在"

        # 读取并验证内容
        with open(output_path, 'r') as f:
            obj_content = f.read()

        # 验证 mtllib
        assert "mtllib" in obj_content, "OBJ 应该包含 mtllib"
        assert "usemtl" in obj_content, "OBJ 应该包含 usemtl"

        # 验证顶点
        assert obj_content.count("v ") == 3, "应该有 3 个顶点"

        # 验证 UV（每个 corner 一个 vt，3 个 corner）
        assert obj_content.count("vt ") == 3, "应该有 3 个 vt"

        # 验证面格式（v/vt 格式）
        assert "f " in obj_content, "应该有面定义"
        assert "/" in obj_content, "面应该使用 v/vt 格式"

    def test_obj_writer_uv_count(self, tmp_path):
        """测试 UV 数量等于 num_faces * 3"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        # 创建 2 个面的 mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 随机 UV
        corner_uvs = np.random.rand(6, 2).astype(np.float32)  # 2 faces * 3 corners

        # 创建完整管线
        checkpoint_path, texture_path = create_test_checkpoint(tmp_path)
        pipeline = TextureSamplerFieldDemoPipeline(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 导出
        output_path = tmp_path / "test_uv_count.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 验证 UV 数量
        with open(output_path, 'r') as f:
            obj_content = f.read()

        vt_count = obj_content.count("vt ")
        assert vt_count == 6, f"UV 数量应该等于 num_faces * 3: {vt_count} != 6"

    def test_obj_writer_mtl_map_kd(self, tmp_path):
        """测试 MTL 包含 map_Kd"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        corner_uvs = np.random.rand(3, 2).astype(np.float32)

        # 创建完整管线
        checkpoint_path, texture_path = create_test_checkpoint(tmp_path)
        pipeline = TextureSamplerFieldDemoPipeline(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 导出
        output_path = tmp_path / "test_mtl.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 验证 MTL
        mtl_path = output_path.with_suffix('.mtl')
        with open(mtl_path, 'r') as f:
            mtl_content = f.read()

        assert "map_Kd" in mtl_content, "MTL 应该包含 map_Kd"
        assert "test.png" in mtl_content, "MTL 应该引用纹理文件"


class TestProceduralTextureError:
    """测试程序化纹理的错误处理"""

    def test_procedural_texture_error(self, tmp_path):
        """测试程序化纹理应该报错"""
        # 创建假 checkpoint（texture_path = "procedural"）
        model = TextureSamplerField(
            num_mixtures=2,
            hidden_dim=16,
            num_layers=2,
            positional_encoding_freqs=4,
            use_scale_input=True,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'loss': 0.5,
            'best_loss': 0.5,
            'num_mixtures': 2,
            'hidden_dim': 16,
            'num_layers': 2,
            'positional_encoding_freqs': 4,
            'use_scale_input': True,
            'min_scale': 0.001,
            'max_scale': 0.05,
            'texture_path': "procedural",  # 程序化纹理
        }
        torch.save(checkpoint, checkpoint_path)

        # 尝试加载（不提供 texture 参数）
        with pytest.raises(ValueError) as exc_info:
            TextureSamplerFieldInference(
                checkpoint_path=str(checkpoint_path),
                device="cpu",
            )

        assert "程序化纹理" in str(exc_info.value) or "procedural" in str(exc_info.value).lower()


class TestCriticalFixes:
    """测试关键修复的正确性"""

    def test_checkpoint_weights_loaded(self, tmp_path):
        """测试 checkpoint 权重被正确加载"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        # 创建测试 checkpoint
        checkpoint_path, texture_path = create_test_checkpoint(tmp_path)

        # 加载推理器
        inference = TextureSamplerFieldInference(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 验证模型参数与 checkpoint 一致
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_params = checkpoint['model_state_dict']

        for name, param in inference.model.named_parameters():
            checkpoint_param = checkpoint_params[name]
            assert torch.allclose(param, checkpoint_param, atol=1e-6), \
                f"参数 {name} 与 checkpoint 不一致"

    def test_face_vt_indices_unique(self, tmp_path):
        """测试每个 face corner 的 VT 索引是唯一的"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        corner_uvs = np.random.rand(3, 2).astype(np.float32)

        # 创建完整管线
        checkpoint_path, texture_path = create_test_checkpoint(tmp_path)
        pipeline = TextureSamplerFieldDemoPipeline(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 导出
        output_path = tmp_path / "test_vt.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 解析 face 行
        with open(output_path, 'r') as f:
            for line in f:
                if line.startswith('f '):
                    parts = line.strip().split()[1:]  # ['1/1', '2/2', '3/3']
                    vt_indices = [int(p.split('/')[1]) for p in parts]  # [1, 2, 3]
                    assert vt_indices == [1, 2, 3], \
                        f"Face corners 应该有不同的 VT 索引: {vt_indices}"

    def test_mtl_map_kd_relative_path(self, tmp_path):
        """测试 MTL map_Kd 路径相对于 OBJ 目录可解析"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        corner_uvs = np.random.rand(3, 2).astype(np.float32)

        # 创建完整管线
        checkpoint_path, texture_path = create_test_checkpoint(tmp_path)
        pipeline = TextureSamplerFieldDemoPipeline(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 导出到子目录
        output_dir = tmp_path / "subdir"
        output_dir.mkdir()
        output_path = output_dir / "test_relative.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 解析 MTL 获取 map_Kd 路径
        mtl_path = output_path.with_suffix('.mtl')
        with open(mtl_path, 'r') as f:
            for line in f:
                if 'map_Kd' in line:
                    map_kd_path = line.split()[-1]
                    break

        # 验证路径相对于 OBJ 目录可解析
        texture_from_mtl = output_dir / map_kd_path
        assert texture_from_mtl.exists(), \
            f"MTL map_Kd 路径 {map_kd_path} 相对于 {output_dir} 应该存在: {texture_from_mtl}"

    def test_no_simplify_flag(self, tmp_path):
        """测试 --no-simplify 标志生效"""
        from scripts.infer_texture_sampler_field import TextureSamplerFieldDemoPipeline

        # 创建高模
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)

        import trimesh
        high_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        high_mesh_path = tmp_path / "high.obj"
        high_mesh.export(str(high_mesh_path))

        # 创建完整管线
        checkpoint_path, texture_path = create_test_checkpoint(tmp_path)
        pipeline = TextureSamplerFieldDemoPipeline(
            checkpoint_path=str(checkpoint_path),
            texture_path=str(texture_path),
            device="cpu",
        )

        # 使用 simplify=False
        low_mesh = pipeline.prepare_low_mesh(
            input_mesh_path=str(high_mesh_path),
            simplify=False,
        )

        # 验证面数不变（未减面）
        assert len(low_mesh.faces) == len(high_mesh.faces), \
            f"simplify=False 时面数应该不变: {len(low_mesh.faces)} != {len(high_mesh.faces)}"


if __name__ == "__main__":
    # 使用 pytest 运行完整测试
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
    exit(result.returncode)
