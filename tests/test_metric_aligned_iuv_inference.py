"""
MA-IUVF 推理测试

测试推理功能和OBJ导出
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import shutil
import json

from src.inference.metric_aligned_iuv_inference import MetricAlignedIUVInference
from src.models.metric_aligned_iuv_field import create_model


class TestInferenceLoading:
    """测试推理器加载"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def dummy_checkpoint(self, temp_dir):
        """创建dummy checkpoint"""
        # 创建模型
        model = create_model(
            num_charts=3,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,  # 与checkpoint一致
        )

        # 创建checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'epoch': 10,
            'loss': 0.5,
            'best_loss': 0.4,
            'num_charts': 3,
            'hidden_dim': 32,
            'num_layers': 2,
            'positional_encoding_freqs': 4,  # 与模型一致
            'baker_metadata': {
                'num_charts': 3,
            },
        }

        checkpoint_path = temp_dir / "dummy.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def test_load_checkpoint(self, dummy_checkpoint):
        """测试加载checkpoint"""
        inference = MetricAlignedIUVInference(
            checkpoint_path=str(dummy_checkpoint),
            device="cpu",
        )

        assert inference.model is not None
        assert inference.model.num_charts == 3

    def test_missing_required_fields(self, temp_dir):
        """测试缺少必要字段"""
        # 创建不完整的checkpoint
        checkpoint = {
            'model_state_dict': {},
            'num_charts': 3,
            # 缺少其他字段
        }

        checkpoint_path = temp_dir / "incomplete.pt"
        torch.save(checkpoint, checkpoint_path)

        # 应该抛出异常
        with pytest.raises(ValueError, match="缺少必要字段"):
            MetricAlignedIUVInference(
                checkpoint_path=str(checkpoint_path),
                device="cpu",
            )


class TestPrediction:
    """测试预测功能"""

    @pytest.fixture
    def inference(self, temp_dir):
        """创建推理器"""
        # 创建模型（使用固定参数确保一致）
        model = create_model(
            num_charts=3,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,  # 固定为4
        )

        # 创建checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'epoch': 10,
            'loss': 0.5,
            'best_loss': 0.4,
            'num_charts': 3,
            'hidden_dim': 32,
            'num_layers': 2,
            'positional_encoding_freqs': 4,  # 与模型一致
            'baker_metadata': {'num_charts': 3},
        }

        checkpoint_path = temp_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        return MetricAlignedIUVInference(str(checkpoint_path), device="cpu")

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    def test_predict_shapes(self, inference):
        """测试预测输出shape"""
        positions = np.random.randn(10, 3).astype(np.float32)

        logits, uv_preds = inference.predict(positions, batch_size=5)

        # 验证shape
        assert logits.shape == (10, 3), f"logits shape错误: {logits.shape}"
        assert uv_preds.shape == (10, 3, 2), f"uv_preds shape错误: {uv_preds.shape}"

    def test_predict_no_nan(self, inference):
        """测试预测无NaN"""
        positions = np.random.randn(20, 3).astype(np.float32)

        logits, uv_preds = inference.predict(positions)

        assert not np.isnan(logits).any(), "logits包含NaN"
        assert not np.isnan(uv_preds).any(), "uv_preds包含NaN"

    def test_select_uvs_argmax(self, inference):
        """测试argmax模式UV选择"""
        # 模拟输出
        logits = np.random.randn(10, 3).astype(np.float32)
        uv_preds = np.random.rand(10, 3, 2).astype(np.float32)

        selected_uvs, chart_ids = inference.select_uvs(logits, uv_preds, mode="argmax")

        # 验证shape
        assert selected_uvs.shape == (10, 2), f"selected_uvs shape错误: {selected_uvs.shape}"
        assert chart_ids.shape == (10,), f"chart_ids shape错误: {chart_ids.shape}"

        # 验证chart ID在有效范围
        assert np.all(chart_ids >= 0) and np.all(chart_ids < 3), "chart ID超出范围"

    def test_select_uvs_sample(self, inference):
        """测试sample模式UV选择"""
        logits = np.random.randn(10, 3).astype(np.float32)
        uv_preds = np.random.rand(10, 3, 2).astype(np.float32)

        selected_uvs, chart_ids = inference.select_uvs(logits, uv_preds, mode="sample")

        # 验证shape
        assert selected_uvs.shape == (10, 2)
        assert chart_ids.shape == (10,)

        # 验证chart ID在有效范围
        assert np.all(chart_ids >= 0) and np.all(chart_ids < 3)


class TestOBJExport:
    """测试OBJ导出"""

    @pytest.fixture
    def dummy_checkpoint(self, temp_dir):
        """创建dummy checkpoint"""
        model = create_model(
            num_charts=3,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,  # 与其他测试一致
        )

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'epoch': 10,
            'loss': 0.5,
            'best_loss': 0.4,
            'num_charts': 3,
            'hidden_dim': 32,
            'num_layers': 2,
            'positional_encoding_freqs': 4,  # 与模型一致
            'baker_metadata': {'num_charts': 3},
        }

        checkpoint_path = temp_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def dummy_texture(self, temp_dir):
        """创建dummy纹理"""
        from PIL import Image

        texture_path = temp_dir / "texture.png"
        Image.new('RGB', (64, 64), color=(128, 128, 128)).save(texture_path)

        return texture_path

    def test_obj_writer_format(self, dummy_checkpoint, dummy_texture, temp_dir):
        """测试OBJ writer输出格式"""
        from scripts.infer_metric_aligned_iuv import MetricAlignedIUVDemoPipeline

        # 创建管线
        pipeline = MetricAlignedIUVDemoPipeline(
            checkpoint_path=str(dummy_checkpoint),
            texture_path=str(dummy_texture),
            device="cpu",
        )

        # 创建dummy mesh
        import trimesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 预测UV
        corner_uvs = np.random.rand(3, 2).astype(np.float32)

        # 导出OBJ
        output_path = temp_dir / "test.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 验证OBJ文件
        assert output_path.exists(), "OBJ文件不存在"

        with open(output_path, 'r') as f:
            content = f.read()

        # 检查必要内容
        assert "mtllib" in content, "OBJ缺少mtllib"
        assert "usemtl" in content, "OBJ缺少usemtl"
        assert content.count("v ") == 3, "OBJ顶点数错误"
        assert content.count("vt ") == 3, "OBJ UV数错误"
        assert "f " in content, "OBJ缺少面"

    def test_face_vt_indices(self, dummy_checkpoint, dummy_texture, temp_dir):
        """测试face的VT索引递增"""
        from scripts.infer_metric_aligned_iuv import MetricAlignedIUVDemoPipeline

        pipeline = MetricAlignedIUVDemoPipeline(
            checkpoint_path=str(dummy_checkpoint),
            texture_path=str(dummy_texture),
            device="cpu",
        )

        # 创建mesh：2个面
        import trimesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 预测UV（6个corner）
        corner_uvs = np.random.rand(6, 2).astype(np.float32)

        # 导出
        output_path = temp_dir / "test.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 解析face行
        with open(output_path, 'r') as f:
            for line in f:
                if line.startswith('f '):
                    parts = line.strip().split()[1:]  # ['v1/vt1', 'v2/vt2', 'v3/vt3']
                    vt_indices = [int(p.split('/')[1]) for p in parts]

                    # 每个corner的VT索引应该不同
                    assert len(vt_indices) == 3, "face应该有3个corner"
                    assert vt_indices[0] != vt_indices[1], "corner的VT索引应该不同"
                    assert vt_indices[1] != vt_indices[2], "corner的VT索引应该不同"

    def test_mtl_map_kd(self, dummy_checkpoint, dummy_texture, temp_dir):
        """测试MTL包含map_Kd"""
        from scripts.infer_metric_aligned_iuv import MetricAlignedIUVDemoPipeline

        pipeline = MetricAlignedIUVDemoPipeline(
            checkpoint_path=str(dummy_checkpoint),
            texture_path=str(dummy_texture),
            device="cpu",
        )

        # 创建mesh
        import trimesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        corner_uvs = np.random.rand(3, 2).astype(np.float32)

        # 导出
        output_path = temp_dir / "test.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
        )

        # 验证MTL
        mtl_path = output_path.with_suffix('.mtl')
        assert mtl_path.exists(), "MTL文件不存在"

        with open(mtl_path, 'r') as f:
            content = f.read()

        assert "map_Kd" in content, "MTL缺少map_Kd"

    def test_texture_copy(self, dummy_checkpoint, dummy_texture, temp_dir):
        """测试纹理复制"""
        from scripts.infer_metric_aligned_iuv import MetricAlignedIUVDemoPipeline

        pipeline = MetricAlignedIUVDemoPipeline(
            checkpoint_path=str(dummy_checkpoint),
            texture_path=str(dummy_texture),
            device="cpu",
        )

        # 创建mesh
        import trimesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        corner_uvs = np.random.rand(3, 2).astype(np.float32)

        # 导出（默认复制纹理）
        output_path = temp_dir / "test.obj"
        pipeline.export_obj_with_uv(
            mesh=mesh,
            corner_uvs=corner_uvs,
            output_obj_path=str(output_path),
            copy_texture=True,
        )

        # 验证纹理被复制
        copied_texture = temp_dir / "texture.png"
        assert copied_texture.exists(), "纹理未被复制"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
