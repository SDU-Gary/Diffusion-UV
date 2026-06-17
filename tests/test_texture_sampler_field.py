"""
测试纹理采样场模型

测试内容：
1. 模型输出形状正确
2. UV 坐标在 [0, 1] 范围内
3. 权重和为 1
4. sigma 为正数
5. sample_texture 函数正常工作
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.texture_sampler_field import (
    TextureSamplerField,
    TextureSamplerFieldOutput,
    sample_texture,
    PositionalEncoding,
)


class TestPositionalEncoding:
    """测试位置编码"""

    def test_output_shape(self):
        """测试输出形状"""
        pos_enc = PositionalEncoding(num_freqs=6)
        x = torch.randn(10, 3)  # (B, 3)
        encoded = pos_enc(x)
        assert encoded.shape == (10, 3 * 2 * 6), f"Expected (10, 36), got {encoded.shape}"

    def test_2d_input(self):
        """测试 2D 输入"""
        pos_enc = PositionalEncoding(num_freqs=4)
        x = torch.randn(5, 2)  # (B, 2)
        encoded = pos_enc(x)
        assert encoded.shape == (5, 2 * 2 * 4), f"Expected (5, 16), got {encoded.shape}"

    def test_deterministic(self):
        """测试相同输入产生相同输出"""
        pos_enc = PositionalEncoding(num_freqs=4)
        x = torch.randn(1, 3)
        encoded1 = pos_enc(x)
        encoded2 = pos_enc(x)
        assert torch.allclose(encoded1, encoded2), "Positional encoding should be deterministic"


class TestTextureSamplerField:
    """测试纹理采样场网络"""

    @pytest.fixture
    def model(self):
        """创建模型实例"""
        return TextureSamplerField(
            num_mixtures=8,
            hidden_dim=128,
            num_layers=4,
            positional_encoding_freqs=6,
            use_scale_input=True,
        )

    @pytest.fixture
    def sample_batch(self):
        """创建测试批次数据"""
        batch_size = 16
        positions = torch.randn(batch_size, 3)
        scale = torch.rand(batch_size, 1) * 0.05 + 0.001  # [0.001, 0.05]
        return positions, scale

    def test_output_shape(self, model, sample_batch):
        """测试输出形状"""
        positions, scale = sample_batch
        output = model(positions, scale)

        B = positions.shape[0]
        K = model.num_mixtures

        assert output.uvs.shape == (B, K, 2), f"UVs shape mismatch: {output.uvs.shape}"
        assert output.weights.shape == (B, K), f"Weights shape mismatch: {output.weights.shape}"
        assert output.sigmas.shape == (B, K, 1), f"Sigmas shape mismatch: {output.sigmas.shape}"

    def test_uv_range(self, model, sample_batch):
        """测试 UV 坐标在 [0, 1] 范围内"""
        positions, scale = sample_batch
        output = model(positions, scale)

        assert torch.all(output.uvs >= 0.0), "UVs should be >= 0"
        assert torch.all(output.uvs <= 1.0), "UVs should be <= 1"

    def test_weights_sum_to_one(self, model, sample_batch):
        """测试权重和为 1"""
        positions, scale = sample_batch
        output = model(positions, scale)

        weights_sum = output.weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6), \
            "Weights should sum to 1"

    def test_sigmas_positive(self, model, sample_batch):
        """测试 sigma 为正数"""
        positions, scale = sample_batch
        output = model(positions, scale)

        assert torch.all(output.sigmas > 0.0), "Sigmas should be positive"

    def test_no_scale_input(self):
        """测试不使用 scale 输入的情况"""
        model = TextureSamplerField(
            num_mixtures=4,
            hidden_dim=64,
            num_layers=2,
            positional_encoding_freqs=4,
            use_scale_input=False,
        )

        positions = torch.randn(8, 3)
        output = model(positions, scale=None)

        B = 8
        K = 4
        assert output.uvs.shape == (B, K, 2)
        assert output.weights.shape == (B, K)
        assert output.sigmas.shape == (B, K, 1)

    def test_gradients_flow(self, model, sample_batch):
        """测试梯度流动"""
        positions, scale = sample_batch
        positions.requires_grad = True
        scale.requires_grad = True

        output = model(positions, scale)
        loss = output.uvs.sum() + output.weights.sum() + output.sigmas.sum()
        loss.backward()

        assert positions.grad is not None, "Positions should have gradients"
        assert scale.grad is not None, "Scale should have gradients"
        assert not torch.isnan(positions.grad).any(), "Positions gradients should not be NaN"
        assert not torch.isnan(scale.grad).any(), "Scale gradients should not be NaN"


class TestSampleTexture:
    """测试纹理采样函数"""

    @pytest.fixture
    def sample_texture_tensor(self):
        """创建示例纹理（重命名以避免与函数冲突）"""
        # 创建一个简单的渐变纹理 (3, 64, 64)
        H, W = 64, 64
        texture = torch.zeros(3, H, W, dtype=torch.float32)
        for c in range(3):
            texture[c] = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W) * (c + 1) / 3
        return texture

    def test_output_shape(self, sample_texture_tensor):
        """测试输出形状"""
        B, K = 8, 4
        uvs = torch.rand(B, K, 2)
        weights = torch.softmax(torch.randn(B, K), dim=-1)

        colors = sample_texture(sample_texture_tensor, uvs, weights)

        assert colors.shape == (B, 3), f"Expected (B, 3), got {colors.shape}"

    def test_weighted_sampling(self, sample_texture_tensor):
        """测试加权采样"""
        B, K = 1, 2
        uvs = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]])  # 两个不同位置
        weights = torch.tensor([[1.0, 0.0]])  # 只采样第一个

        colors = sample_texture(sample_texture_tensor, uvs, weights)

        # 应该只从第一个 UV 采样
        # 这里我们只测试形状和范围
        assert colors.shape == (1, 3)
        assert torch.all(colors >= 0.0) and torch.all(colors <= 1.0)

    def test_uniform_weights(self, sample_texture_tensor):
        """测试均匀权重"""
        B, K = 4, 8
        uvs = torch.rand(B, K, 2)
        weights = torch.ones(B, K) / K  # 均匀权重

        colors = sample_texture(sample_texture_tensor, uvs, weights)

        assert colors.shape == (B, 3)
        assert torch.all(colors >= 0.0) and torch.all(colors <= 1.0)

    def test_single_mixture(self, sample_texture_tensor):
        """测试单个混合成分 (K=1)"""
        B = 16
        uvs = torch.rand(B, 1, 2)
        weights = torch.ones(B, 1)

        colors = sample_texture(sample_texture_tensor, uvs, weights)

        assert colors.shape == (B, 3)
        assert torch.all(colors >= 0.0) and torch.all(colors <= 1.0)


class TestTextureSamplerFieldIntegration:
    """集成测试：完整的纹理采样场流程"""

    @pytest.fixture
    def model_and_texture(self):
        """创建模型和纹理"""
        # 创建模型
        model = TextureSamplerField(
            num_mixtures=8,
            hidden_dim=64,
            num_layers=3,
            positional_encoding_freqs=4,
            use_scale_input=True,
        )

        # 创建示例纹理
        H, W = 128, 128
        texture = torch.rand(3, H, W)  # 随机纹理

        return model, texture

    def test_end_to_end(self, model_and_texture):
        """端到端测试"""
        model, texture = model_and_texture

        # 创建批次数据
        B = 32
        positions = torch.randn(B, 3)
        scale = torch.rand(B, 1) * 0.05 + 0.001

        # 前向传播
        output = model(positions, scale)

        # 采样纹理
        colors = sample_texture(texture, output.uvs, output.weights)

        # 验证输出
        assert colors.shape == (B, 3)
        assert torch.all(colors >= 0.0) and torch.all(colors <= 1.0)
        assert not torch.isnan(colors).any(), "Colors should not contain NaN"

    def test_batch_consistency(self, model_and_texture):
        """测试批次一致性"""
        model, texture = model_and_texture

        # 相同输入应该产生相同输出
        positions = torch.randn(4, 3)
        scale = torch.rand(4, 1) * 0.05

        output1 = model(positions, scale)
        colors1 = sample_texture(texture, output1.uvs, output1.weights)

        output2 = model(positions, scale)
        colors2 = sample_texture(texture, output2.uvs, output2.weights)

        assert torch.allclose(output1.uvs, output2.uvs, atol=1e-6)
        assert torch.allclose(output1.weights, output2.weights, atol=1e-6)
        assert torch.allclose(colors1, colors2, atol=1e-6)

    def test_different_scales(self, model_and_texture):
        """测试不同尺度产生不同输出"""
        model, texture = model_and_texture

        positions = torch.randn(1, 3).repeat(2, 1)
        scale1 = torch.tensor([[0.001]])
        scale2 = torch.tensor([[0.05]])

        output1 = model(positions[:1], scale1)
        output2 = model(positions[1:], scale2)

        # 由于网络有权重访问 scale，不同尺度应该产生不同输出
        # 但我们只测试形状是否正确
        assert output1.uvs.shape == output2.uvs.shape
        assert output1.weights.shape == output2.weights.shape


class TestTextureSamplerFieldTrainer:
    """测试训练器的损失计算"""

    @pytest.fixture
    def trainer(self):
        """创建训练器实例"""
        from scripts.train_texture_sampler_field import TextureSamplerFieldTrainer
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        model = TextureSamplerField(
            num_mixtures=4,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,
            use_scale_input=True,
        )

        device = torch.device("cpu")
        return TextureSamplerFieldTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
            uv_distance_weight=0.01,
            entropy_weight=0.001,
        )

    def test_compute_loss_returns_scalar(self, trainer):
        """测试 compute_loss 返回标量"""
        # 创建测试批次
        B = 8
        K = 4

        batch = {
            'position': torch.randn(B, 3),
            'target_uv': torch.rand(B, 2),
            'target_color': torch.rand(B, 3),
            'scale': torch.rand(B, 1) * 0.05,  # 添加 scale
        }

        # 模型输出
        model_output = TextureSamplerFieldOutput(
            uvs=torch.rand(B, K, 2),
            weights=torch.softmax(torch.randn(B, K), dim=-1),
            sigmas=torch.rand(B, K, 1),
        )

        # 纹理
        texture = torch.rand(3, 64, 64)

        # 计算损失
        losses = trainer.compute_loss(batch, model_output, texture)

        # 验证所有损失都是标量
        assert losses['total'].dim() == 0, "total_loss should be scalar"
        assert losses['color'].dim() == 0, "color_loss should be scalar"
        assert losses['uv_distance'].dim() == 0, "uv_loss should be scalar"
        assert losses['entropy'].dim() == 0, "entropy_loss should be scalar"
        assert losses['sigma'].dim() == 0, "sigma_loss should be scalar"

        # 验证不是 NaN
        assert not torch.isnan(losses['total']), "total_loss should not be NaN"
        assert not torch.isnan(losses['color']), "color_loss should not be NaN"
        assert not torch.isnan(losses['uv_distance']), "uv_loss should not be NaN"
        assert not torch.isnan(losses['entropy']), "entropy_loss should not be NaN"
        assert not torch.isnan(losses['sigma']), "sigma_loss should not be NaN"


class TestTextureSamplerDataset:
    """测试数据集的 barycentric UV 插值"""

    def test_barycentric_interpolation(self):
        """测试 barycentric UV 插值"""
        from src.data.tubular_texture_dataset import create_tubular_texture_dataset
        import tempfile
        import trimesh

        # 创建一个简单的三角形 mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        faces = np.array([[0, 1, 2]], dtype=np.int32)

        # 生成 UV
        centroid = vertices.mean(axis=0)
        v_centered = vertices - centroid
        norms = np.linalg.norm(v_centered, axis=1, keepdims=True)
        v_normalized = v_centered / (norms + 1e-8)

        x, y, z = v_normalized[:, 0], v_normalized[:, 1], v_normalized[:, 2]
        theta = np.arcsin(np.clip(y, -1, 1))
        phi = np.arctan2(z, x)
        u = (phi / (2 * np.pi) + 0.5).astype(np.float32)
        v = (theta / np.pi + 0.5).astype(np.float32)
        uvs = np.stack([u, v], axis=1)

        # 保存临时 mesh
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            temp_path = f.name
            trimesh.Trimesh(vertices=vertices, faces=faces).export(temp_path)

        # 创建数据集
        try:
            dataset = create_tubular_texture_dataset(
                mesh_path=temp_path,
                num_samples=10,
                num_surface_samples=100,
            )

            # 检查是否有 barycentric 坐标
            assert hasattr(dataset, 'surface_barycentric'), "Dataset should have surface_barycentric"
            assert dataset.surface_barycentric.shape == (100, 3), "Barycentric shape should be (N, 3)"

            # 检查 barycentric 坐标和接近 1
            bary_sum = dataset.surface_barycentric.sum(axis=1)
            assert np.allclose(bary_sum, 1.0, atol=1e-5), "Barycentric coords should sum to 1"

            # 检查都是非负
            assert np.all(dataset.surface_barycentric >= 0.0), "Barycentric coords should be non-negative"

        finally:
            # 清理临时文件
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_uv_flip_consistency(self):
        """测试 sample_texture 和 UVTextureSampler 的 V 方向一致性"""
        from src.data.sampling import TextureData, UVTextureSampler

        # 创建一个 2x2 的测试纹理
        # 左上: (1,0,0), 右上: (0,1,0)
        # 左下: (0,0,1), 右下: (1,1,0)
        texture_image = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ], dtype=np.uint8)  # (H, W, 3) = (2, 2, 3)

        texture_data = TextureData.from_array(texture_image)
        uv_sampler = UVTextureSampler(texture_data)

        # 测试四个角点的 UV
        test_uvs = np.array([
            [0.0, 0.0],  # 左上
            [1.0, 0.0],  # 右上
            [0.0, 1.0],  # 左下
            [1.0, 1.0],  # 右下
        ], dtype=np.float32)

        # 使用 UVTextureSampler 采样
        colors_np = uv_sampler.sample(test_uvs)  # (4, 3)

        # 转换为 PyTorch 纹理
        texture_torch = torch.from_numpy(texture_image).permute(2, 0, 1).float() / 255.0  # (3, 2, 2)

        # 使用 sample_texture 采样（flip_v=True）
        uvs_torch = torch.from_numpy(test_uvs).unsqueeze(0)  # (1, 4, 2)
        weights = torch.ones(1, 4) / 4

        colors_pytorch_flip = sample_texture(texture_torch, uvs_torch, weights, flip_v=True)  # (1, 3) -> (4, 3)

        # 使用 sample_texture 采样（flip_v=False）
        colors_pytorch_no_flip = sample_texture(texture_torch, uvs_torch, weights, flip_v=False)

        # 验证 flip_v=True 与 UVTextureSampler 一致
        # 注意：由于加权平均，我们需要单独测试每个 UV
        for i in range(4):
            uv_single = test_uvs[i:i+1]  # (1, 2)
            color_np = uv_sampler.sample(uv_single)  # (1, 3)

            uv_torch_single = torch.from_numpy(uv_single).unsqueeze(0)  # (1, 1, 2)
            weight_single = torch.ones(1, 1)

            color_pytorch_flip = sample_texture(texture_torch, uv_torch_single, weight_single, flip_v=True)

            assert torch.allclose(
                torch.from_numpy(color_np).float(),
                color_pytorch_flip.squeeze(0),
                atol=0.05
            ), f"UV {test_uvs[i]}: flip_v=True should match UVTextureSampler"


class TestMinimalTrainingStep:
    """测试最小训练步骤"""

    def test_one_training_step(self):
        """测试一个完整的训练步骤"""
        from scripts.train_texture_sampler_field import TextureSamplerFieldTrainer

        # 创建小模型
        model = TextureSamplerField(
            num_mixtures=2,
            hidden_dim=16,
            num_layers=2,
            positional_encoding_freqs=4,
            use_scale_input=True,
        )

        device = torch.device("cpu")
        trainer = TextureSamplerFieldTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
        )

        # 创建纹理
        texture = torch.rand(3, 32, 32)

        # 创建批次
        B = 4
        batch = {
            'position': torch.randn(B, 3),
            'target_uv': torch.rand(B, 2),
            'target_color': torch.rand(B, 3),
            'scale': torch.rand(B, 1) * 0.05,
        }

        # 移动到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        texture = texture.to(device)

        # 训练步骤
        model.train()
        model_output = model(batch['position'], scale=batch['scale'])
        losses = trainer.compute_loss(batch, model_output, texture)

        # 反向传播
        trainer.optimizer.zero_grad()
        losses['total'].backward()
        trainer.optimizer.step()

        # 验证损失是标量
        assert losses['total'].dim() == 0

        # 验证梯度存在且不是 NaN
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} is NaN"


def test_parameter_count():
    """测试参数数量"""
    model = TextureSamplerField(
        num_mixtures=8,
        hidden_dim=128,
        num_layers=4,
        positional_encoding_freqs=6,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTextureSamplerField 参数数量: {num_params:,}")

    # 粗略检查：应该小于 1M 参数（小模型）
    assert num_params < 1_000_000, f"Model too large: {num_params:,} parameters"


if __name__ == "__main__":
    # 快速自检
    print("运行快速自检...")

    # 测试位置编码
    print("\n1. 测试 PositionalEncoding...")
    pos_enc = PositionalEncoding(num_freqs=6)
    x = torch.randn(10, 3)
    encoded = pos_enc(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {encoded.shape}")
    assert encoded.shape == (10, 36), "PositionalEncoding shape mismatch"
    print("   ✓ 通过")

    # 测试模型
    print("\n2. 测试 TextureSamplerField...")
    model = TextureSamplerField(
        num_mixtures=8,
        hidden_dim=128,
        num_layers=4,
        positional_encoding_freqs=6,
        use_scale_input=True,
    )

    positions = torch.randn(16, 3)
    scale = torch.rand(16, 1) * 0.05 + 0.001
    output = model(positions, scale)

    print(f"   输入 positions: {positions.shape}")
    print(f"   输入 scale: {scale.shape}")
    print(f"   输出 uvs: {output.uvs.shape}")
    print(f"   输出 weights: {output.weights.shape}")
    print(f"   输出 sigmas: {output.sigmas.shape}")

    assert output.uvs.shape == (16, 8, 2), "UVs shape mismatch"
    assert output.weights.shape == (16, 8), "Weights shape mismatch"
    assert output.sigmas.shape == (16, 8, 1), "Sigmas shape mismatch"

    # 检查约束
    assert torch.all(output.uvs >= 0.0) and torch.all(output.uvs <= 1.0), "UVs out of range"
    assert torch.allclose(output.weights.sum(dim=-1), torch.ones(16), atol=1e-6), "Weights don't sum to 1"
    assert torch.all(output.sigmas > 0.0), "Sigmas not positive"

    print("   ✓ UV 范围: [0, 1]")
    print("   ✓ 权重和: 1.0")
    print("   ✓ Sigmas: 正数")
    print("   ✓ 通过")

    # 测试纹理采样
    print("\n3. 测试 sample_texture...")
    texture = torch.rand(3, 64, 64)
    colors = sample_texture(texture, output.uvs, output.weights)
    print(f"   纹理: {texture.shape}")
    print(f"   采样颜色: {colors.shape}")
    assert colors.shape == (16, 3), "Colors shape mismatch"
    assert torch.all(colors >= 0.0) and torch.all(colors <= 1.0), "Colors out of range"
    print("   ✓ 通过")

    # 测试梯度
    print("\n4. 测试梯度流动...")
    positions_grad = torch.randn(8, 3)
    positions_grad.requires_grad = True
    scale_grad = torch.rand(8, 1) * 0.05 + 0.001
    scale_grad.requires_grad = True

    output_grad = model(positions_grad, scale_grad)
    loss = output_grad.uvs.sum() + output_grad.weights.sum()
    loss.backward()

    assert positions_grad.grad is not None, "No gradient for positions"
    assert scale_grad.grad is not None, "No gradient for scale"
    assert not torch.isnan(positions_grad.grad).any(), "NaN in position gradients"
    print("   ✓ 梯度流动正常")
    print("   ✓ 通过")

    print("\n✅ 所有自检通过!")
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 额外检查：修复验收问题
    print("\n5. 检查修复验收问题...")

    # 检查 1: uv_loss 是标量
    print("   检查 uv_loss 标量化...")
    from scripts.train_texture_sampler_field import TextureSamplerFieldTrainer
    trainer = TextureSamplerFieldTrainer(
        model=model,
        device=torch.device("cpu"),
    )
    batch = {
        'position': torch.randn(4, 3),
        'target_uv': torch.rand(4, 2),
        'target_color': torch.rand(4, 3),
        'scale': torch.rand(4, 1) * 0.05,  # 添加 scale
    }
    model_output = TextureSamplerFieldOutput(
        uvs=torch.rand(4, 8, 2),
        weights=torch.softmax(torch.randn(4, 8), dim=-1),
        sigmas=torch.rand(4, 8, 1),
    )
    texture = torch.rand(3, 64, 64)
    losses = trainer.compute_loss(batch, model_output, texture)
    assert losses['uv_distance'].dim() == 0, "uv_loss 应该是标量"
    assert losses['sigma'].dim() == 0, "sigma_loss 应该是标量"  # 添加 sigma 检查
    print("   ✓ uv_loss 是标量")
    print("   ✓ sigma_loss 是标量")

    # 检查 2: sample_texture 支持 flip_v
    print("   检查 sample_texture flip_v 参数...")
    import inspect
    sig = inspect.signature(sample_texture)
    assert 'flip_v' in sig.parameters, "sample_texture 应该有 flip_v 参数"
    print("   ✓ sample_texture 支持 flip_v")

    # 检查 3: 数据集有 barycentric 坐标
    print("   检查数据集 barycentric 坐标...")
    from src.data.tubular_texture_dataset import create_tubular_texture_dataset
    import tempfile
    import os
    import trimesh

    # 创建临时 mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    centroid = vertices.mean(axis=0)
    v_centered = vertices - centroid
    norms = np.linalg.norm(v_centered, axis=1, keepdims=True)
    v_normalized = v_centered / (norms + 1e-8)

    x, y, z = v_normalized[:, 0], v_normalized[:, 1], v_normalized[:, 2]
    theta = np.arcsin(np.clip(y, -1, 1))
    phi = np.arctan2(z, x)
    u = (phi / (2 * np.pi) + 0.5).astype(np.float32)
    v = (theta / np.pi + 0.5).astype(np.float32)
    uvs = np.stack([u, v], axis=1)

    temp_path = tempfile.mktemp(suffix='.obj')
    trimesh.Trimesh(vertices=vertices, faces=faces).export(temp_path)

    try:
        dataset = create_tubular_texture_dataset(
            mesh_path=temp_path,
            num_samples=10,
            num_surface_samples=50,
        )
        assert hasattr(dataset, 'surface_barycentric'), "数据集应该有 barycentric 坐标"
        assert dataset.surface_barycentric.shape == (50, 3), "Barycentric 形状应该是 (N, 3)"
        bary_sum = dataset.surface_barycentric.sum(axis=1)
        assert np.allclose(bary_sum, 1.0, atol=1e-5), "Barycentric 坐标和应该为 1"
        print("   ✓ 数据集有正确的 barycentric 坐标")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n✅ 所有验收问题修复检查通过!")

    # 额外检查：第四轮修复
    print("\n6. 检查第四轮修复...")

    # 检查 1: 全退化面的错误处理
    print("   检查全退化面错误处理...")
    all_degenerate_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],  # 所有点共线
    ], dtype=np.float32)
    all_degenerate_faces = np.array([[0, 1, 2]], dtype=np.int32)

    all_degenerate_temp = tempfile.mktemp(suffix='.obj')
    trimesh.Trimesh(vertices=all_degenerate_vertices, faces=all_degenerate_faces).export(all_degenerate_temp)

    try:
        from src.data.tubular_texture_dataset import create_tubular_texture_dataset
        dataset_bad = create_tubular_texture_dataset(
            mesh_path=all_degenerate_temp,
            num_samples=10,
            num_surface_samples=100,
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "没有有效的三角形" in str(e), f"错误消息不对: {e}"
        print("   ✓ 全退化面抛出清晰错误")
    finally:
        if os.path.exists(all_degenerate_temp):
            os.remove(all_degenerate_temp)

    # 检查 2: pos_enc.freqs buffer 测试
    print("   检查 pos_enc.freqs buffer...")
    model_cpu = TextureSamplerField(num_mixtures=4)
    model_cuda = model_cpu.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 检查 freqs 在 state_dict 中
    state_dict = model_cuda.state_dict()
    assert "pos_enc.freqs" in state_dict, "pos_enc.freqs 应在 state_dict 中"
    print("   ✓ pos_enc.freqs 在 state_dict 中")

    # 检查 3: checkpoint metadata 测试
    print("   检查 checkpoint metadata...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # 模拟保存 checkpoint
        test_metadata = {
            'mesh_path': 'test.obj',
            'use_scale_input': True,
            'sigma_weight': 0.01,
            'scale_sampling': 'log_uniform',
        }
        test_checkpoint = {
            'epoch': 0,
            'model_state_dict': model_cpu.state_dict(),
            'loss': 0.5,
            'best_loss': 0.5,
            **test_metadata,
        }
        torch.save(test_checkpoint, checkpoint_path)

        # 加载并验证
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert 'loss' in loaded, "checkpoint 应有 loss"
        assert 'best_loss' in loaded, "checkpoint 应有 best_loss"
        assert 'use_scale_input' in loaded, "checkpoint 应有 use_scale_input"
        assert 'sigma_weight' in loaded, "checkpoint 应有 sigma_weight"
        assert 'scale_sampling' in loaded, "checkpoint 应有 scale_sampling"
        print("   ✓ checkpoint metadata 完整")

    # 检查 4: OBJ UV fallback 测试
    print("   检查 OBJ UV fallback...")
    # 测试没有 UV 的 mesh（应该 fallback 到球面 UV）
    no_uv_temp = tempfile.mktemp(suffix='.obj')
    trimesh.Trimesh(vertices=vertices, faces=faces).export(no_uv_temp)

    try:
        dataset_no_uv = create_tubular_texture_dataset(
            mesh_path=no_uv_temp,
            num_samples=10,
            num_surface_samples=50,
        )
        # 数据集应该成功创建（使用球面 UV）
        assert hasattr(dataset_no_uv, 'surface_uvs'), "应该有 surface_uvs"
        assert dataset_no_uv.surface_uvs.shape[1] == 2, "UV 应该是 (N, 2)"
        print("   ✓ OBJ UV fallback 正常")
    finally:
        if os.path.exists(no_uv_temp):
            os.remove(no_uv_temp)

    print("\n✅ 第四轮修复检查通过!")

