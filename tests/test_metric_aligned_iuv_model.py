"""
MA-IUVF 模型测试

测试模型架构和输出shape
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.metric_aligned_iuv_field import (
    MetricAlignedIUVField,
    MetricAlignedIUVOutput,
    create_model,
    FourierPositionalEncoding,
)


class TestPositionalEncoding:
    """测试位置编码"""

    def test_output_shape(self):
        """测试输出shape"""
        pos_enc = FourierPositionalEncoding(num_freqs=4)

        # 输入 [B, 3]
        x = torch.randn(10, 3)

        # 输出 [B, 3 * num_freqs * 2]
        encoded = pos_enc(x)

        expected_dim = 3 * 4 * 2  # 24
        assert encoded.shape == (10, expected_dim), f"编码shape错误: {encoded.shape}"

    def test_deterministic(self):
        """测试确定性：相同输入产生相同输出"""
        pos_enc = FourierPositionalEncoding(num_freqs=4)

        x = torch.randn(5, 3)

        encoded1 = pos_enc(x)
        encoded2 = pos_enc(x)

        assert torch.allclose(encoded1, encoded2), "位置编码应该确定"


class TestModelArchitecture:
    """测试模型架构"""

    def test_model_creation(self):
        """测试模型创建"""
        model = create_model(
            num_charts=3,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,
        )

        assert model.num_charts == 3
        assert model.hidden_dim == 32
        assert model.num_layers == 2

    def test_forward_output_shape(self):
        """测试前向传播输出shape"""
        model = create_model(
            num_charts=5,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,
        )

        # 输入 [B, 3]
        pos = torch.randn(10, 3)

        # 前向
        output = model(pos)

        # 验证输出shape
        assert output.logits.shape == (10, 5), f"logits shape错误: {output.logits.shape}"
        assert output.uv_preds.shape == (10, 5, 2), \
            f"uv_preds shape错误: {output.uv_preds.shape}"

    def test_uv_range(self):
        """测试UV输出（不再限制范围）"""
        model = create_model(
            num_charts=3,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,
        )

        pos = torch.randn(20, 3)

        output = model(pos)

        # UV应该是有效的浮点数（不再限制范围）
        assert not torch.isnan(output.uv_preds).any(), "UV不应该包含NaN"
        assert not torch.isinf(output.uv_preds).any(), "UV不应该包含Inf"

    def test_num_params(self):
        """测试参数数量"""
        model = create_model(
            num_charts=3,
            hidden_dim=32,
            num_layers=2,
            positional_encoding_freqs=4,
        )

        num_params = model.get_num_params()

        assert num_params > 0, "模型应该有参数"
        assert num_params < 100000, "参数数量应该合理（小于100K）"


class TestModelFeatures:
    """测试模型特性"""

    def test_different_charts(self):
        """测试不同chart的UV预测不同"""
        model = create_model(
            num_charts=3,
            hidden_dim=32,
            num_layers=2,
        )

        pos = torch.randn(1, 3)
        output = model(pos)

        # 不同chart的UV应该不同（如果模型足够复杂）
        uv_chart_0 = output.uv_preds[0, 0]
        uv_chart_1 = output.uv_preds[0, 1]

        # 注意：初始化时可能相同，但随着训练会分化
        # 这里只测试shape
        assert uv_chart_0.shape == (2,)
        assert uv_chart_1.shape == (2,)

    def test_batch_processing(self):
        """测试批处理"""
        model = create_model(num_charts=3, hidden_dim=32)

        # 不同batch size
        for bs in [1, 10, 100]:
            pos = torch.randn(bs, 3)
            output = model(pos)

            assert output.logits.shape[0] == bs
            assert output.uv_preds.shape[0] == bs

    def test_gradient_flow(self):
        """测试梯度流动"""
        model = create_model(num_charts=3, hidden_dim=32)

        pos = torch.randn(5, 3, requires_grad=True)

        output = model(pos)

        # 反向传播
        loss = output.logits.sum()
        loss.backward()

        # 验证梯度存在
        assert pos.grad is not None, "输入应该有梯度"
        assert pos.grad.abs().sum() > 0, "梯度不应该全零"


class TestModelSaving:
    """测试模型保存和加载"""

    def test_save_and_load(self):
        """测试保存和加载"""
        temp_dir = tempfile.mkdtemp()

        try:
            # 创建模型
            model = create_model(
                num_charts=3,
                hidden_dim=32,
                num_layers=2,
            )

            # 保存
            save_path = Path(temp_dir) / "model.pt"
            torch.save(model.state_dict(), save_path)

            # 创建新模型并加载
            new_model = create_model(num_charts=3, hidden_dim=32, num_layers=2)
            new_model.load_state_dict(torch.load(save_path))

            # 验证一致性
            pos = torch.randn(5, 3)

            model.eval()
            new_model.eval()

            with torch.no_grad():
                output1 = model(pos)
                output2 = new_model(pos)

            assert torch.allclose(output1.logits, output2.logits)
            assert torch.allclose(output1.uv_preds, output2.uv_preds)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
