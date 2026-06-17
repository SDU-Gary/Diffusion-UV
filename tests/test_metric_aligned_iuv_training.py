"""
MA-IUVF 训练测试

测试loss计算和训练流程
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.metric_aligned_iuv_field import create_model, MetricAlignedIUVOutput
from src.training.metric_aligned_iuv_losses import (
    gather_chart_uvs,
    compute_uv_jacobian,
    compute_metric_loss,
    compute_anchor_loss,
    compute_chart_com_loss,
    compute_classification_loss,
    compute_metric_aligned_iuv_loss,
    validate_jacobian_math,
    validate_normal_zero_grad,
)


class TestLossComponents:
    """测试loss组件"""

    def test_gather_chart_uvs(self):
        """测试UV收集"""
        # 模拟输出
        uv_preds = torch.randn(10, 3, 2)  # [B, C, 2]
        chart_id = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # [B]

        # 收集
        selected_uv = gather_chart_uvs(uv_preds, chart_id)

        # 验证shape
        assert selected_uv.shape == (10, 2), f"收集UV shape错误: {selected_uv.shape}"

        # 验证内容：应该从对应chart收集
        for i in range(10):
            expected = uv_preds[i, chart_id[i]]
            assert torch.allclose(selected_uv[i], expected), \
                f"收集的UV不正确: index={i}"

    def test_jacobian_computation(self):
        """测试雅可比计算"""
        # 创建简单模型
        model = create_model(num_charts=3, hidden_dim=32)

        # 输入位置
        pos = torch.randn(5, 3, requires_grad=True)

        # 前向
        output = model(pos)

        # 选择chart
        chart_id = torch.zeros(5, dtype=torch.long)  # 全部选择chart 0

        # 收集UV
        selected_uv = gather_chart_uvs(output.uv_preds, chart_id)

        # 计算雅可比
        jacobian = compute_uv_jacobian(selected_uv, pos)

        # 验证shape
        assert jacobian.shape == (5, 2, 3), f"雅可比shape错误: {jacobian.shape}"

        # 验证无NaN
        assert not torch.isnan(jacobian).any(), "雅可比包含NaN"
        assert not torch.isinf(jacobian).any(), "雅可比包含Inf"

    def test_metric_loss(self):
        """测试metric loss"""
        j_pred = torch.randn(10, 2, 3)
        j_gt = torch.randn(10, 2, 3)

        loss = compute_metric_loss(j_pred, j_gt)

        # 验证loss类型
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # 标量
        assert not torch.isnan(loss), "metric loss包含NaN"

    def test_anchor_loss(self):
        """测试anchor loss"""
        uv_pred = torch.rand(10, 2)
        uv_anchor = torch.rand(10, 2)

        loss = compute_anchor_loss(uv_pred, uv_anchor)

        # 验证loss类型
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss), "anchor loss包含NaN"

    def test_chart_com_loss_value(self):
        """测试 chart-wise 质心 loss 数值"""
        uv_pred = torch.tensor([
            [1.0, 1.0],
            [3.0, 1.0],
            [10.0, 0.0],
            [12.0, 0.0],
        ])
        uv_anchor = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
        ])
        chart_id = torch.tensor([0, 0, 1, 1])

        # chart 0: mean delta = [1, 1], norm^2 = 2
        # chart 1: mean delta = [3, 0], norm^2 = 9
        # average = 5.5
        loss = compute_chart_com_loss(uv_pred, uv_anchor, chart_id, num_charts=3)
        assert torch.allclose(loss, torch.tensor(5.5)), f"CoM loss错误: {loss}"

    def test_chart_com_loss_uniform_chart_gradient(self):
        """测试质心 loss 对同一 chart 内样本施加均匀平移梯度"""
        uv_pred = torch.tensor([
            [1.0, 1.0],
            [3.0, 1.0],
            [10.0, 0.0],
            [12.0, 0.0],
        ], requires_grad=True)
        uv_anchor = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
        ])
        chart_id = torch.tensor([0, 0, 1, 1])

        loss = compute_chart_com_loss(uv_pred, uv_anchor, chart_id, num_charts=2)
        loss.backward()

        assert torch.allclose(uv_pred.grad[0], uv_pred.grad[1])
        assert torch.allclose(uv_pred.grad[2], uv_pred.grad[3])
        assert torch.allclose(uv_pred.grad[0], torch.tensor([0.5, 0.5]))
        assert torch.allclose(uv_pred.grad[2], torch.tensor([1.5, 0.0]))

    def test_classification_loss(self):
        """测试分类loss"""
        logits = torch.randn(10, 5)
        chart_id = torch.randint(0, 5, (10,))

        loss = compute_classification_loss(logits, chart_id)

        # 验证loss类型
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss), "classification loss包含NaN"


class TestFullLoss:
    """测试完整loss计算"""

    def test_loss_dict_structure(self):
        """测试loss字典结构"""
        # 创建模型
        model = create_model(num_charts=3, hidden_dim=32)

        # 模拟数据
        pos = torch.randn(10, 3, requires_grad=True)
        j_3d_gt = torch.randn(10, 2, 3)
        uv_anchor = torch.rand(10, 2)
        chart_id = torch.randint(0, 3, (10,))

        # 前向
        output = model(pos)

        # 计算loss
        loss_dict = compute_metric_aligned_iuv_loss(
            model_output=output,
            pos=pos,
            j_3d_gt=j_3d_gt,
            uv_anchor=uv_anchor,
            chart_id=chart_id,
            metric_weight=1.0,
            anchor_weight=1e-4,
            cls_weight=0.1,
        )

        # 验证loss字典结构
        required_keys = ['total', 'metric', 'anchor', 'anchor_weighted', 'com', 'com_weighted', 'cls']
        for key in required_keys:
            assert key in loss_dict, f"loss dict缺少key: {key}"
            assert isinstance(loss_dict[key], torch.Tensor), f"{key}应该是tensor"
            assert loss_dict[key].dim() == 0, f"{key}应该是标量"

    def test_loss_backward(self):
        """测试loss可以反向传播"""
        # 创建模型
        model = create_model(num_charts=3, hidden_dim=32)

        # 模拟数据
        pos = torch.randn(10, 3, requires_grad=True)
        j_3d_gt = torch.randn(10, 2, 3)
        uv_anchor = torch.rand(10, 2)
        chart_id = torch.randint(0, 3, (10,))

        # 前向
        output = model(pos)

        # 计算loss
        loss_dict = compute_metric_aligned_iuv_loss(
            model_output=output,
            pos=pos,
            j_3d_gt=j_3d_gt,
            uv_anchor=uv_anchor,
            chart_id=chart_id,
        )

        # 反向传播
        loss_dict['total'].backward()

        # 验证梯度
        assert pos.grad is not None, "pos应该有梯度"
        assert not torch.isnan(pos.grad).any(), "pos梯度包含NaN"

        # 验证模型梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name}应该有梯度"
                assert not torch.isnan(param.grad).any(), f"{name}梯度包含NaN"

    def test_loss_no_nan(self):
        """测试loss不包含NaN"""
        model = create_model(num_charts=3, hidden_dim=32)

        pos = torch.randn(10, 3, requires_grad=True)
        j_3d_gt = torch.randn(10, 2, 3)
        uv_anchor = torch.rand(10, 2)
        chart_id = torch.randint(0, 3, (10,))

        output = model(pos)

        loss_dict = compute_metric_aligned_iuv_loss(
            model_output=output,
            pos=pos,
            j_3d_gt=j_3d_gt,
            uv_anchor=uv_anchor,
            chart_id=chart_id,
        )

        # 检查所有loss不包含NaN
        for key, value in loss_dict.items():
            assert not torch.isnan(value), f"{key}包含NaN"
            assert not torch.isinf(value), f"{key}包含Inf"


class TestSyntheticOverfit:
    """测试合成数据过拟合"""

    def test_simple_overfit(self):
        """测试简单数据过拟合：metric loss应该下降"""
        # 创建极简数据：单个三角形
        # 平面三角形：v0=(0,0,0), v1=(1,0,0), v2=(0,1,0)
        # UV：uv0=(0,0), uv1=(2,0), uv2=(0,3)

        from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker

        # 构造单三角形数据
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

        # 计算GT雅可比
        baker = MetricAlignedIUVBaker.__new__(MetricAlignedIUVBaker)
        j_3d_gt = baker._compute_triangle_jacobian(vertices, uvs)

        # 生成样本（在三角形附近）
        num_samples = 20
        # 采样重心坐标
        bary_coords = torch.randn(num_samples, 3)
        bary_coords = torch.abs(bary_coords) / torch.abs(bary_coords).sum(dim=1, keepdim=True)

        # 表面位置
        pos = (bary_coords @ vertices).detach()  # [N, 3]
        pos = pos + torch.randn_like(pos) * 0.01  # 加噪声

        # 计算每个采样点的UV anchor（根据重心坐标插值）
        uv_anchor_expanded = (uvs.unsqueeze(0).expand(num_samples, -1, 2) * bary_coords.unsqueeze(-1)).sum(dim=1)  # [N, 2]

        # 扩展数据
        j_3d_gt_expanded = j_3d_gt.unsqueeze(0).expand(num_samples, -1, -1)
        chart_id = torch.zeros(num_samples, dtype=torch.long)

        # 创建模型（单chart）
        model = create_model(num_charts=1, hidden_dim=32, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 训练
        initial_loss = None
        final_loss = None

        for step in range(50):
            pos_req = pos.clone().detach().requires_grad_(True)

            output = model(pos_req)

            loss_dict = compute_metric_aligned_iuv_loss(
                model_output=output,
                pos=pos_req,
                j_3d_gt=j_3d_gt_expanded,
                uv_anchor=uv_anchor_expanded,
                chart_id=chart_id,
                metric_weight=1.0,
                anchor_weight=1e-4,
                cls_weight=0.0,  # 单chart不需要分类loss
            )

            loss = loss_dict['total']

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 49:
                final_loss = loss.item()

        # 验证loss下降
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss: {final_loss:.6f}")

        assert final_loss < initial_loss, \
            f"Loss应该下降: initial={initial_loss:.6f}, final={final_loss:.6f}"

        # 至少下降50%
        assert final_loss < initial_loss * 0.5, \
            f"Loss应该显著下降: {initial_loss:.6f} -> {final_loss:.6f}"


class TestMathValidation:
    """测试数学验证函数"""

    def test_validate_jacobian_math(self):
        """测试雅可比数学验证"""
        result = validate_jacobian_math()
        assert result == True, "雅可比数学验证失败"

    def test_validate_normal_zero_grad(self):
        """测试法向零梯度验证"""
        result = validate_normal_zero_grad()
        assert result == True, "法向零梯度验证失败"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
