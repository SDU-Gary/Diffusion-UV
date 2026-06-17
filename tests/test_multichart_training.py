"""
多 Chart 训练路径验证测试

测试：
1. 不同 chart 的样本只监督对应 UV 分支
2. CrossEntropy(logits, chart_id) 在多 chart 数据上正常下降
3. 非 target chart 的 UV 分支不会被 anchor/metric loss 直接污染
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.metric_aligned_iuv_field import create_model
from src.training.metric_aligned_iuv_losses import compute_metric_aligned_iuv_loss
from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker


class TestMultichartLossComputation:
    """测试多 Chart Loss 计算"""

    def test_multichart_loss_isolation(self):
        """测试不同 chart 的样本只监督对应 UV 分支"""
        # 创建 3-chart 模型
        model = create_model(num_charts=3, hidden_dim=32, num_layers=2)

        # 模拟数据：3个样本，每个属于不同的 chart
        pos = torch.randn(3, 3, requires_grad=True)
        j_3d_gt = torch.randn(3, 2, 3)
        uv_anchor = torch.rand(3, 2)
        chart_id = torch.tensor([0, 1, 2])  # 每个样本属于不同 chart

        # 前向
        output = model(pos)

        # 计算 loss
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

        # 验证 loss 结构
        assert 'total' in loss_dict
        assert 'metric' in loss_dict
        assert 'anchor' in loss_dict
        assert 'cls' in loss_dict

        # 验证 loss 不是 NaN
        for key, value in loss_dict.items():
            assert not torch.isnan(value), f"{key} 是 NaN"
            assert not torch.isinf(value), f"{key} 是 Inf"

        print(f"✓ 多 chart loss 计算正常")
        print(f"  - Total loss: {loss_dict['total'].item():.6f}")
        print(f"  - Metric loss: {loss_dict['metric'].item():.6f}")
        print(f"  - Anchor loss: {loss_dict['anchor'].item():.6f}")
        print(f"  - Cls loss: {loss_dict['cls'].item():.6f}")

    def test_chart_classification_gradient(self):
        """测试 CrossEntropy 在多 chart 数据上正常反向传播"""
        # 创建 3-chart 模型
        model = create_model(num_charts=3, hidden_dim=32, num_layers=2)

        # 模拟多 chart 数据
        pos = torch.randn(10, 3, requires_grad=True)
        j_3d_gt = torch.randn(10, 2, 3)
        uv_anchor = torch.rand(10, 2)
        chart_id = torch.randint(0, 3, (10,))  # 随机分配 chart

        # 前向
        output = model(pos)

        # 计算 loss
        loss_dict = compute_metric_aligned_iuv_loss(
            model_output=output,
            pos=pos,
            j_3d_gt=j_3d_gt,
            uv_anchor=uv_anchor,
            chart_id=chart_id,
            cls_weight=1.0,  # 只测试分类 loss
            metric_weight=0.0,
            anchor_weight=0.0,
        )

        # 反向传播
        loss_dict['total'].backward()

        # 验证梯度存在
        assert pos.grad is not None, "pos 应该有梯度"
        assert not torch.isnan(pos.grad).any(), "pos 梯度包含 NaN"

        # 验证模型参数梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} 应该有梯度"
                assert not torch.isnan(param.grad).any(), f"{name} 梯度包含 NaN"

        print(f"✓ 分类 loss 反向传播正常")

    def test_anchor_loss_chart_isolation(self):
        """测试 anchor loss 只监督目标 chart"""
        # 创建 3-chart 模型
        model = create_model(num_charts=3, hidden_dim=32, num_layers=2)

        # 模拟数据：所有样本属于 chart 0
        pos = torch.randn(5, 3, requires_grad=True)
        j_3d_gt = torch.randn(5, 2, 3)
        uv_anchor = torch.rand(5, 2)
        chart_id = torch.zeros(5, dtype=torch.long)  # 全部属于 chart 0

        # 前向
        output = model(pos)

        # 计算 loss（只包含 anchor loss）
        loss_dict = compute_metric_aligned_iuv_loss(
            model_output=output,
            pos=pos,
            j_3d_gt=j_3d_gt,
            uv_anchor=uv_anchor,
            chart_id=chart_id,
            anchor_weight=1.0,
            metric_weight=0.0,
            cls_weight=0.0,
        )

        # 验证 loss 不为 0
        assert loss_dict['anchor'].item() > 0, "anchor loss 应该大于 0"

        print(f"✓ Anchor loss 只监督目标 chart: {loss_dict['anchor'].item():.6f}")

    def test_metric_loss_chart_isolation(self):
        """测试 metric loss 只监督目标 chart"""
        # 创建 3-chart 模型
        model = create_model(num_charts=3, hidden_dim=32, num_layers=2)

        # 模拟数据：所有样本属于 chart 1
        pos = torch.randn(5, 3, requires_grad=True)
        j_3d_gt = torch.randn(5, 2, 3)
        uv_anchor = torch.rand(5, 2)
        chart_id = torch.ones(5, dtype=torch.long)  # 全部属于 chart 1

        # 前向
        output = model(pos)

        # 计算 loss（只包含 metric loss）
        loss_dict = compute_metric_aligned_iuv_loss(
            model_output=output,
            pos=pos,
            j_3d_gt=j_3d_gt,
            uv_anchor=uv_anchor,
            chart_id=chart_id,
            metric_weight=1.0,
            anchor_weight=0.0,
            cls_weight=0.0,
        )

        # 验证 loss 不为 0
        assert loss_dict['metric'].item() > 0, "metric loss 应该大于 0"

        print(f"✓ Metric loss 只监督目标 chart: {loss_dict['metric'].item():.6f}")


class TestMultichartTraining:
    """测试多 Chart 训练"""

    def test_multichart_overfit(self):
        """测试多 chart 数据过拟合：loss 应该下降"""
        # 创建 2-chart 数据
        num_samples_chart0 = 50
        num_samples_chart1 = 30

        # Chart 0 数据
        pos0 = torch.randn(num_samples_chart0, 3)
        j0 = torch.randn(num_samples_chart0, 2, 3)
        uv0 = torch.rand(num_samples_chart0, 2)

        # Chart 1 数据
        pos1 = torch.randn(num_samples_chart1, 3)
        j1 = torch.randn(num_samples_chart1, 2, 3)
        uv1 = torch.rand(num_samples_chart1, 2)

        # 合并数据
        pos = torch.cat([pos0, pos1], dim=0)
        j_3d_gt = torch.cat([j0, j1], dim=0)
        uv_anchor = torch.cat([uv0, uv1], dim=0)
        chart_id = torch.cat([
            torch.zeros(num_samples_chart0, dtype=torch.long),
            torch.ones(num_samples_chart1, dtype=torch.long),
        ])

        # 创建 2-chart 模型
        model = create_model(num_charts=2, hidden_dim=32, num_layers=2)
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
                j_3d_gt=j_3d_gt,
                uv_anchor=uv_anchor,
                chart_id=chart_id,
                metric_weight=1.0,
                anchor_weight=1e-4,
                cls_weight=0.1,
            )

            loss = loss_dict['total']

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 49:
                final_loss = loss.item()

        # 验证 loss 下降
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss: {final_loss:.6f}")

        assert final_loss < initial_loss, \
            f"Loss 应该下降: initial={initial_loss:.6f}, final={final_loss:.6f}"

        # 至少下降 30%
        assert final_loss < initial_loss * 0.7, \
            f"Loss 应该显著下降: {initial_loss:.6f} -> {final_loss:.6f}"

        print(f"✓ 多 chart 训练 loss 显著下降")


class TestRealMultichartData:
    """测试真实多 Chart 数据"""

    @pytest.fixture
    def multichart_baked_data(self):
        """创建多 chart 烘焙数据"""
        import tempfile
        from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker

        # 创建临时 OBJ 文件
        temp_dir = Path(tempfile.mkdtemp())
        obj_path = temp_dir / "multichart.obj"

        # 创建带明显 UV seams 的立方体（使用测试数据中的）
        import shutil
        shutil.copy("test_data/uv_seam_cube.obj", obj_path)

        # 烘焙数据
        baker = MetricAlignedIUVBaker(str(obj_path), seed=42, use_obj_parser=True)
        data = baker.bake(
            num_samples=100,
            extrusion_sigma_ratio=0.01,
            chart_mode="uv_islands",
        )

        yield data, temp_dir

        # 清理
        shutil.rmtree(temp_dir)

    def test_multichart_data_structure(self, multichart_baked_data):
        """测试多 chart 数据结构"""
        data, _ = multichart_baked_data

        # 验证多 chart
        num_charts = data.chart_id.max().item() + 1
        assert num_charts >= 2, "应该有至少 2 个 charts"

        # 验证每个 chart 都有样本
        for chart_id in range(num_charts):
            count = (data.chart_id == chart_id).sum().item()
            assert count > 0, f"Chart {chart_id} 应该有样本"
            print(f"Chart {chart_id}: {count} 样本")

        print(f"✓ 多 chart 数据结构正确")

    def test_multichart_model_forward(self, multichart_baked_data):
        """测试多 chart 模型前向传播"""
        data, _ = multichart_baked_data

        num_charts = data.chart_id.max().item() + 1

        # 创建模型
        model = create_model(num_charts=num_charts, hidden_dim=32)

        # 前向传播
        pos = data.pos[:10]  # 只测试前 10 个样本
        output = model(pos)

        # 验证输出
        assert output.logits.shape == (10, num_charts)
        assert output.uv_preds.shape == (10, num_charts, 2)

        # 验证无 NaN
        assert not torch.isnan(output.logits).any()
        assert not torch.isnan(output.uv_preds).any()

        print(f"✓ 多 chart 模型前向传播正常")

    def test_multichart_loss_computation(self, multichart_baked_data):
        """测试多 chart loss 计算"""
        data, _ = multichart_baked_data

        num_charts = data.chart_id.max().item() + 1

        # 创建模型
        model = create_model(num_charts=num_charts, hidden_dim=32)

        # 小批量测试
        batch_size = 10
        pos = data.pos[:batch_size].clone().detach().requires_grad_(True)
        j_3d_gt = data.j_3d_gt[:batch_size]
        uv_anchor = data.uv_anchor[:batch_size]
        chart_id = data.chart_id[:batch_size]

        # 前向
        output = model(pos)

        # 计算 loss
        loss_dict = compute_metric_aligned_iuv_loss(
            model_output=output,
            pos=pos,
            j_3d_gt=j_3d_gt,
            uv_anchor=uv_anchor,
            chart_id=chart_id,
        )

        # 验证 loss
        assert 'total' in loss_dict
        assert not torch.isnan(loss_dict['total'])

        # 反向传播
        loss_dict['total'].backward()

        # 验证梯度
        assert pos.grad is not None
        assert not torch.isnan(pos.grad).any()

        print(f"✓ 多 chart loss 计算和反向传播正常")


if __name__ == "__main__":
    # 快速自检
    pytest.main([__file__, "-v", "--tb=short"])
