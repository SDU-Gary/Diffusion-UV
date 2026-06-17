"""
测试完整元数据保存
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker, load_baked_data

def test_complete_metadata():
    """测试完整元数据保存和加载"""
    print("=== 测试完整元数据 ===\n")

    obj_path = "test_data/uv_seam_cube.obj"

    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # 烘焙数据
        print("1. 烘载数据")
        baker = MetricAlignedIUVBaker(obj_path, seed=42, use_obj_parser=True)
        data = baker.bake(
            num_samples=100,
            extrusion_sigma_ratio=0.01,
            chart_mode="uv_islands",
        )

        # 保存数据（带所有元数据）
        print("2. 保存数据")
        output_path = temp_dir / "baked.pt"
        baker.save(
            data,
            str(output_path),
            chart_mode="uv_islands",
            extrusion_sigma_ratio=0.01,
            texture_path="test_texture.png",
        )

        # 加载数据
        print("3. 加载数据")
        loaded_data, metadata = load_baked_data(str(output_path))

        # 检查所有必需字段
        print("4. 检查元数据字段")
        required_fields = [
            'mesh_path',
            'num_samples',
            'num_charts',
            'chart_mode',
            'extrusion_sigma_ratio',
            'extrusion_sigma',
            'bbox_min',
            'bbox_max',
            'uv_convention',
            'use_obj_parser',
        ]

        missing_fields = []
        for field in required_fields:
            if field not in metadata:
                missing_fields.append(field)
                print(f"   ✗ 缺少字段: {field}")
            else:
                print(f"   ✓ {field}: {metadata[field]}")

        if missing_fields:
            print(f"\n✗ 缺少字段: {missing_fields}")
            return False

        # 检查可选字段
        print("5. 检查可选字段")
        optional_fields = [
            'texture_path',
            'chart_stats',
        ]

        for field in optional_fields:
            if field in metadata:
                print(f"   ✓ {field}: {metadata[field]}")
            else:
                print(f"   - {field}: 未设置（可选）")

        # 验证字段内容
        print("6. 验证字段内容")
        assert metadata['mesh_path'] == str(obj_path), "mesh_path 不匹配"
        assert metadata['num_samples'] == len(data.pos), "num_samples 不匹配"
        assert metadata['num_charts'] == data.chart_id.max().item() + 1, "num_charts 不匹配"
        assert metadata['chart_mode'] == "uv_islands", "chart_mode 不匹配"
        assert metadata['uv_convention'] == "bottom_left_origin", "uv_convention 不匹配"
        assert metadata['use_obj_parser'] == True, "use_obj_parser 不匹配"
        assert metadata['texture_path'] == "test_texture.png", "texture_path 不匹配"

        # 验证数值字段
        assert isinstance(metadata['extrusion_sigma_ratio'], float), "extrusion_sigma_ratio 应该是 float"
        assert isinstance(metadata['extrusion_sigma'], float), "extrusion_sigma 应该是 float"
        assert metadata['extrusion_sigma'] > 0, "extrusion_sigma 应该大于 0"

        # 验证 bbox
        assert len(metadata['bbox_min']) == 3, "bbox_min 应该有 3 个元素"
        assert len(metadata['bbox_max']) == 3, "bbox_max 应该有 3 个元素"

        # 验证 chart_stats（如果有）
        if 'chart_stats' in metadata:
            stats = metadata['chart_stats']
            assert 'num_charts' in stats, "chart_stats 应该包含 num_charts"
            assert 'chart_sizes' in stats, "chart_stats 应该包含 chart_sizes"
            assert 'num_uv_seams' in stats, "chart_stats 应该包含 num_uv_seams"

            print(f"   ✓ chart_stats:")
            print(f"     - num_charts: {stats['num_charts']}")
            print(f"     - chart_sizes: {stats['chart_sizes']}")
            print(f"     - num_uv_seams: {stats['num_uv_seams']}")

        print(f"\n✓ 所有元数据字段完整且正确")

        # 验证数据一致性
        print("7. 验证数据一致性")
        assert torch.allclose(loaded_data.pos, data.pos), "pos 不一致"
        assert torch.allclose(loaded_data.j_3d_gt, data.j_3d_gt), "j_3d_gt 不一致"
        assert torch.allclose(loaded_data.uv_anchor, data.uv_anchor), "uv_anchor 不一致"
        assert torch.all(loaded_data.chart_id == data.chart_id), "chart_id 不一致"
        print(f"   ✓ 数据一致性验证通过")

        return True

    finally:
        # 清理
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import torch
    success = test_complete_metadata()
    sys.exit(0 if success else 1)
