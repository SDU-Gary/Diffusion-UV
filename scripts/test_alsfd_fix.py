"""
测试ALSFD修复版本

验证修复版本是否解决了法线场消失的问题
"""

import numpy as np
import sys
from pathlib import Path
import logging
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.geometry.alsfd_diffusion_fixed import ALSFDVectorFieldDiffusionFixed
from src.data.obj_parser import parse_obj_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_model(model_path: str):
    """加载测试模型"""
    logger.info(f"加载测试模型: {model_path}")
    model_data = parse_obj_file(model_path)
    logger.info(f"模型加载完成: {len(model_data['vertices'])}顶点, {len(model_data['faces'])}面")
    return model_data


def test_fixed_version():
    """测试修复版本"""
    logger.info("=== 测试ALSFD修复版本 ===")

    # 加载模型
    model_data = load_test_model('data/models/stanford_bunny_procedural.obj')
    vertices = model_data['vertices']
    faces = model_data['faces']

    # 创建修复版ALSFD扩散器
    alsfd_diffusion = ALSFDVectorFieldDiffusionFixed(vertices, faces)

    # 测试1：生成切向量场
    logger.info("\n" + "="*80)
    logger.info("测试1：生成切向量场")
    tangent_vectors = alsfd_diffusion.generate_tangent_vector_field()

    tangent_magnitudes = np.linalg.norm(tangent_vectors, axis=1)
    logger.info(f"切向量场模长范围: [{tangent_magnitudes.min():.6f}, {tangent_magnitudes.max():.6f}]")
    logger.info(f"切向量场模长均值: {tangent_magnitudes.mean():.6f}")

    # 验证切向性
    tangent_metrics = alsfd_diffusion.verify_tangent_preservation(tangent_vectors)
    logger.info(f"切向保持性: {tangent_metrics['tangent_preservation_ratio']:.6f}")

    # 测试2：扩散过程（不使用过度投影）
    logger.info("\n" + "="*80)
    logger.info("测试2：扩散过程（不使用过度投影）")

    diffused_vectors = alsfd_diffusion.diffuse_vector_field(
        tangent_vectors,
        time_step=0.001,
        num_iterations=10,
        use_projection=False  # 修复版：不使用过度投影
    )

    diffused_magnitudes = np.linalg.norm(diffused_vectors, axis=1)
    logger.info(f"扩散后模长范围: [{diffused_magnitudes.min():.6e}, {diffused_magnitudes.max():.6e}]")
    logger.info(f"扩散后模长均值: {diffused_magnitudes.mean():.6e}")

    # 验证切向性
    diffused_metrics = alsfd_diffusion.verify_tangent_preservation(diffused_vectors)
    logger.info(f"扩散后切向保持性: {diffused_metrics['tangent_preservation_ratio']:.6f}")

    # 测试3：扩散质量评估
    logger.info("\n" + "="*80)
    logger.info("测试3：扩散质量评估")
    quality_metrics = alsfd_diffusion.evaluate_diffusion_quality(tangent_vectors, diffused_vectors)
    logger.info(f"质量指标: {quality_metrics}")

    # 测试4：Bochner等价性
    logger.info("\n" + "="*80)
    logger.info("测试4：Bochner等价性")
    equivalence = alsfd_diffusion.compare_with_surface_bochner(diffused_vectors)
    logger.info(f"等价性分数: {equivalence['equivalence_score']:.6f}")

    return {
        'tangent_vector_field': {
            'mean_magnitude': float(tangent_magnitudes.mean()),
            'tangent_preservation': float(tangent_metrics['tangent_preservation_ratio']),
        },
        'diffused_vector_field': {
            'mean_magnitude': float(diffused_magnitudes.mean()),
            'tangent_preservation': float(diffused_metrics['tangent_preservation_ratio']),
        },
        'quality_metrics': quality_metrics,
        'bochner_equivalence': equivalence,
    }


def test_with_projection():
    """测试使用过度投影的情况（对比）"""
    logger.info("\n" + "="*80)
    logger.info("测试5：使用过度投影（对比测试）")

    # 加载模型
    model_data = load_test_model('data/models/stanford_bunny_procedural.obj')
    vertices = model_data['vertices']
    faces = model_data['faces']

    # 创建修复版ALSFD扩散器
    alsfd_diffusion = ALSFDVectorFieldDiffusionFixed(vertices, faces)

    # 生成切向量场
    tangent_vectors = alsfd_diffusion.generate_tangent_vector_field()

    # 使用过度投影
    diffused_with_projection = alsfd_diffusion.diffuse_vector_field(
        tangent_vectors,
        time_step=0.001,
        num_iterations=10,
        use_projection=True  # 使用过度投影
    )

    projected_magnitudes = np.linalg.norm(diffused_with_projection, axis=1)
    logger.info(f"使用投影后模长均值: {projected_magnitudes.mean():.6e}")

    return {
        'with_projection': {
            'mean_magnitude': float(projected_magnitudes.mean()),
        }
    }


def main():
    """主函数"""
    logger.info("开始ALSFD修复版本测试")

    # 测试修复版本
    fixed_results = test_fixed_version()

    # 测试过度投影（对比）
    projection_results = test_with_projection()

    # 综合结果
    final_results = {
        'fixed_version_results': fixed_results,
        'projection_comparison': projection_results,
    }

    # 保存结果
    output_path = Path('outputs/alsfd_fixed_test')
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / 'alsfd_fixed_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"\n测试结果已保存到: {output_file}")

    # 输出总结
    logger.info("\n" + "="*80)
    logger.info("=== 测试总结 ===")

    # 判断修复是否成功
    success = True
    issues = []

    if fixed_results['tangent_vector_field']['mean_magnitude'] < 0.5:
        issues.append("切向量场模长过小")
        success = False

    if fixed_results['diffused_vector_field']['mean_magnitude'] < 0.1:
        issues.append("扩散后向量场模长过小")
        success = False

    if fixed_results['diffused_vector_field']['tangent_preservation'] < 0.95:
        issues.append("扩散后切向保持性不足")
        success = False

    if fixed_results['quality_metrics']['bochner_equivalence'] < 0.9:
        issues.append("Bochner等价性不足")
        success = False

    if success:
        logger.info("✅ 修复版本测试成功！")
        logger.info("所有关键指标都在合理范围内")
    else:
        logger.warning("⚠️ 修复版本仍有问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    logger.info(f"\n关键指标:")
    logger.info(f"  切向量场模长: {fixed_results['tangent_vector_field']['mean_magnitude']:.6f}")
    logger.info(f"  扩散后模长: {fixed_results['diffused_vector_field']['mean_magnitude']:.6e}")
    logger.info(f"  切向保持性: {fixed_results['diffused_vector_field']['tangent_preservation']:.6f}")
    logger.info(f"  Bochner等价性: {fixed_results['quality_metrics']['bochner_equivalence']:.6f}")

    return final_results


if __name__ == "__main__":
    main()
