"""
对比原始ALSFD与修复版本

深入分析为什么原始版本能达到99.999996%等价性，
而修复版本只有27%等价性
"""

import numpy as np
import sys
from pathlib import Path
import logging
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.geometry.alsfd_diffusion import ALSFDVectorFieldDiffusion
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


def analyze_laplacian_difference():
    """分析两个版本Laplacian的差异"""
    logger.info("=== 分析Laplacian差异 ===")

    # 加载模型
    model_data = load_test_model('data/models/stanford_bunny_procedural.obj')
    vertices = model_data['vertices']
    faces = model_data['faces']

    # 创建原始版本
    logger.info("\n创建原始ALSFD...")
    alsfd_original = ALSFDVectorFieldDiffusion(vertices, faces)

    # 创建修复版本
    logger.info("\n创建修复版ALSFD...")
    alsfd_fixed = ALSFDVectorFieldDiffusionFixed(vertices, faces)

    # 对比Laplacian算子
    logger.info("\n对比Laplacian算子...")

    # 原始版本的切向Laplacian
    L_original = alsfd_original._build_tangent_laplacian()
    logger.info(f"原始版Laplacian: {L_original.shape}, {L_original.nnz}非零元素")

    # 修复版本的ALSFD算子
    L_fixed = alsfd_fixed.build_alsfd_operator()
    logger.info(f"修复版ALSFD算子: {L_fixed.shape}, {L_fixed.nnz}非零元素")

    # 分析对角元素
    diag_original = L_original.diagonal()
    diag_fixed = L_fixed.diagonal()

    logger.info(f"原始版对角元素范围: [{diag_original.min():.6f}, {diag_original.max():.6f}]")
    logger.info(f"修复版对角元素范围: [{diag_fixed.min():.6f}, {diag_fixed.max():.6f}]")

    logger.info(f"原始版对角元素均值: {diag_original.mean():.6f}")
    logger.info(f"修复版对角元素均值: {diag_fixed.mean():.6f}")

    return {
        'original_laplacian': {
            'shape': list(L_original.shape),
            'nnz': int(L_original.nnz),
            'diag_mean': float(diag_original.mean()),
            'diag_range': [float(diag_original.min()), float(diag_original.max())],
        },
        'fixed_laplacian': {
            'shape': list(L_fixed.shape),
            'nnz': int(L_fixed.nnz),
            'diag_mean': float(diag_fixed.mean()),
            'diag_range': [float(diag_fixed.min()), float(diag_fixed.max())],
        },
    }


def test_original_version_theory():
    """测试原始版本的理论验证"""
    logger.info("\n=== 测试原始版本理论验证 ===")

    # 加载模型
    model_data = load_test_model('data/models/stanford_bunny_procedural.obj')
    vertices = model_data['vertices']
    faces = model_data['faces']

    # 创建原始ALSFD
    alsfd_original = ALSFDVectorFieldDiffusion(vertices, faces)

    # 使用法线作为初始向量场（原始版本的方式）
    initial_vectors = alsfd_original.normals.copy()

    # 执行扩散
    logger.info("执行原始版本扩散...")
    diffused_original = alsfd_original.diffuse_vector_field(
        initial_vectors,
        time_step=0.001,
        num_iterations=10
    )

    # 验证Bochner等价性
    logger.info("验证原始版本Bochner等价性...")
    equivalence_original = alsfd_original.compare_with_surface_bochner(diffused_original)

    # 分析数值范围
    magnitudes = np.linalg.norm(diffused_original, axis=1)
    logger.info(f"原始版本扩散后模长: {magnitudes.mean():.6e}")

    return {
        'original_bochner_equivalence': equivalence_original,
        'diffused_magnitude': float(magnitudes.mean()),
    }


def test_implementation_difference():
    """测试实现差异"""
    logger.info("\n=== 测试实现差异 ===")

    # 加载模型
    model_data = load_test_model('data/models/stanford_bunny_procedural.obj')
    vertices = model_data['vertices']
    faces = model_data['faces']

    # 创建两个版本
    alsfd_original = ALSFDVectorFieldDiffusion(vertices, faces)
    alsfd_fixed = ALSFDVectorFieldDiffusionFixed(vertices, faces)

    # 测试相同的计算
    test_vector = np.array([1.0, 0.0, 0.0])

    # 测试投影
    logger.info("测试投影矩阵...")
    P_original = alsfd_original.projection_matrices[0]
    P_fixed = alsfd_fixed.projection_matrices[0]

    projected_original = P_original @ test_vector
    projected_fixed = P_fixed @ test_vector

    logger.info(f"原始版投影结果: {projected_original}")
    logger.info(f"修复版投影结果: {projected_fixed}")

    return {
        'projection_test': {
            'original': projected_original.tolist(),
            'fixed': projected_fixed.tolist(),
        }
    }


def analyze_theory_validation_paradox():
    """分析理论验证悖论"""
    logger.info("\n=== 分析理论验证悖论 ===")

    logger.info("悖论：")
    logger.info("  - 原始版本理论验证：99.999996%等价性")
    logger.info("  - 原始版本法线场：完全消失（2.696e-21）")
    logger.info("  - 修复版本法线场：健康（1.087078）")
    logger.info("  - 修复版本等价性：27%")

    logger.info("\n可能的原因：")
    logger.info("1. 原始版本的理论验证在数值消失的情况下进行")
    logger.info("   - 当向量场接近零时，任何两个向量场都高度'等价'")
    logger.info("   - 99.999996%等价性可能是因为两者都接近零向量")
    logger.info("")
    logger.info("2. 修复版本使用真实的切向量场，数值健康")
    logger.info("   - 真实的数值差异暴露了算子构建的问题")
    logger.info("   - 27%等价性反映了真实的算子差异")
    logger.info("")
    logger.info("3. ALSFD理论验证可能存在根本性问题")
    logger.info("   - 理论基于切向量场假设")
    logger.info("   - 但原始实现使用法线作为初始条件")
    logger.info("   - 验证结果可能是数值假象")

    return {
        'paradox_analysis': {
            'original_validation': '99.999996%等价性（可能是数值假象）',
            'original_magnitude': '2.696e-21（完全消失）',
            'fixed_validation': '27%等价性（真实差异）',
            'fixed_magnitude': '1.087078（数值健康）',
            'conclusion': '原始理论验证可能是数值假象',
        }
    }


def main():
    """主函数"""
    logger.info("开始对比分析")

    results = {}

    # 1. 分析Laplacian差异
    laplacian_results = analyze_laplacian_difference()
    results['laplacian_comparison'] = laplacian_results

    # 2. 测试原始版本理论验证
    logger.info("\n" + "="*80)
    original_theory_results = test_original_version_theory()
    results['original_theory_validation'] = original_theory_results

    # 3. 测试实现差异
    logger.info("\n" + "="*80)
    implementation_results = test_implementation_difference()
    results['implementation_difference'] = implementation_results

    # 4. 分析悖论
    logger.info("\n" + "="*80)
    paradox_results = analyze_theory_validation_paradox()
    results['paradox_analysis'] = paradox_results

    # 保存结果
    output_path = Path('outputs/alsfd_comparison')
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / 'alsfd_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n对比分析结果已保存到: {output_file}")

    # 输出总结
    logger.info("\n" + "="*80)
    logger.info("=== 对比分析总结 ===")

    logger.info("\n关键发现：")
    logger.info("1. 原始版本的'完美'理论验证可能是数值假象")
    logger.info("   - 向量场消失后，任何两个向量场都高度相关")
    logger.info("   - 99.999996%等价性 ≈ 0向量场等价性")
    logger.info("")
    logger.info("2. 修复版本暴露了真实的算法问题")
    logger.info("   - ALSFD算子构建仍然不正确")
    logger.info("   - Bochner等价性只有27%")
    logger.info("   - 但数值稳定性大幅改善")
    logger.info("")
    logger.info("3. ALSFD理论本身可能需要重新审视")
    logger.info("   - 理论验证基于数值假象")
    logger.info("   - 实际实现存在根本性困难")
    logger.info("   - 需要更深入的理论分析")

    logger.info(f"\n对比指标：")
    logger.info(f"  原始版Laplacian对角均值: {laplacian_results['original_laplacian']['diag_mean']:.6f}")
    logger.info(f"  修复版Laplacian对角均值: {laplacian_results['fixed_laplacian']['diag_mean']:.6f}")
    logger.info(f"  原始版等价性: {original_theory_results['original_bochner_equivalence']['equivalence_score']:.6f}")
    logger.info(f"  原始版扩散后模长: {original_theory_results['diffused_magnitude']:.6e}")

    return results


if __name__ == "__main__":
    main()
