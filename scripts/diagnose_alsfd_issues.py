"""
ALSFD方法问题诊断脚本

详细分析ALSFD实现中的具体问题，包括：
1. Laplacian算子构建问题
2. 过度投影问题
3. 数值稳定性问题
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


def diagnose_laplacian_operator(alsfd_diffusion):
    """诊断Laplacian算子构建问题"""
    logger.info("=== 诊断Laplacian算子 ===")

    # 1. 检查网格Laplacian
    logger.info("1. 检查网格Laplacian...")
    L_mesh = alsfd_diffusion._build_mesh_laplacian()

    logger.info(f"   Laplacian矩阵形状: {L_mesh.shape}")
    logger.info(f"   非零元素数: {L_mesh.nnz}")
    logger.info(f"   矩阵密度: {L_mesh.nnz / (L_mesh.shape[0] * L_mesh.shape[1]):.6e}")

    # 检查对角元素
    diagonal = L_mesh.diagonal()
    logger.info(f"   对角元素范围: [{diagonal.min():.6f}, {diagonal.max():.6f}]")
    logger.info(f"   对角元素均值: {diagonal.mean():.6f}")
    logger.info(f"   负对角元素比例: {np.sum(diagonal < 0) / len(diagonal):.2%}")

    # 检查非对角元素（需要排除对角元素）
    L_mesh_csr = L_mesh.tocsr()
    all_data = L_mesh_csr.data
    # 获取行索引，找出非对角元素
    rows = []
    cols = []
    for i in range(L_mesh_csr.shape[0]):
        start = L_mesh_csr.indptr[i]
        end = L_mesh_csr.indptr[i+1]
        for j in range(start, end):
            col = L_mesh_csr.indices[j]
            if i != col:  # 非对角元素
                rows.append(i)
                cols.append(col)

    # 提取非对角元素数据
    off_diag_data = []
    for i, col in enumerate(cols):
        # 找到对应的数据值
        row_idx = rows[i]
        start = L_mesh_csr.indptr[row_idx]
        end = L_mesh_csr.indptr[row_idx+1]
        for j in range(start, end):
            if L_mesh_csr.indices[j] == col:
                off_diag_data.append(abs(L_mesh_csr.data[j]))
                break

    if off_diag_data:
        logger.info(f"   非对角元素均值: {np.mean(off_diag_data):.6f}")
    else:
        logger.info(f"   非对角元素: 无")

    # 2. 检查切向Laplacian
    logger.info("2. 检查切向Laplacian...")
    L_tangent = alsfd_diffusion._build_tangent_laplacian()

    logger.info(f"   切向Laplacian矩阵形状: {L_tangent.shape}")
    logger.info(f"   非零元素数: {L_tangent.nnz}")
    logger.info(f"   矩阵密度: {L_tangent.nnz / (L_tangent.shape[0] * L_tangent.shape[1]):.6e}")

    # 检查对角元素
    tangent_diagonal = L_tangent.diagonal()
    logger.info(f"   对角元素范围: [{tangent_diagonal.min():.6f}, {tangent_diagonal.max():.6f}]")
    logger.info(f"   对角元素均值: {tangent_diagonal.mean():.6f}")

    # 3. 检查投影矩阵
    logger.info("3. 检查投影矩阵...")
    projection_matrices = alsfd_diffusion.projection_matrices

    # 检查第一个投影矩阵
    P0 = projection_matrices[0]
    logger.info(f"   第一个投影矩阵:\n{P0}")

    # 检查投影矩阵的性质
    det = np.linalg.det(P0)
    rank = np.linalg.matrix_rank(P0)
    logger.info(f"   投影矩阵行列式: {det:.6f}")
    logger.info(f"   投影矩阵秩: {rank} (应该是2)")

    # 检查对称性
    is_symmetric = np.allclose(P0, P0.T)
    logger.info(f"   投影矩阵对称性: {is_symmetric}")

    # 检查幂等性 P^2 = P
    P_squared = P0 @ P0
    is_idempotent = np.allclose(P_squared, P0)
    logger.info(f"   投影矩阵幂等性: {is_idempotent}")

    return {
        'laplacian_density': L_mesh.nnz / (L_mesh.shape[0] * L_mesh.shape[1]),
        'diagonal_mean': float(diagonal.mean()),
        'negative_diag_ratio': float(np.sum(diagonal < 0) / len(diagonal)),
        'projection_det': float(det),
        'projection_rank': int(rank),
        'projection_symmetric': bool(is_symmetric),
        'projection_idempotent': bool(is_idempotent),
    }


def test_projection_decay(alsfd_diffusion, num_iterations=20):
    """测试投影导致的数值衰减"""
    logger.info("=== 测试投影衰减 ===")

    # 创建初始向量场（法线）
    initial_vectors = alsfd_diffusion.normals.copy()

    # 分析初始向量场
    initial_magnitudes = np.linalg.norm(initial_vectors, axis=1)
    logger.info(f"初始向量场模长范围: [{initial_magnitudes.min():.6f}, {initial_magnitudes.max():.6f}]")
    logger.info(f"初始向量场模长均值: {initial_magnitudes.mean():.6f}")

    # 测试连续投影的效果
    current_vectors = initial_vectors.copy()
    magnitude_history = [initial_magnitudes.mean()]

    for i in range(num_iterations):
        # 执行投影
        current_vectors = alsfd_diffusion.compute_tangent_frame_field(current_vectors)

        # 计算模长
        magnitudes = np.linalg.norm(current_vectors, axis=1)
        mean_magnitude = magnitudes.mean()
        magnitude_history.append(mean_magnitude)

        if i % 5 == 0:
            logger.info(f"   投影迭代 {i}: 平均模长 = {mean_magnitude:.6e}")

    # 分析衰减
    decay_ratio = magnitude_history[-1] / magnitude_history[0]
    logger.info(f"总衰减比例: {decay_ratio:.6e}")
    logger.info(f"是否严重衰减: {decay_ratio < 0.1}")

    return {
        'initial_magnitude': float(magnitude_history[0]),
        'final_magnitude': float(magnitude_history[-1]),
        'decay_ratio': float(decay_ratio),
        'magnitude_history': [float(m) for m in magnitude_history],
    }


def test_diffusion_process(alsfd_diffusion):
    """测试扩散过程中的数值变化"""
    logger.info("=== 测试扩散过程 ===")

    # 使用法线作为初始向量场
    initial_vectors = alsfd_diffusion.normals.copy()

    # 分析初始向量场
    initial_magnitudes = np.linalg.norm(initial_vectors, axis=1)
    logger.info(f"初始法线场模长范围: [{initial_magnitudes.min():.6f}, {initial_magnitudes.max():.6f}]")
    logger.info(f"初始法线场模长均值: {initial_magnitudes.mean():.6f}")

    # 执行短时间扩散
    logger.info("执行短时间扩散 (1次迭代)...")
    diffused_1 = alsfd_diffusion.diffuse_vector_field(
        initial_vectors.copy(),
        time_step=0.001,
        num_iterations=1
    )

    magnitudes_1 = np.linalg.norm(diffused_1, axis=1)
    logger.info(f"1次迭代后模长范围: [{magnitudes_1.min():.6e}, {magnitudes_1.max():.6e}]")
    logger.info(f"1次迭代后模长均值: {magnitudes_1.mean():.6e}")

    # 执行长时间扩散
    logger.info("执行长时间扩散 (10次迭代)...")
    diffused_10 = alsfd_diffusion.diffuse_vector_field(
        initial_vectors.copy(),
        time_step=0.001,
        num_iterations=10
    )

    magnitudes_10 = np.linalg.norm(diffused_10, axis=1)
    logger.info(f"10次迭代后模长范围: [{magnitudes_10.min():.6e}, {magnitudes_10.max():.6e}]")
    logger.info(f"10次迭代后模长均值: {magnitudes_10.mean():.6e}")

    # 分析衰减
    decay_1 = magnitudes_1.mean() / initial_magnitudes.mean()
    decay_10 = magnitudes_10.mean() / initial_magnitudes.mean()

    logger.info(f"1次迭代衰减比例: {decay_1:.6e}")
    logger.info(f"10次迭代衰减比例: {decay_10:.6e}")

    return {
        'initial_mean': float(initial_magnitudes.mean()),
        'iteration_1_mean': float(magnitudes_1.mean()),
        'iteration_10_mean': float(magnitudes_10.mean()),
        'decay_1': float(decay_1),
        'decay_10': float(decay_10),
    }


def analyze_numerical_stability(alsfd_diffusion):
    """分析数值稳定性"""
    logger.info("=== 分析数值稳定性 ===")

    # 1. 检查SDF梯度（法线）
    normals = alsfd_diffusion.normals
    normal_magnitudes = np.linalg.norm(normals, axis=1)

    logger.info(f"法线模长范围: [{normal_magnitudes.min():.6f}, {normal_magnitudes.max():.6f}]")
    logger.info(f"法线模长均值: {normal_magnitudes.mean():.6f}")
    logger.info(f"非法线向量数: {np.sum(normal_magnitudes < 0.9)}")
    logger.info(f"异常法线数: {np.sum(normal_magnitudes > 1.1)}")

    # 2. 检查SDF值
    sdf_values = alsfd_diffusion.sdf_values
    logger.info(f"SDF值范围: [{sdf_values.min():.6f}, {sdf_values.max():.6f}]")
    logger.info(f"SDF值均值: {sdf_values.mean():.6f}")

    # 3. 检查投影矩阵数值范围
    projection_matrices = alsfd_diffusion.projection_matrices
    P_flat = projection_matrices.reshape(-1, 9)

    logger.info(f"投影矩阵元素范围: [{P_flat.min():.6f}, {P_flat.max():.6f}]")
    logger.info(f"投影矩阵元素均值: {P_flat.mean():.6f}")

    # 4. 检查条件数
    condition_numbers = []
    for i in range(min(100, len(projection_matrices))):  # 采样检查
        P = projection_matrices[i]
        # 计算非零奇异值
        singular_values = np.linalg.svd(P, compute_uv=False)
        non_zero_singulars = singular_values[singular_values > 1e-10]
        if len(non_zero_singulars) > 0:
            cond = non_zero_singulars.max() / (non_zero_singulars.min() + 1e-10)
            condition_numbers.append(cond)

    logger.info(f"投影矩阵条件数范围: [{min(condition_numbers):.6f}, {max(condition_numbers):.6f}]")
    logger.info(f"投影矩阵条件数均值: {np.mean(condition_numbers):.6f}")

    return {
        'normal_magnitude_mean': float(normal_magnitudes.mean()),
        'non_unit_normals': int(np.sum(normal_magnitudes < 0.9)),
        'sdf_range': [float(sdf_values.min()), float(sdf_values.max())],
        'projection_condition_mean': float(np.mean(condition_numbers)),
    }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ALSFD方法问题诊断")
    parser.add_argument('--model', type=str,
                       default='data/models/stanford_bunny_procedural.obj',
                       help='测试模型路径')
    parser.add_argument('--output', type=str,
                       default='outputs/alsfd_diagnosis',
                       help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model_data = load_test_model(args.model)
    vertices = model_data['vertices']
    faces = model_data['faces']

    logger.info("=== 开始ALSFD方法诊断 ===")

    # 创建ALSFD扩散器
    alsfd_diffusion = ALSFDVectorFieldDiffusion(vertices, faces)

    # 执行各项诊断
    diagnosis_results = {
        'model_path': args.model,
        'model_stats': {
            'num_vertices': len(vertices),
            'num_faces': len(faces),
        },
    }

    # 1. Laplacian算子诊断
    logger.info("\n" + "="*80)
    laplacian_diagnosis = diagnose_laplacian_operator(alsfd_diffusion)
    diagnosis_results['laplacian_diagnosis'] = laplacian_diagnosis

    # 2. 投影衰减测试
    logger.info("\n" + "="*80)
    projection_decay = test_projection_decay(alsfd_diffusion, num_iterations=20)
    diagnosis_results['projection_decay'] = projection_decay

    # 3. 扩散过程测试
    logger.info("\n" + "="*80)
    diffusion_process = test_diffusion_process(alsfd_diffusion)
    diagnosis_results['diffusion_process'] = diffusion_process

    # 4. 数值稳定性分析
    logger.info("\n" + "="*80)
    numerical_stability = analyze_numerical_stability(alsfd_diffusion)
    diagnosis_results['numerical_stability'] = numerical_stability

    # 保存结果
    output_file = output_path / 'alsfd_diagnosis_results.json'
    with open(output_file, 'w') as f:
        json.dump(diagnosis_results, f, indent=2, default=str)

    logger.info(f"诊断结果已保存到: {output_file}")

    # 输出总结
    logger.info("\n" + "="*80)
    logger.info("=== 诊断总结 ===")

    # 判断严重程度
    severity_issues = []

    if laplacian_diagnosis['laplacian_density'] < 0.001:
        severity_issues.append("Laplacian矩阵过于稀疏")

    if laplacian_diagnosis['negative_diag_ratio'] > 0.5:
        severity_issues.append("Laplacian对角元素异常")

    if projection_decay['decay_ratio'] < 0.1:
        severity_issues.append("投影导致严重数值衰减")

    if diffusion_process['decay_10'] < 1e-10:
        severity_issues.append("扩散过程导致法线场消失")

    if numerical_stability['non_unit_normals'] > 100:
        severity_issues.append("法线归一化问题")

    if len(severity_issues) > 0:
        logger.warning("发现严重问题:")
        for issue in severity_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("未发现严重数值问题")

    logger.info(f"总计发现 {len(severity_issues)} 个严重问题")

    return diagnosis_results


if __name__ == "__main__":
    main()
