"""
ALSFD法线场连续性深入验证

深入检查ALSFD生成的法线场是否真的处处连续，
包括局部连续性、数值稳定性、与热方法对比等。
"""

import numpy as np
import sys
from pathlib import Path
import logging
import json
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.geometry.alsfd_diffusion import ALSFDVectorFieldDiffusion
from src.geometry.heat_method import HeatMethodNormalField
from src.geometry.projection import ClosestPointProjection
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


def analyze_vector_field_magnitude(vectors: np.ndarray, name: str = "Vector Field"):
    """
    分析向量场的数值范围

    Args:
        vectors: [V, 3] 向量场
        name: 向量场名称

    Returns:
        dict: 数值分析结果
    """
    magnitudes = np.linalg.norm(vectors, axis=1)

    analysis = {
        'name': name,
        'min_magnitude': float(np.min(magnitudes)),
        'max_magnitude': float(np.max(magnitudes)),
        'mean_magnitude': float(np.mean(magnitudes)),
        'std_magnitude': float(np.std(magnitudes)),
        'median_magnitude': float(np.median(magnitudes)),
        'num_zero_vectors': int(np.sum(magnitudes < 1e-10)),
        'num_small_vectors': int(np.sum(magnitudes < 1e-6)),
        'num_large_vectors': int(np.sum(magnitudes > 1.0)),
    }

    logger.info(f"{name} 数值分析:")
    logger.info(f"  范围: [{analysis['min_magnitude']:.6e}, {analysis['max_magnitude']:.6e}]")
    logger.info(f"  均值: {analysis['mean_magnitude']:.6e}, 标准差: {analysis['std_magnitude']:.6e}")
    logger.info(f"  中位数: {analysis['median_magnitude']:.6e}")
    logger.info(f"  零向量数: {analysis['num_zero_vectors']}, 小向量数: {analysis['num_small_vectors']}")

    return analysis


def verify_local_continuity(vertices, faces, vectors: np.ndarray, name: str = "Vector Field"):
    """
    验证向量场的局部连续性

    Args:
        vertices: [V, 3] 顶点坐标
        faces: [F, 3] 面索引
        vectors: [V, 3] 向量场
        name: 向量场名称

    Returns:
        dict: 连续性分析结果
    """
    logger.info(f"分析 {name} 的局部连续性...")

    # 1. 相邻顶点向量差异
    adjacent_differences = []
    edge_count = 0

    for face in faces:
        for i in range(3):
            vi, vj = face[i], face[(i + 1) % 3]
            diff = np.linalg.norm(vectors[vi] - vectors[vj])
            adjacent_differences.append(diff)
            edge_count += 1

    adjacent_differences = np.array(adjacent_differences)

    # 2. 面内向量一致性（使用重心坐标插值检查）
    face_consistencies = []
    for face in faces:
        face_verts = vertices[face]
        face_vecs = vectors[face]

        # 检查三个顶点向量的一致性
        mean_vec = np.mean(face_vecs, axis=0)
        consistency = 0.0
        for i in range(3):
            consistency += np.linalg.norm(face_vecs[i] - mean_vec)
        face_consistencies.append(consistency / 3.0)

    face_consistencies = np.array(face_consistencies)

    # 3. 局部梯度变化（估计）
    local_gradients = []
    for i in range(len(vertices)):
        # 找到邻居顶点
        neighbors = set()
        for face in faces:
            for j, v_idx in enumerate(face):
                if v_idx == i:
                    # 这个顶点的其他两个邻居
                    neighbors.add(face[(j + 1) % 3])
                    neighbors.add(face[(j + 2) % 3])

        neighbors = list(neighbors)
        if len(neighbors) > 0:
            # 计算与邻居的平均差异作为局部梯度估计
            local_diff = 0.0
            for neighbor in neighbors:
                local_diff += np.linalg.norm(vectors[i] - vectors[neighbor])
            local_gradients.append(local_diff / len(neighbors))

    local_gradients = np.array(local_gradients)

    results = {
        'name': name,
        'adjacent_difference': {
            'mean': float(np.mean(adjacent_differences)),
            'std': float(np.std(adjacent_differences)),
            'median': float(np.median(adjacent_differences)),
            'max': float(np.max(adjacent_differences)),
        },
        'face_consistency': {
            'mean': float(np.mean(face_consistencies)),
            'std': float(np.std(face_consistencies)),
            'median': float(np.median(face_consistencies)),
            'max': float(np.max(face_consistencies)),
        },
        'local_gradient': {
            'mean': float(np.mean(local_gradients)),
            'std': float(np.std(local_gradients)),
            'median': float(np.median(local_gradients)),
            'max': float(np.max(local_gradients)),
        },
    }

    logger.info(f"{name} 连续性分析:")
    logger.info(f"  相邻差异: 均值={results['adjacent_difference']['mean']:.6e}, 标准差={results['adjacent_difference']['std']:.6e}")
    logger.info(f"  面一致性: 均值={results['face_consistency']['mean']:.6e}, 标准差={results['face_consistency']['std']:.6e}")
    logger.info(f"  局部梯度: 均值={results['local_gradient']['mean']:.6e}, 标准差={results['local_gradient']['std']:.6e}")

    return results


def detect_numerical_instability(vertices, vectors: np.ndarray, name: str = "Vector Field"):
    """
    检测数值不稳定性

    Args:
        vertices: [V, 3] 顶点坐标
        vectors: [V, 3] 向量场
        name: 向量场名称

    Returns:
        dict: 不稳定性检测结果
    """
    logger.info(f"检测 {name} 的数值不稳定性...")

    # 1. 检测数值下溢出
    magnitudes = np.linalg.norm(vectors, axis=1)
    underflow_count = np.sum(magnitudes < 1e-10)
    overflow_count = np.sum(magnitudes > 1e10)

    # 2. 检测NaN和Inf
    nan_count = np.sum(np.isnan(vectors))
    inf_count = np.sum(np.isinf(vectors))

    # 3. 检测突然跳跃（相邻点巨大差异）
    # 使用KD树找到空间邻居
    from scipy.spatial import cKDTree
    kdtree = cKDTree(vertices)

    sudden_jumps = []
    for i in range(len(vectors)):
        if i % 100 == 0:  # 采样检测，避免太慢
            distances, indices = kdtree.query(vertices[i], k=6)  # 5个最近邻
            for j, (dist, neighbor_idx) in enumerate(zip(distances, indices)):
                if j == 0:  # 跳过自己
                    continue
                diff = np.linalg.norm(vectors[i] - vectors[neighbor_idx])
                # 如果距离很近但向量差异很大，可能是突然跳跃
                if dist < 0.01 and diff > 1.0:
                    sudden_jumps.append((i, neighbor_idx, dist, diff))

    # 4. 检测法线方向反转（相邻点法线点积接近-1）
    direction_reversals = []
    for i in range(len(vectors) - 1):
        dot_product = np.dot(vectors[i], vectors[i + 1])
        if dot_product < -0.9:  # 接近反向
            direction_reversals.append((i, i + 1, dot_product))

    instability_results = {
        'name': name,
        'underflow_count': int(underflow_count),
        'overflow_count': int(overflow_count),
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'sudden_jump_count': len(sudden_jumps),
        'direction_reversal_count': len(direction_reversals),
        'stability_score': 1.0 - (
            underflow_count + overflow_count + nan_count + inf_count +
            min(len(sudden_jumps), 100) / 100.0 +
            min(len(direction_reversals), 100) / 100.0
        ) / len(vectors),
    }

    logger.info(f"{name} 不稳定性检测:")
    logger.info(f"  下溢出: {instability_results['underflow_count']}, 上溢出: {instability_results['overflow_count']}")
    logger.info(f"  NaN: {instability_results['nan_count']}, Inf: {instability['inf_count']}")
    logger.info(f"  突然跳跃: {instability_results['sudden_jump_count']}, 方向反转: {instability_results['direction_reversal_count']}")
    logger.info(f"  稳定性分数: {instability_results['stability_score']:.6f}")

    return instability_results


def compare_with_heat_method_continuity(vertices, faces):
    """
    对比ALSFD与热方法的法线场连续性

    Args:
        vertices: [V, 3] 顶点坐标
        faces: [F, 3] 面索引

    Returns:
        dict: 对比结果
    """
    logger.info("=== ALSFD vs 热方法法线场连续性对比 ===")

    # 1. ALSFD方法
    logger.info("生成ALSFD法线场...")
    alsfd_diffusion = ALSFDVectorFieldDiffusion(vertices, faces)

    # 使用法线作为初始向量场
    initial_normals = alsfd_diffusion.normals.copy()

    # 执行ALSFD扩散
    alsfd_normals = alsfd_diffusion.diffuse_vector_field(
        initial_normals,
        time_step=0.001,
        num_iterations=10
    )

    # 2. 热方法
    logger.info("生成热方法法线场...")
    heat_method = HeatMethodNormalField(vertices, faces, time_step=0.001)
    heat_normals = heat_method.diffuse_normals(num_iterations=10)

    logger.info("法线场生成完成")

    # 3. 数值分析对比
    alsfd_magnitude = analyze_vector_field_magnitude(alsfd_normals, "ALSFD法线场")
    heat_magnitude = analyze_vector_field_magnitude(heat_normals, "热方法法线场")

    # 4. 连续性分析对比
    alsfd_continuity = verify_local_continuity(vertices, faces, alsfd_normals, "ALSFD法线场")
    heat_continuity = verify_local_continuity(vertices, faces, heat_normals, "热方法法线场")

    # 5. 不稳定性检测对比
    alsfd_instability = detect_numerical_instability(vertices, alsfd_normals, "ALSFD法线场")
    heat_instability = detect_numerical_instability(vertices, heat_normals, "热方法法线场")

    # 6. 综合对比
    comparison = {
        'alsfd_method': {
            'magnitude_analysis': alsfd_magnitude,
            'continuity_analysis': alsfd_continuity,
            'instability_analysis': alsfd_instability,
            'normals': alsfd_normals,
        },
        'heat_method': {
            'magnitude_analysis': heat_magnitude,
            'continuity_analysis': heat_continuity,
            'instability_analysis': heat_instability,
            'normals': heat_normals,
        },
    }

    # 7. 计算连续性分数对比
    alsfd_score = alsfd_continuity['adjacent_difference']['mean']
    heat_score = heat_continuity['adjacent_difference']['mean']

    comparison['continuity_comparison'] = {
        'alsfd_score': alsfd_score,
        'heat_score': heat_score,
        'better_method': 'ALSFD' if alsfd_score < heat_score else 'Heat',
        'improvement_ratio': (heat_score - alsfd_score) / (heat_score + 1e-10),
    }

    logger.info(f"连续性对比:")
    logger.info(f"  ALSFD连续性分数: {alsfd_score:.6e}")
    logger.info(f"  热方法连续性分数: {heat_score:.6e}")
    logger.info(f"  更优方法: {comparison['continuity_comparison']['better_method']}")

    return comparison


def check_normal_field_consistency(alsfd_normals, heat_normals, name1="ALSFD", name2="Heat"):
    """
    检查两种方法生成的法线场的一致性

    Args:
        alsfd_normals: [V, 3] ALSFD法线场
        heat_normals: [V, 3] 热方法法线场

    Returns:
        dict: 一致性分析结果
    """
    logger.info(f"检查 {name1} vs {name2} 法线场一致性...")

    # 1. 逐点差异
    pointwise_differences = np.linalg.norm(alsfd_normals - heat_normals, axis=1)

    # 2. 方向一致性（点积）
    # 归一化后计算点积
    alsfd_normalized = alsfd_normals / (np.linalg.norm(alsfd_normals, axis=1, keepdims=True) + 1e-10)
    heat_normalized = heat_normals / (np.linalg.norm(heat_normals, axis=1, keepdims=True) + 1e-10)
    dot_products = np.sum(alsfd_normalized * heat_normalized, axis=1)

    # 3. 统计分析
    consistency = {
        'mean_difference': float(np.mean(pointwise_differences)),
        'std_difference': float(np.std(pointwise_differences)),
        'median_difference': float(np.median(pointwise_differences)),
        'max_difference': float(np.max(pointwise_differences)),
        'mean_dot_product': float(np.mean(dot_products)),
        'std_dot_product': float(np.std(dot_products)),
        'correlation': float(np.mean(dot_products)),  # 因为都是单位向量，点积就是相关性
        'agreement_ratio': float(np.sum(dot_products > 0.9) / len(dot_products)),
    }

    logger.info(f"一致性分析:")
    logger.info(f"  平均差异: {consistency['mean_difference']:.6e}")
    logger.info(f"  平均相关性: {consistency['correlation']:.6f}")
    logger.info(f"  一致比例: {consistency['agreement_ratio']:.6%}")

    return consistency


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ALSFD法线场连续性深入验证")
    parser.add_argument('--model', type=str,
                       default='data/models/stanford_bunny_procedural.obj',
                       help='测试模型路径')
    parser.add_argument('--output', type=str,
                       default='outputs/alsfd_continuity_validation',
                       help='输出目录')
    parser.add_argument('--time-step', type=float, default=1e-3,
                       help='时间步长')
    parser.add_argument('--iterations', type=int, default=10,
                       help='扩散迭代次数')

    args = parser.parse_args()

    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model_data = load_test_model(args.model)
    vertices = model_data['vertices']
    faces = model_data['faces']

    logger.info("=== 开始ALSFD法线场连续性深入验证 ===")

    # 执行对比分析
    comparison = compare_with_heat_method_continuity(vertices, faces)

    # 检查一致性
    consistency = check_normal_field_consistency(
        comparison['alsfd_method']['normals'],
        comparison['heat_method']['normals'],
        "ALSFD", "热方法"
    )

    # 综合结果
    final_results = {
        'model_path': args.model,
        'model_stats': {
            'num_vertices': len(vertices),
            'num_faces': len(faces),
        },
        'continuity_validation': comparison,
        'consistency_analysis': consistency,
        'conclusion': generate_continuity_conclusion(comparison, consistency),
    }

    # 保存结果
    output_file = output_path / 'continuity_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"结果已保存到: {output_file}")

    # 输出结论
    logger.info("=== 连续性验证结论 ===")
    conclusion = final_results['conclusion']
    logger.info(f"连续性状态: {conclusion['continuity_status']}")
    logger.info(f"数值稳定性: {conclusion['stability_status']}")
    logger.info(f"推荐状态: {conclusion['recommendation']}")

    return final_results


def generate_continuity_conclusion(comparison, consistency):
    """生成连续性验证结论"""
    # 关键指标
    alsfd_score = comparison['continuity_comparison']['alsfd_score']
    heat_score = comparison['continuity_comparison']['heat_score']
    alsfd_stability = comparison['alsfd_method']['instability_analysis']['stability_score']
    heat_stability = comparison['heat_method']['instability_analysis']['stability_score']
    agreement = consistency['agreement_ratio']

    # 连续性判断
    if alsfd_score < 0.1 and alsfd_stability > 0.9:
        continuity_status = "连续且稳定"
    elif alsfd_score < 0.1:
        continuity_status = "连续但不稳定"
    elif alsfd_stability > 0.9:
        continuity_status = "不连续但稳定"
    else:
        continuity_status = "既不连续也不稳定"

    # 稳定性判断
    stability_status = "数值稳定" if max(alsfd_stability, heat_stability) > 0.9 else "存在数值问题"

    # 推荐状态
    if agreement > 0.95 and continuity_status == "连续且稳定":
        recommendation = "两种方法都可靠，推荐热方法（效率更高）"
    elif agreement > 0.8:
        recommendation = "两种方法基本一致，可根据需求选择"
    elif alsfd_stability > heat_stability:
        recommendation = "ALSFD更稳定，但计算时间较长"
    else:
        recommendation = "推荐热方法（更实用）"

    conclusion = {
        'continuity_status': continuity_status,
        'stability_status': stability_status,
        'recommendation': recommendation,
        'key_findings': {
            'alsfd_continuity': alsfd_score,
            'heat_continuity': heat_score,
            'alsfd_stability': alsfd_stability,
            'heat_stability': heat_stability,
            'method_agreement': agreement,
        },
    }

    return conclusion


if __name__ == "__main__":
    main()