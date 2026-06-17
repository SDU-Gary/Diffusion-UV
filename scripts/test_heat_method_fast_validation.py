"""
快速投影连续性验证脚本

专注于优化投影连续性，同时保持合理的计算时间。
使用更少但更具代表性的测试点。
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


def generate_representative_test_points(vertices, faces, num_points=200):
    """
    生成代表性测试点（更少但更有代表性）

    策略：
    1. 基于面面积加权采样
    2. 覆盖不同曲率区域
    3. 使用较小的偏移量
    """
    logger.info(f"生成 {num_points} 个代表性测试点...")

    # 计算面面积和重心
    face_areas = []
    face_centroids = []
    face_normals = []

    for face in faces:
        face_verts = vertices[face]
        edge1 = face_verts[1] - face_verts[0]
        edge2 = face_verts[2] - face_verts[0]

        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        face_areas.append(area)

        normal = np.cross(edge1, edge2)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        face_normals.append(normal)

        centroid = np.mean(face_verts, axis=0)
        face_centroids.append(centroid)

    face_areas = np.array(face_areas)
    face_normals = np.array(face_normals)
    face_centroids = np.array(face_centroids)

    # 按面积加权采样
    probabilities = face_areas / np.sum(face_areas)
    face_indices = np.random.choice(len(faces), size=num_points, p=probabilities)

    test_points = []
    for face_idx in face_indices:
        face = faces[face_idx]
        face_verts = vertices[face]
        normal = face_normals[face_idx]

        # 随机重心坐标
        r1 = np.random.random()
        r2 = np.random.random()

        if r1 + r2 > 1.0:
            r1 = 1.0 - r1
            r2 = 1.0 - r2

        r0 = 1.0 - r1 - r2

        # 面内点
        point = r0 * face_verts[0] + r1 * face_verts[1] + r2 * face_verts[2]

        # 较小的法向偏移（保持狭窄邻域）
        offset = np.random.uniform(-0.002, 0.002)
        point = point + offset * normal

        test_points.append(point)

    return np.array(test_points, dtype=np.float32)


def fast_optimized_projection_test(vertices, faces, smooth_normals, test_points):
    """
    快速优化投影测试

    策略：
    1. 使用顶点投影（快速）
    2. 应用角度感知插值（提高质量）
    3. 应用轻量级平滑（提高连续性）
    """
    logger.info("=== 快速优化投影测试 ===")

    start_time = time.time()

    # 初始化投影器
    projection = ClosestPointProjection(vertices, faces, smooth_normals)

    # 估计投影距离
    edge_lengths = []
    for i in range(len(vertices) - 1):
        edge_length = np.linalg.norm(vertices[i + 1] - vertices[i])
        if edge_length > 1e-10:
            edge_lengths.append(edge_length)

    max_distance = np.median(edge_lengths) if edge_lengths else 0.001

    logger.info(f"使用投影距离: {max_distance:.6f}")

    # 计算空间法线场（优化参数）
    spatial_result = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance * 1.5,
        use_vertex_projection=False,  # 使用精确投影但点数少
        apply_smoothing=True,
        smoothing_iterations=2  # 轻量级平滑
    )

    computation_time = time.time() - start_time

    # 评估质量
    quality_metrics = projection.evaluate_projection_quality(test_points)

    logger.info(f"快速测试完成: {computation_time:.3f}秒")
    logger.info(f"投影成功率: {quality_metrics['projection_success_rate']:.2%}")
    logger.info(f"连续性分数: {quality_metrics['continuity_score']:.3f}")

    return {
        'computation_time': computation_time,
        'quality_metrics': quality_metrics,
        'spatial_normals': spatial_result['normals'],
        'test_points': test_points,
    }


def compare_projection_methods(vertices, faces, smooth_normals, test_points):
    """
    对比不同的投影方法
    """
    logger.info("=== 对比投影方法 ===")

    projection = ClosestPointProjection(vertices, faces, smooth_normals)

    # 估计投影距离
    edge_lengths = []
    for i in range(len(vertices) - 1):
        edge_length = np.linalg.norm(vertices[i + 1] - vertices[i])
        if edge_length > 1e-10:
            edge_lengths.append(edge_length)
    max_distance = np.median(edge_lengths) if edge_lengths else 0.001

    results = {}

    # 方法1：原始顶点投影
    logger.info("测试方法1: 原始顶点投影")
    start_time = time.time()
    result1 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance,
        use_vertex_projection=True,
        apply_smoothing=False
    )
    time1 = time.time() - start_time
    quality1 = projection.evaluate_projection_quality(test_points)

    results['vertex_projection'] = {
        'time': time1,
        'continuity_score': quality1['continuity_score'],
        'success_rate': quality1['projection_success_rate'],
        'avg_normal_diff': quality1['avg_normal_difference'],
    }

    # 方法2：精确投影
    logger.info("测试方法2: 精确面投影")
    start_time = time.time()
    result2 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance,
        use_vertex_projection=False,
        apply_smoothing=False
    )
    time2 = time.time() - start_time
    quality2 = projection.evaluate_projection_quality(test_points)

    results['face_projection'] = {
        'time': time2,
        'continuity_score': quality2['continuity_score'],
        'success_rate': quality2['projection_success_rate'],
        'avg_normal_diff': quality2['avg_normal_difference'],
    }

    # 方法3：顶点投影 + 平滑
    logger.info("测试方法3: 顶点投影 + 平滑")
    start_time = time.time()
    result3 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance,
        use_vertex_projection=True,
        apply_smoothing=True,
        smoothing_iterations=2
    )
    time3 = time.time() - start_time
    quality3 = projection.evaluate_projection_quality(test_points)

    results['vertex_projection_smooth'] = {
        'time': time3,
        'continuity_score': quality3['continuity_score'],
        'success_rate': quality3['projection_success_rate'],
        'avg_normal_diff': quality3['avg_normal_difference'],
    }

    # 方法4：精确投影 + 平滑
    logger.info("测试方法4: 精确投影 + 平滑")
    start_time = time.time()
    result4 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance,
        use_vertex_projection=False,
        apply_smoothing=True,
        smoothing_iterations=2
    )
    time4 = time.time() - start_time
    quality4 = projection.evaluate_projection_quality(test_points)

    results['face_projection_smooth'] = {
        'time': time4,
        'continuity_score': quality4['continuity_score'],
        'success_rate': quality4['projection_success_rate'],
        'avg_normal_diff': quality4['avg_normal_difference'],
    }

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="快速投影连续性优化验证")
    parser.add_argument('--model', type=str,
                       default='data/models/stanford_bunny_procedural.obj',
                       help='测试模型路径')
    parser.add_argument('--output', type=str,
                       default='outputs/heat_projection_optimization',
                       help='输出目录')
    parser.add_argument('--num-points', type=int, default=200,
                       help='测试点数量（较少但更有代表性）')

    args = parser.parse_args()

    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    model_data = load_test_model(args.model)
    vertices = model_data['vertices']
    faces = model_data['faces']

    # 2. 计算热方法平滑法线
    logger.info("=== 计算热方法平滑法线 ===")
    heat_method = HeatMethodNormalField(vertices, faces, time_step=1e-3)
    smooth_normals = heat_method.diffuse_normals(num_iterations=10)

    # 3. 生成代表性测试点
    test_points = generate_representative_test_points(
        vertices, faces, args.num_points
    )

    # 4. 对比不同投影方法
    comparison_results = compare_projection_methods(
        vertices, faces, smooth_normals, test_points
    )

    # 5. 保存结果
    results = {
        'model_path': args.model,
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'num_test_points': args.num_points,
        'comparison': comparison_results,
        'best_method': max(comparison_results.items(),
                          key=lambda x: x[1]['continuity_score']),
        'fastest_effective_method': min(
            [(k, v) for k, v in comparison_results.items()
             if v['continuity_score'] > 0.5],
            key=lambda x: x[1]['time'],
            default=('none', {})
        )
    }

    output_file = output_path / 'projection_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"结果已保存到: {output_file}")

    # 6. 输出总结
    logger.info("=== 投影优化总结 ===")
    logger.info(f"测试方法数量: {len(comparison_results)}")

    for method_name, method_results in comparison_results.items():
        logger.info(f"{method_name}:")
        logger.info(f"  时间: {method_results['time']:.3f}秒")
        logger.info(f"  连续性: {method_results['continuity_score']:.3f}")
        logger.info(f"  成功率: {method_results['success_rate']:.2%}")

    logger.info(f"\n最佳方法 (连续性): {results['best_method'][0]}")
    logger.info(f"最快有效方法: {results['fastest_effective_method'][0]}")

    return results


if __name__ == "__main__":
    main()