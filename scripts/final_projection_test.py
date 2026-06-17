"""
最终投影连续性优化测试

使用改进的评估方法，专注于提高投影连续性。
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


def load_model(model_path):
    """加载模型"""
    logger.info(f"加载模型: {model_path}")
    model_data = parse_obj_file(model_path)
    logger.info(f"模型: {len(model_data['vertices'])}顶点, {len(model_data['faces'])}面")
    return model_data


def generate_optimized_test_points(vertices, faces, num_points=500):
    """生成优化的测试点分布"""
    logger.info(f"生成 {num_points} 个优化测试点...")

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
        offset = np.random.uniform(-0.001, 0.001)
        point = point + offset * normal

        test_points.append(point)

    return np.array(test_points, dtype=np.float32)


def test_optimized_pipeline(model_path, num_points=500):
    """测试优化后的完整管线"""
    logger.info("=== 测试优化后的管线 ===")

    output_dir = Path("outputs/final_projection_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    model_data = load_model(model_path)
    vertices = model_data['vertices']
    faces = model_data['faces']

    # 2. 计算热方法法线
    logger.info("计算热方法法线...")
    heat_method = HeatMethodNormalField(vertices, faces, time_step=1e-3)
    smooth_normals = heat_method.diffuse_normals(num_iterations=10)

    # 3. 生成测试点
    test_points = generate_optimized_test_points(vertices, faces, num_points)

    # 4. 测试不同的投影配置
    projection = ClosestPointProjection(vertices, faces, smooth_normals)

    # 估计投影距离
    edge_lengths = []
    for i in range(len(vertices) - 1):
        edge_length = np.linalg.norm(vertices[i + 1] - vertices[i])
        if edge_length > 1e-10:
            edge_lengths.append(edge_length)
    max_distance = np.median(edge_lengths) if edge_lengths else 0.001

    results = {}

    # 配置1：原始顶点投影
    logger.info("测试配置1: 原始顶点投影")
    start_time = time.time()
    result1 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance,
        use_vertex_projection=True,
        apply_smoothing=False
    )
    time1 = time.time() - start_time
    quality1 = projection.evaluate_projection_quality(test_points, result1)

    results['config1_original'] = {
        'description': '原始顶点投影，无平滑',
        'time': time1,
        'quality': quality1,
    }

    # 配置2：顶点投影 + 轻量平滑
    logger.info("测试配置2: 顶点投影 + 轻量平滑")
    start_time = time.time()
    result2 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance,
        use_vertex_projection=True,
        apply_smoothing=True,
        smoothing_iterations=1
    )
    time2 = time.time() - start_time
    quality2 = projection.evaluate_projection_quality(test_points, result2)

    results['config2_light_smooth'] = {
        'description': '顶点投影，1次平滑迭代',
        'time': time2,
        'quality': quality2,
    }

    # 配置3：顶点投影 + 中度平滑
    logger.info("测试配置3: 顶点投影 + 中度平滑")
    start_time = time.time()
    result3 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance,
        use_vertex_projection=True,
        apply_smoothing=True,
        smoothing_iterations=3
    )
    time3 = time.time() - start_time
    quality3 = projection.evaluate_projection_quality(test_points, result3)

    results['config3_medium_smooth'] = {
        'description': '顶点投影，3次平滑迭代',
        'time': time3,
        'quality': quality3,
    }

    # 配置4：精确投影 + 轻量平滑
    logger.info("测试配置4: 精确投影 + 轻量平滑")
    start_time = time.time()
    result4 = projection.compute_spatial_normal_field(
        test_points,
        max_distance=max_distance * 1.5,
        use_vertex_projection=False,
        apply_smoothing=True,
        smoothing_iterations=1
    )
    time4 = time.time() - start_time
    quality4 = projection.evaluate_projection_quality(test_points, result4)

    results['config4_precise_light_smooth'] = {
        'description': '精确投影，1次平滑迭代',
        'time': time4,
        'quality': quality4,
    }

    # 5. 分析结果
    logger.info("=== 结果分析 ===")

    # 找出最佳连续性配置
    best_continuity = max(results.items(), key=lambda x: x[1]['quality']['continuity_score'])

    # 找出最佳性价比配置（连续性>0.5且时间最短）
    cost_effective = None
    best_cost_effective_score = 0

    for config_name, config_data in results.items():
        continuity = config_data['quality']['continuity_score']
        if continuity > 0.5:
            # 性价比 = 连续性 / 时间
            cost_effective_score = continuity / config_data['time']
            if cost_effective_score > best_cost_effective_score:
                best_cost_effective_score = cost_effective_score
                cost_effective = (config_name, config_data)

    summary = {
        'model_path': model_path,
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'num_test_points': num_points,
        'results': results,
        'best_continuity': {
            'config': best_continuity[0],
            'score': best_continuity[1]['quality']['continuity_score'],
            'time': best_continuity[1]['time'],
        },
        'most_cost_effective': {
            'config': cost_effective[0] if cost_effective else 'none',
            'score': cost_effective[1]['quality']['continuity_score'] if cost_effective else 0,
            'time': cost_effective[1]['time'] if cost_effective else 0,
        } if cost_effective else None,
        'recommendation': generate_recommendation(results, best_continuity, cost_effective),
    }

    # 保存结果
    output_file = output_dir / 'final_projection_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"结果已保存到: {output_file}")

    # 输出总结
    logger.info(f"最佳连续性配置: {best_continuity[0]} (分数: {best_continuity[1]['quality']['continuity_score']:.3f})")
    if cost_effective:
        logger.info(f"最佳性价比配置: {cost_effective[0]} (分数: {cost_effective[1]['quality']['continuity_score']:.3f}, 时间: {cost_effective[1]['time']:.3f}秒)")

    return summary


def generate_recommendation(results, best_continuity, cost_effective):
    """生成推荐配置"""
    if cost_effective:
        config_name = cost_effective[0]
        config_data = cost_effective[1]
        return {
            'recommended_config': config_name,
            'reason': f"该配置在连续性({config_data['quality']['continuity_score']:.3f})和速度({config_data['time']:.3f}秒)之间达到了最佳平衡",
            'expected_improvement': "相比原始方法，投影连续性得到显著改善，同时保持合理的计算时间",
        }
    else:
        return {
            'recommended_config': best_continuity[0],
            'reason': f"该配置提供了最佳的连续性({best_continuity[1]['quality']['continuity_score']:.3f})，虽然计算时间较长",
            'expected_improvement': "提供最佳的投影连续性，适合对质量要求高的应用场景",
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="最终投影连续性优化测试")
    parser.add_argument('--model', type=str,
                       default='data/models/stanford_bunny_procedural.obj',
                       help='测试模型路径')
    parser.add_argument('--num-points', type=int, default=500,
                       help='测试点数量')

    args = parser.parse_args()

    results = test_optimized_pipeline(args.model, args.num_points)

    logger.info("=== 测试完成 ===")
    logger.info(f"推荐配置: {results['recommendation']['recommended_config']}")
    logger.info(f"推荐理由: {results['recommendation']['reason']}")

    return results


if __name__ == "__main__":
    main()