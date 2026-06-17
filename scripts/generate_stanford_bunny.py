#!/usr/bin/env python3
"""
生成正确的 Stanford Bunny Procedural OBJ 文件

修复UV索引超出范围的问题：
- 确保 UV 索引始终在 [1, 20] 范围内
- 每个chart使用独立的UV范围
"""

import numpy as np
from pathlib import Path

def generate_stanford_bunny_obj():
    """生成Stanford Bunny procedural OBJ"""

    # 参数
    grid_size = 4  # 4x4x4 grid
    spacing = 0.05
    num_charts = 8

    vertices = []
    uvs = []
    faces = []

    # 生成顶点
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                vx = x * spacing
                vy = y * spacing
                vz = z * spacing
                vertices.append((vx, vy, vz))

    # 生成 UV 坐标 (20个不同的UV点)
    # Chart 0-4: 0-0.5 range
    # Chart 5-7: 0.5-1.0 range
    uv_grid = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
        (0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (1.0, 0.25),
        (0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5), (1.0, 0.5),
        (0.0, 0.75), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75), (1.0, 0.75),
    ]
    uvs = uv_grid

    # 计算顶点数量
    num_vertices = len(vertices)
    print(f"生成 {num_vertices} 个顶点")

    # 生成面并分配chart
    chart_faces = {i: [] for i in range(num_charts)}

    # 将3D网格分区为不同的charts
    # 每个chart占据不同的空间区域
    for x in range(grid_size - 1):
        for y in range(grid_size - 1):
            for z in range(grid_size - 1):
                # 计算这个立方体属于哪个chart
                chart_id = (x + y + z) % num_charts

                # 为每个立方体创建12个三角形（6个面，每个面2个三角形）
                # 立方体的8个顶点索引
                v000 = x * grid_size * grid_size + y * grid_size + z
                v001 = v000 + 1
                v010 = x * grid_size * grid_size + (y + 1) * grid_size + z
                v011 = v010 + 1
                v100 = (x + 1) * grid_size * grid_size + y * grid_size + z
                v101 = v100 + 1
                v110 = (x + 1) * grid_size * grid_size + (y + 1) * grid_size + z
                v111 = v110 + 1

                # 立方体的6个面，每个面2个三角形
                # 面的顺序：底、顶、前、后、左、右
                cube_faces = [
                    # 底面 (z=0)
                    ([v000, v010, v011], [v000, v011, v001]),
                    # 顶面 (z=1)
                    ([v100, v101, v111], [v100, v111, v110]),
                    # 前面 (y=0)
                    ([v000, v001, v101], [v000, v101, v100]),
                    # 后面 (y=1)
                    ([v010, v110, v111], [v010, v111, v011]),
                    # 左面 (x=0)
                    ([v000, v100, v110], [v000, v110, v010]),
                    # 右面 (x=1)
                    ([v001, v011, v111], [v001, v111, v101]),
                ]

                # 添加三角形到对应的chart
                for face_pair in cube_faces:
                    for tri in face_pair:
                        chart_faces[chart_id].append(tri)

    # 为每个chart分配UV索引范围
    uv_ranges = {}
    uvs_per_chart = len(uvs) // num_charts  # 每个chart分配20//8=2个UV不够，需要重新设计

    # 重新设计：让所有charts共享UV空间，但使用不同的UV子集
    # 这样每个chart可以有足够的UV变化来创建seams
    uv_assignment = {}
    for chart_id in range(num_charts):
        # 每个chart使用不同的UV子集
        start_idx = chart_id * 2
        uv_assignment[chart_id] = [start_idx, start_idx + 1,
                                   (start_idx + 2) % len(uvs), (start_idx + 3) % len(uvs)]

    print(f"UV ranges per chart: {uv_assignment}")

    # 为每个面分配UV索引
    face_id = 0
    for chart_id in range(num_charts):
        chart_face_list = chart_faces[chart_id]
        chart_uvs = uv_assignment[chart_id]

        print(f"Chart {chart_id}: {len(chart_face_list)} faces")

        for tri_idx, tri in enumerate(chart_face_list):
            # 为每个顶点分配UV索引
            uv_indices = []
            for i, v_idx in enumerate(tri):
                # 循环使用chart的UV集合
                uv_idx = chart_uvs[i % len(chart_uvs)]
                uv_indices.append(uv_idx + 1)  # OBJ是1-based

            # 创建面（使用 v/vt 格式）
            face_str = f"f {tri[0]+1}/{uv_indices[0]} {tri[1]+1}/{uv_indices[1]} {tri[2]+1}/{uv_indices[2]}"
            faces.append(face_str)
            face_id += 1

    print(f"生成 {len(faces)} 个面")

    # 写入OBJ文件
    output_path = Path("data/models/stanford_bunny_procedural.obj")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Stanford Bunny Procedural - Correct UV Indices\n")
        f.write(f"# Vertices: {len(vertices)}, UVs: {len(uvs)}, Faces: {len(faces)}, Charts: {num_charts}\n")
        f.write(f"# UV index range: 1-{len(uvs)}\n\n")

        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

        f.write("\n")

        # 写入UV坐标
        for uv in uvs:
            f.write(f"vt {uv[0]:.2f} {uv[1]:.2f}\n")

        f.write("\n")

        # 写入面
        for face in faces:
            f.write(face + "\n")

    print(f"生成OBJ文件: {output_path}")
    print(f"统计: {len(vertices)}顶点, {len(uvs)}UV, {len(faces)}面")

    # 验证UV索引范围
    max_uv_index = 0
    for face in faces:
        parts = face.split()[1:]
        for part in parts:
            indices = part.split('/')
            if len(indices) >= 2 and indices[1]:
                uv_idx = int(indices[1])
                max_uv_index = max(max_uv_index, uv_idx)

    print(f"最大UV索引: {max_uv_index} (范围: 1-{len(uvs)})")

    if max_uv_index > len(uvs):
        print(f"ERROR: UV索引超出范围!")
        return False

    return True

if __name__ == "__main__":
    success = generate_stanford_bunny_obj()
    if success:
        print("✓ OBJ文件生成成功")
    else:
        print("✗ OBJ文件生成失败")
        exit(1)