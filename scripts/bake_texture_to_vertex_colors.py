#!/usr/bin/env python3
"""
将纹理贴图烘焙到顶点颜色中

读取UV坐标和纹理贴图，为每个顶点采样纹理颜色，生成带顶点颜色的OBJ
"""

import numpy as np
from PIL import Image
import trimesh
from pathlib import Path

def bake_texture_to_vertices():
    """将纹理颜色烘焙到顶点颜色"""

    print("=" * 70)
    print("将纹理贴图烘焙到顶点颜色")
    print("=" * 70)

    # 1. 读取数据
    print("\n1. 读取网格数据...")
    obj_file = 'data/models/stanford-bunny.obj'
    mesh = trimesh.load(obj_file)

    vertices = mesh.vertices
    faces = mesh.faces

    print(f"  顶点数: {len(vertices)}")
    print(f"  面数: {len(faces)}")

    # 2. 读取UV坐标
    print("\n2. 读取UV坐标...")
    uv_data = np.load('data/textures/bunny_uvs.npz')
    uvs = uv_data['uvs']
    uv_faces = uv_data['faces']

    print(f"  UV坐标数: {len(uvs)}")
    print(f"  UV面数: {len(uv_faces)}")

    # 3. 读取纹理贴图
    print("\n3. 读取纹理贴图...")
    texture_img = Image.open('data/textures/bunny_texture.png')
    texture_array = np.array(texture_img)

    print(f"  纹理大小: {texture_img.size}")
    print(f"  纹理模式: {texture_img.mode}")
    print(f"  纹理数组形状: {texture_array.shape}")

    # 转换为RGB
    if texture_array.shape[2] == 4:  # RGBA
        texture_rgb = texture_array[:, :, :3]
    else:
        texture_rgb = texture_array

    texture_width, texture_height = texture_img.size

    # 4. 为每个顶点采样纹理颜色
    print("\n4. 为顶点采样纹理颜色...")
    vertex_colors = np.zeros((len(vertices), 3), dtype=np.float32)

    # 由于顶点数可能与UV数不匹配，我们需要根据面来插值
    # 简化处理：直接使用UV坐标对应的顶点

    # 确保顶点数与UV数一致（使用较小的数量）
    num_vertices = min(len(vertices), len(uvs))

    print(f"  处理 {num_vertices} 个顶点...")

    for i in range(num_vertices):
        uv = uvs[i]

        # 将UV坐标转换为像素坐标
        # OpenGL坐标系：原点在左下角，u向右，v向上
        # 图像坐标系：原点在左上角，u向右，v向下
        u_pixel = int(uv[0] * (texture_width - 1))
        v_pixel = int((1.0 - uv[1]) * (texture_height - 1))  # 翻转v轴

        # 确保在有效范围内
        u_pixel = max(0, min(u_pixel, texture_width - 1))
        v_pixel = max(0, min(v_pixel, texture_height - 1))

        # 采样纹理
        color = texture_rgb[v_pixel, u_pixel].astype(np.float32) / 255.0
        vertex_colors[i] = color

    # 处理剩余的顶点（如果有）
    if len(vertices) > len(uvs):
        print(f"  为额外的 {len(vertices) - len(uvs)} 个顶点设置默认颜色")
        # 使用最后一个颜色或平均颜色
        default_color = vertex_colors[-1] if len(vertex_colors) > 0 else [0.5, 0.5, 0.5]
        for i in range(len(uvs), len(vertices)):
            vertex_colors[i] = default_color

    print(f"  ✓ 顶点颜色范围: [{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]")

    # 5. 统计颜色信息
    print("\n5. 颜色统计:")
    unique_colors = len(np.unique((vertex_colors * 255).astype(np.uint8), axis=0))
    print(f"  唯一颜色数: {unique_colors}")

    r_mean, g_mean, b_mean = vertex_colors.mean(axis=0)
    print(f"  平均RGB: ({r_mean:.3f}, {g_mean:.3f}, {b_mean:.3f})")

    r_std, g_std, b_std = vertex_colors.std(axis=0)
    print(f"  RGB标准差: R={r_std:.3f} G={g_std:.3f} B={b_std:.3f}")

    # 6. 创建带顶点颜色的网格
    print("\n6. 创建带顶点颜色的网格...")
    colored_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 设置顶点颜色
    vertex_colors_uint8 = (vertex_colors * 255).astype(np.uint8)
    # 添加alpha通道
    vertex_colors_rgba = np.zeros((len(vertex_colors_uint8), 4), dtype=np.uint8)
    vertex_colors_rgba[:, :3] = vertex_colors_uint8
    vertex_colors_rgba[:, 3] = 255  # 完全不透明

    colored_mesh.visual.vertex_colors = vertex_colors_rgba

    # 7. 保存带颜色的OBJ
    print("\n7. 保存带颜色的OBJ文件...")
    output_file = 'data/models/stanford_bunny_colored.obj'
    colored_mesh.export(output_file)

    print(f"  ✓ 已保存: {output_file}")

    # 8. 生成NPZ格式（用于采样数据可视化）
    print("\n8. 生成NPZ采样数据...")
    npz_file = 'data/samples/stanford_bunny_textured.npz'
    Path('data/samples').mkdir(parents=True, exist_ok=True)

    sample_data = {
        'points': vertices.astype(np.float32),
        'colors': vertex_colors.astype(np.float32),
        'uvs': uvs.astype(np.float32),
        'faces': faces.astype(np.int32)
    }

    np.savez(npz_file, **sample_data)
    print(f"  ✓ 已保存: {npz_file}")

    print("\n" + "=" * 70)
    print("✓ 纹理烘焙完成！")
    print("=" * 70)

    print("\n生成的文件:")
    print(f"  1. {output_file} - 带顶点颜色的OBJ (可用于viewer)")
    print(f"  2. {npz_file} - 采样数据NPZ (包含points, colors, uvs)")

    print(f"\n现在可以:")
    print(f"  - 查看带颜色的模型: python3 scripts/viewer_3d.py {output_file}")
    print(f"  - 查看采样数据: python3 scripts/viewer_3d.py {npz_file}")
    print(f"  - 对比查看: python3 scripts/viewer_3d.py {output_file} outputs/inference_results/colored_bunny.obj")

    return output_file, npz_file

if __name__ == "__main__":
    bake_texture_to_vertices()