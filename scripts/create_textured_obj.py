#!/usr/bin/env python3
"""
创建带UV坐标的OBJ文件，用于纹理可视化

读取NPZ中的UV数据和纹理贴图，生成完整的带纹理OBJ文件
"""

import numpy as np
import trimesh
from pathlib import Path
import struct

def create_textured_obj():
    """创建带UV坐标和材质的OBJ文件"""

    print("=" * 70)
    print("创建带纹理的Stanford Bunny OBJ文件")
    print("=" * 70)

    # 1. 读取原始OBJ文件
    print("\n1. 读取原始OBJ文件...")
    obj_file = 'data/models/stanford-bunny.obj'

    # 使用 trimesh 的加载结果，而不是直接解析 OBJ 文本。
    # 原始 Stanford bunny OBJ 文本中存在重复/未引用顶点；trimesh 加载后
    # 的顶点和面索引空间才与 data/textures/bunny_uvs.npz 对齐。
    mesh = trimesh.load(obj_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    print(f"  顶点数: {len(vertices)}")
    print(f"  面数: {len(faces)}")

    # 2. 读取UV坐标
    print("\n2. 读取UV坐标...")
    uv_data = np.load('data/textures/bunny_uvs.npz')
    uvs = uv_data['uvs']
    uv_faces = uv_data['faces']

    print(f"  UV坐标数: {len(uvs)}")
    print(f"  UV面数: {len(uv_faces)}")
    print(f"  UV范围: u[{uvs[:,0].min():.3f}, {uvs[:,0].max():.3f}] v[{uvs[:,1].min():.3f}, {uvs[:,1].max():.3f}]")

    # 3. 验证数据一致性
    print("\n3. 验证数据一致性...")
    if len(vertices) == len(uvs):
        print(f"  ✓ 顶点数与UV数匹配: {len(vertices)}")
    else:
        raise ValueError(f"顶点数与UV数不匹配: {len(vertices)} vs {len(uvs)}")

    if len(faces) == len(uv_faces):
        print(f"  ✓ 面数匹配: {len(faces)}")
    else:
        raise ValueError(f"面数不匹配: {len(faces)} vs {len(uv_faces)}")

    if np.array_equal(faces, uv_faces.astype(faces.dtype)):
        print("  ✓ OBJ faces 与 UV faces 完全一致")
    else:
        raise ValueError("OBJ faces 与 UV faces 不一致，不能安全写入逐顶点 UV")

    # 4. 创建材质文件
    print("\n4. 创建材质文件...")
    mtl_content = """# Stanford Bunny Material
newmtl bunny_material
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.0 0.0 0.0
Ns 10.0
d 1.0
map_Kd ../textures/bunny_texture.png
"""

    mtl_file = 'data/models/stanford_bunny.mtl'
    with open(mtl_file, 'w') as f:
        f.write(mtl_content)
    print(f"  ✓ 材质文件已创建: {mtl_file}")

    # 5. 创建带UV的OBJ文件
    print("\n5. 创建带UV的OBJ文件...")
    output_obj = 'data/models/stanford_bunny_textured.obj'

    with open(output_obj, 'w') as f:
        # 写入头部
        f.write("# Stanford Bunny with UV coordinates\n")
        f.write("# Generated for texture visualization\n")
        f.write(f"mtllib stanford_bunny.mtl\n")
        f.write("usemtl bunny_material\n")

        # 写入顶点
        f.write(f"# {len(vertices)} vertices\n")
        for i, v in enumerate(vertices):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # 写入UV坐标
        f.write(f"# {len(uvs)} texture coordinates\n")
        for i, uv in enumerate(uvs):
            # OpenGL纹理坐标系，v轴可能需要翻转
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # 写入面（包含UV索引）
        f.write(f"# {len(faces)} faces\n")
        for i, face in enumerate(faces):
            if i < len(uv_faces):
                uv_face = uv_faces[i]
                # OBJ索引从1开始，格式：vertex_index/uv_index
                face_line = "f"
                for j in range(len(face)):
                    v_idx = face[j] + 1
                    uv_idx = uv_face[j] + 1
                    face_line += f" {v_idx}/{uv_idx}"
                f.write(face_line + "\n")
            else:
                # 如果没有对应的UV，使用顶点索引
                face_line = "f"
                for j in range(len(face)):
                    v_idx = face[j] + 1
                    face_line += f" {v_idx}/{v_idx}"
                f.write(face_line + "\n")

    print(f"  ✓ 带UV的OBJ文件已创建: {output_obj}")

    # 6. 检查纹理图片
    print("\n6. 检查纹理图片...")
    from PIL import Image
    try:
        tex_img = Image.open('data/textures/bunny_texture.png')
        print(f"  ✓ 纹理图片: {tex_img.size} {tex_img.mode}")
    except Exception as e:
        print(f"  ⚠ 纹理图片错误: {e}")

    print("\n" + "=" * 70)
    print("✓ 带纹理的模型创建完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print(f"  1. {output_obj} - 带UV坐标的OBJ文件")
    print(f"  2. {mtl_file} - 材质文件")
    print(f"  3. 纹理贴图 - data/textures/bunny_texture.png")

    print(f"\n现在可以在viewer中查看带纹理的模型:")
    print(f"  python3 scripts/viewer_3d.py {output_obj}")

    return output_obj

if __name__ == "__main__":
    create_textured_obj()
