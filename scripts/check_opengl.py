#!/usr/bin/env python3
"""
OpenGL 环境检查脚本

快速诊断 OpenGL 在当前环境中是否可用
"""

import sys

def main():
    print("=== OpenGL 环境检查 ===\n")

    # 检查 PyOpenGL
    try:
        import OpenGL.GL as gl
        print("✓ PyOpenGL 已安装")
    except ImportError as e:
        print(f"✗ PyOpenGL 未安装: {e}")
        print("  安装命令: pip install PyOpenGL PyOpenGL_accelerate")
        return False

    # 检查显示器环境
    display_env = os.environ.get('DISPLAY', '')
    if display_env:
        print(f"✓ 显示器环境: DISPLAY={display_env}")
    else:
        print("⚠ 无显示器环境 (DISPLAY 未设置)")
        print("  OpenGL 可能需要虚拟显示或 osmesa")

    # 尝试初始化 OpenGL
    print("\n尝试初始化 OpenGL...")
    try:
        # 尝试创建简单上下文
        from OpenGL.GLUT import glutInit, glutInitDisplayMode, glutCreateWindow, GLUT_RGBA, GLUT_DOUBLE, GLUT_DEPTH

        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

        window = glutCreateWindow(b"OpenGL Test")
        print(f"✓ GLUT 窗口创建成功")

        # 检查版本
        version = gl.glGetString(gl.GL_VERSION)
        vendor = gl.glGetString(gl.GL_VENDOR)
        renderer = gl.glGetString(gl.GL_RENDERER)

        print(f"  版本: {version}")
        print(f"  厂商: {vendor}")
        print(f"  渲染器: {renderer}")

        return True

    except Exception as e:
        print(f"✗ OpenGL 初始化失败: {e}")
        print("\n可能的解决方案:")
        print("1. 安装虚拟显示: sudo apt-get install xvfb")
        print("2. 使用 osmesa: pip install PyOpenGL-osmesa")
        print("3. 依赖 CPU 渲染路径 (已可用，推荐)")

        return False

if __name__ == "__main__":
    try:
        import os
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)