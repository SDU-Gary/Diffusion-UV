#!/usr/bin/env python3
"""
OpenGL 诊断脚本

检查 OpenGL 环境和具体问题
"""

import sys

def check_opengl():
    """检查 OpenGL 环境"""
    print("=== OpenGL 环境诊断 ===")

    try:
        import OpenGL.GL as gl
        print("✓ PyOpenGL 已安装")

        # 检查版本
        try:
            version = gl.glGetString(gl.GL_VERSION)
            print(f"✓ OpenGL 版本: {version}")
        except:
            print("✗ 无法获取 OpenGL 版本")

        # 尝试创建上下文
        try:
            from OpenGL.GLUT import glutInit, glutCreateWindow, glutDestroyWindow
            glutInit()
            window = glutCreateWindow(b"OpenGL Test")
            print(f"✓ GLUT 窗口创建成功: {window}")
            glutDestroyWindow(window)
        except Exception as e:
            print(f"✗ GLUT 窗口创建失败: {e}")

        # 检查具体能力
        try:
            from pyglet.gl import gl_info
            print(f"✓ GLSL 版本: {gl_info.get('GL_SHADING_LANGUAGE_VERSION')}")
            print(f"✓ 最大纹理尺寸: {gl_info.get('GL_MAX_TEXTURE_SIZE')}")
            print(f"✓ 最大颜色附件: {gl_info.get('GL_MAX_COLOR_ATTACHMENTS')}")
        except ImportError:
            print("⚠ pyglet 未安装，跳过详细检查")
        except Exception as e:
            print(f"✗ 获取能力信息失败: {e}")

    except ImportError as e:
        print(f"✗ PyOpenGL 未安装: {e}")
        return False

    return True

def test_simple_framebuffer():
    """测试简单帧缓冲区创建"""
    print("\n=== 帧缓冲区测试 ===")

    try:
        import OpenGL.GL as gl
        import numpy as np

        # 尝试创建简单的帧缓冲区
        print("尝试创建简单帧缓冲区...")

        # 创建 FBO
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        print("✓ FBO 创建成功")

        # 创建纹理
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 256, 256, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)
        print("✓ 纹理附件创建成功")

        # 创建深度缓冲
        depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, 256, 256)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth)
        print("✓ 深度缓冲创建成功")

        # 检查状态
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        status_name = {
            gl.GL_FRAMEBUFFER_COMPLETE: "COMPLETE",
            gl.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: "INCOMPLETE_ATTACHMENT",
            gl.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: "INCOMPLETE_MISSING_ATTACHMENT",
            gl.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: "INCOMPLETE_DRAW_BUFFER",
            gl.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: "INCOMPLETE_READ_BUFFER",
            gl.GL_FRAMEBUFFER_UNSUPPORTED: "UNSUPPORTED",
            gl.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: "INCOMPLETE_MULTISAMPLE",
            gl.GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: "INCOMPLETE_LAYER_TARGETS",
            0: "UNKNOWN_ERROR"
        }.get(status, f"UNKNOWN({status})")

        if status == gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"✓ 帧缓冲区状态: {status_name}")
        else:
            print(f"✗ 帧缓冲区状态: {status_name}")

        # 清理
        gl.glDeleteFramebuffers(1, [fbo])
        gl.glDeleteTextures(1, [texture])
        gl.glDeleteRenderbuffers(1, [depth])
        print("✓ 清理成功")

        return status == gl.GL_FRAMEBUFFER_COMPLETE

    except Exception as e:
        print(f"✗ 帧缓冲区测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integer_texture():
    """测试整数纹理支持"""
    print("\n=== 整数纹理测试 ===")

    try:
        import OpenGL.GL as gl

        # 创建整数纹理
        print("尝试创建整数纹理...")

        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32I, 256, 256, 0, gl.GL_RED_INTEGER, gl.GL_INT, None)
        print("✓ R32I 纹理创建成功")

        # 测试清除
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)

        # 设置绘制缓冲区
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])

        # 尝试清除
        gl.glClearBufferiv(gl.GL_COLOR, 0, np.array([-1], dtype=np.int32))
        print("✓ 整数缓冲区清除成功")

        # 检查状态
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status == gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"✓ 帧缓冲区状态: COMPLETE")
        else:
            print(f"✗ 帧缓冲区状态: {status}")

        # 清理
        gl.glDeleteFramebuffers(1, [fbo])
        gl.glDeleteTextures(1, [texture])

        return status == gl.GL_FRAMEBUFFER_COMPLETE

    except Exception as e:
        print(f"✗ 整数纹理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("OpenGL 诊断工具\n")

    # 检查环境
    if not check_opengl():
        print("\n❌ OpenGL 环境检查失败")
        return 1

    # 初始化 GLUT 窗口
    try:
        from OpenGL.GLUT import glutInit, glutInitDisplayMode, glutInitWindowSize, glutCreateWindow, glutHideWindow
        glutInit()
        glutInitDisplayMode(gl.GLUT_DOUBLE | gl.GLUT_RGBA | gl.GLUT_DEPTH)
        glutInitWindowSize(256, 256)
        window = glutCreateWindow(b"OpenGL Diagnostic")
        glutHideWindow()
        print("✓ 诊断窗口创建成功")
    except Exception as e:
        print(f"✗ 窗口创建失败: {e}")
        return 1

    # 运行测试
    results = []
    results.append(("帧缓冲区", test_simple_framebuffer()))
    results.append(("整数纹理", test_integer_texture()))

    # 总结
    print("\n=== 诊断结果 ===")
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}: {'通过' if result else '失败'}")

    if all(r[1] for r in results):
        print("\n✓ 所有测试通过，OpenGL 环境正常")
        return 0
    else:
        print("\n✗ 部分测试失败，OpenGL 环境可能有问题")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as e:
        print(f"\n❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    input("\n按回车键退出...")
    sys.exit(exit_code)