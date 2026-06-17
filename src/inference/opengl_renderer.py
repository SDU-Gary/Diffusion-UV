"""
基于 PyOpenGL 的 G-Buffer 离屏渲染器 (完全重写版)

实现：
1. GLFW hidden window 创建正确的 OpenGL context
2. 多附件 G-Buffer (world_position, face_id, barycentric)
3. 真实的逐像素世界坐标和重心坐标
4. 正确的 face_id 编码 (使用 integer 和 flat 插值)
5. 去掉 geometry shader，使用 vertex + fragment shader
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    OPENGL_AVAILABLE = True
    logger.info("✓ glfw 可用")
except ImportError as e:
    OPENGL_AVAILABLE = False
    logger.warning(f"✗ glfw 不可用: {e}")


class OpenGLGBufferRenderer:
    """
    OpenGL G-Buffer 渲染器 (专业版)

    使用多附件 G-Buffer 技术获取逐像素的精确信息
    """

    def __init__(
        self,
        mesh_vertices: np.ndarray,  # [V, 3]
        mesh_faces: np.ndarray,     # [F, 3]
        resolution: Tuple[int, int] = (512, 512),
        view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        if not OPENGL_AVAILABLE:
            raise RuntimeError("glfw 不可用，请安装: pip install glfw")

        self.mesh_vertices = mesh_vertices.astype(np.float32)
        self.mesh_faces = mesh_faces.astype(np.int32)
        self.width, self.height = resolution

        # 计算视口 bbox。传入 view_bounds 时，高低模可共享同一相机。
        if view_bounds is None:
            self.bbox_min = mesh_vertices.min(axis=0)
            self.bbox_max = mesh_vertices.max(axis=0)
        else:
            self.bbox_min = np.asarray(view_bounds[0], dtype=np.float32)
            self.bbox_max = np.asarray(view_bounds[1], dtype=np.float32)
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2
        self.bbox_size = np.maximum(self.bbox_max - self.bbox_min, 1e-8)
        self.bbox_diagonal = np.linalg.norm(self.bbox_size)

        logger.info(f"OpenGL G-Buffer 渲染器: {resolution}")
        logger.info(f"  顶点: {len(mesh_vertices)}, 面: {len(mesh_faces)}")

        # 初始化 OpenGL
        self._init_opengl_context()
        self._create_gbuffer()
        self._compile_shaders()
        self._create_mesh_buffers()

    def _init_opengl_context(self):
        """创建 GLFW hidden window 和 OpenGL context"""
        try:
            # 初始化 GLFW
            if not glfw.init():
                raise RuntimeError("GLFW 初始化失败")

            # 配置 window
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden window
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

            # 创建 window
            self.window = glfw.create_window(
                self.width, self.height,
                b"OpenGL G-Buffer".decode(),
                None,  # monitor (使用主monitor)
                None   # share (不共享)
            )

            if not self.window:
                glfw.terminate()
                raise RuntimeError("GLFW window 创建失败")

            # 设置当前 context
            glfw.make_context_current(self.window)

            # 验证 context
            version = glGetString(GL_VERSION)
            renderer = glGetString(GL_RENDERER)
            logger.info(f"  OpenGL 版本: {version}")
            logger.info(f"  渲染器: {renderer}")

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"OpenGL context 创建失败: {e}")

    def _create_gbuffer(self):
        """创建多附件 G-Buffer"""
        try:
            # 创建 FBO
            self.fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

            # COLOR0: world_position (RGBA32F)
            self.pos_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.pos_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height,
                         0, GL_RGBA, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.pos_texture, 0)

            # COLOR1: face_id_plus_one (R32UI, 使用 unsigned int)
            self.face_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.face_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, self.width, self.height,
                         0, GL_RED_INTEGER, GL_UNSIGNED_INT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                               GL_TEXTURE_2D, self.face_texture, 0)

            # COLOR2: barycentric (RGB32F)
            self.bary_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.bary_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, self.width, self.height,
                             0, GL_RGB, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,
                               GL_TEXTURE_2D, self.bary_texture, 0)

            # 深度缓冲
            self.depth_buffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                       GL_RENDERBUFFER, self.depth_buffer)

            # 设置 draw buffers
            draw_buffers = [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2]
            glDrawBuffers(3, draw_buffers)

            # 检查 framebuffer 状态
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                status_name = {
                    GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: "INCOMPLETE_ATTACHMENT",
                    GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: "INCOMPLETE_MISSING_ATTACHMENT",
                    GL_FRAMEBUFFER_UNSUPPORTED: "UNSUPPORTED",
                }.get(status, f"UNKNOWN({status})")
                raise RuntimeError(f"G-Buffer 不完整: {status_name}")

            logger.info("  G-Buffer 创建成功")

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"G-Buffer 创建失败: {e}")

    def _compile_shaders(self):
        """编译 G-Buffer shader"""
        # 顶点着色器
        vs = """
        #version 410 core
        layout (location = 0) in vec3 aPosition;
        layout (location = 1) in uint aFaceID;
        layout (location = 2) in vec3 aBarycentric;

        uniform mat4 uMVP;

        out vec3 vWorldPos;
        flat out uint vFaceID;
        noperspective out vec3 vBarycentric;

        void main() {
            vec4 worldPos = vec4(aPosition, 1.0);
            gl_Position = uMVP * worldPos;
            vWorldPos = worldPos.xyz;
            vFaceID = aFaceID;
            vBarycentric = aBarycentric;
        }
        """

        # 片段着色器
        fs = """
        #version 410 core
        in vec3 vWorldPos;
        flat in uint vFaceID;
        noperspective in vec3 vBarycentric;

        layout (location = 0) out vec4 fWorldPos;
        layout (location = 1) out uint fFaceID;
        layout (location = 2) out vec3 fBarycentric;

        void main() {
            fWorldPos = vec4(vWorldPos, 1.0);
            fFaceID = vFaceID;  // 保持原始face_id，背景将是0xFFFFFFFF
            fBarycentric = vBarycentric;
        }
        """

        try:
            vs_shader = compileShader(vs, GL_VERTEX_SHADER)
            fs_shader = compileShader(fs, GL_FRAGMENT_SHADER)
            self.shader_program = compileProgram(vs_shader, fs_shader)
            logger.info("  G-Buffer shader 编译成功")

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Shader 编译失败: {e}")

    def _create_mesh_buffers(self):
        """创建网格数据（带face_id和barycentric）"""
        num_vertices = len(self.mesh_faces) * 3

        # 每个三角形复制3个顶点，每个顶点附带：
        # - world position
        # - face id (uint)
        # - barycentric corner: (1,0,0), (0,1,0), (0,0,1)
        positions = np.zeros(num_vertices * 3, dtype=np.float32)
        face_ids = np.zeros(num_vertices, dtype=np.uint32)
        barycentrics = np.zeros(num_vertices * 3, dtype=np.float32)

        for i, face in enumerate(self.mesh_faces):
            for j in range(3):
                idx = i * 3 + j
                positions[idx*3:(idx+1)*3] = self.mesh_vertices[face[j]]
                face_ids[idx] = i
                # 设置 barycentric 角落
                barycentrics[idx*3:(idx+1)*3] = [
                    1.0 if j == 0 else 0.0,
                    1.0 if j == 1 else 0.0,
                    1.0 if j == 2 else 0.0
                ]

        # 创建 VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # 位置 VBO
        self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Face ID VBO (整数)
        self.vbo_face = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_face)
        glBufferData(GL_ARRAY_BUFFER, face_ids.nbytes, face_ids, GL_STATIC_DRAW)
        glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, 0, None)
        glEnableVertexAttribArray(1)

        # Barycentric VBO
        self.vbo_bary = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_bary)
        glBufferData(GL_ARRAY_BUFFER, barycentrics.nbytes, barycentrics, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        logger.info(f"  网格数据创建: {num_vertices} 顶点")

    def _setup_mvp_matrix(self):
        """设置 MVP 矩阵"""
        # Match the CPU rasterizer's screen-space convention: aspect-preserving
        # orthographic fit, with optional margins on one axis.
        viewport_aspect = float(self.width) / max(float(self.height), 1.0)
        view_height = max(float(self.bbox_size[1]), float(self.bbox_size[0]) / viewport_aspect)
        view_width = view_height * viewport_aspect
        view_size = np.array(
            [max(view_width, 1e-8), max(view_height, 1e-8), max(float(self.bbox_size[2]), 1e-8)],
            dtype=np.float32,
        )
        view_min = np.array(
            [
                self.bbox_center[0] - view_size[0] * 0.5,
                self.bbox_center[1] - view_size[1] * 0.5,
                self.bbox_min[2],
            ],
            dtype=np.float32,
        )

        mvp = np.eye(4, dtype=np.float32)
        mvp[0, 0] = 2.0 / view_size[0]
        mvp[1, 1] = 2.0 / view_size[1]
        mvp[2, 2] = 2.0 / view_size[2]
        mvp[0, 3] = -2.0 * view_min[0] / view_size[0] - 1.0
        mvp[1, 3] = -2.0 * view_min[1] / view_size[1] - 1.0
        mvp[2, 3] = -2.0 * view_min[2] / view_size[2] - 1.0

        return mvp

    def render(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """渲染并读取 G-Buffer"""
        # 绑定 FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

        # 清除
        glClearBufferfv(GL_COLOR, 0, np.array([0, 0, 0, 1], dtype=np.float32))
        # UINT_MAX marks background pixels. Face 0 is a valid face ID.
        glClearBufferuiv(GL_COLOR, 1, np.array([0xFFFFFFFF], dtype=np.uint32))
        glClearBufferfv(GL_COLOR, 2, np.array([0, 0, 0], dtype=np.float32))
        glClearDepth(1.0)
        glClear(GL_DEPTH_BUFFER_BIT)

        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # 使用 shader
        glUseProgram(self.shader_program)

        # 设置 MVP
        mvp = self._setup_mvp_matrix()
        loc = glGetUniformLocation(self.shader_program, "uMVP")
        glUniformMatrix4fv(loc, 1, GL_TRUE, mvp)

        # 绘制
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, len(self.mesh_faces) * 3)

        # 读取 G-Buffer
        world_pos = self._read_texture(self.pos_texture, GL_RGBA, GL_FLOAT)
        face_id = self._read_texture(self.face_texture, GL_RED_INTEGER, GL_UNSIGNED_INT)
        barycentric = self._read_texture(self.bary_texture, GL_RGB, GL_FLOAT)

        # 处理数据
        world_pos = world_pos[:, :, :3]  # [H, W, 3]
        face_id = face_id[:, :, 0]   # [H, W]
        barycentric = barycentric    # [H, W, 3]

        # 翻转 Y 轴
        world_pos = np.flipud(world_pos)
        face_id = np.flipud(face_id)
        barycentric = np.flipud(barycentric)

        # 标记无效像素为-1 (0xFFFFFFFF转换为有符号int)
        invalid_mask = face_id == np.uint32(0xFFFFFFFF)
        face_id = face_id.astype(np.int32)
        face_id[invalid_mask] = -1

        # 统计有效像素
        valid_pixels = np.sum(face_id >= 0)
        total_pixels = self.width * self.height

        logger.info(f"  OpenGL 渲染: {valid_pixels}/{total_pixels} 像素有效")

        return world_pos, face_id, barycentric

    def _read_texture(self, texture, format, dtype):
        """读取纹理数据"""
        # 设置读取缓冲区
        if format == GL_RED_INTEGER:
            glReadBuffer(GL_COLOR_ATTACHMENT1)
        elif format == GL_RGB:
            glReadBuffer(GL_COLOR_ATTACHMENT2)
        else:
            glReadBuffer(GL_COLOR_ATTACHMENT0)

        type_enum = {
            np.float32: GL_FLOAT,
            GL_FLOAT: GL_FLOAT,
            np.uint32: GL_UNSIGNED_INT,
            GL_UNSIGNED_INT: GL_UNSIGNED_INT,
        }.get(dtype, dtype)

        data = glReadPixels(0, 0, self.width, self.height, format, type_enum)

        # 转换numpy dtype
        np_dtype = np.float32 if dtype in [GL_FLOAT, np.float32] else np.uint32
        return np.frombuffer(data, dtype=np_dtype).reshape(self.height, self.width, -1)

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'fbo'):
            glDeleteFramebuffers(1, [self.fbo])
        if hasattr(self, 'pos_texture'):
            glDeleteTextures(1, [self.pos_texture])
        if hasattr(self, 'face_texture'):
            glDeleteTextures(1, [self.face_texture])
        if hasattr(self, 'bary_texture'):
            glDeleteTextures(1, [self.bary_texture])
        if hasattr(self, 'depth_buffer'):
            glDeleteRenderbuffers(1, [self.depth_buffer])
        if hasattr(self, 'vao'):
            glDeleteVertexArrays(1, [self.vao])
        if hasattr(self, 'vbo_pos'):
            glDeleteBuffers(1, [self.vbo_pos])
        if hasattr(self, 'vbo_face'):
            glDeleteBuffers(1, [self.vbo_face])
        if hasattr(self, 'vbo_bary'):
            glDeleteBuffers(1, [self.vbo_bary])
        if hasattr(self, 'shader_program'):
            glDeleteProgram(self.shader_program)

        if hasattr(self, 'window') and self.window:
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()

        logger.info("  OpenGL 资源已清理")


def render_with_opengl_gbuffer(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    resolution: Tuple[int, int] = (512, 512),
    view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 OpenGL G-Buffer 渲染网格

    Args:
        mesh_vertices: [V, 3] 网格顶点
        mesh_faces: [F, 3] 网格面
        resolution: 渲染分辨率 (width, height)

    Returns:
        world_pos: [H, W, 3] 世界坐标
        face_ids: [H, W] 面ID
        barycentric: [H, W, 3] 重心坐标
    """
    renderer = OpenGLGBufferRenderer(mesh_vertices, mesh_faces, resolution, view_bounds=view_bounds)

    try:
        return renderer.render()
    finally:
        renderer.cleanup()


if __name__ == "__main__":
    # 测试 G-Buffer 渲染器
    import sys
    if len(sys.argv) > 1:
        from .obj_parser import parse_obj_file

        obj_path = sys.argv[1]
        obj_data = parse_obj_file(obj_path)
        vertices = obj_data['vertices']
        faces = obj_data['faces']

        world_pos, face_ids, barycentric = render_with_opengl_gbuffer(vertices, faces, (128, 128))

        print(f"G-Buffer 渲染完成")
        print(f"  有效像素: {np.sum(face_ids >= 0)} / {face_ids.size}")
        print(f"  World pos 范围: {world_pos.min(axis=(0,1))} - {world_pos.max(axis=(0,1))}")
        print(f"  Barycentric 和检查: {np.abs(barycentric.sum(axis=2) - 1.0).max():.6f}")
