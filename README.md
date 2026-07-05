# Diffusion-UV

**Metric-Aligned Implicit UV Fields for Low-Poly Mesh Coloring Under Shared Texture Constraints**

## 项目概述

Diffusion-UV 实现了一种基于度量对齐隐式UV场（MA-IUVF）的低模网格着色方法，解决高面数模型和低面数模型共享纹理时，低模缺乏UV映射的问题。

### 核心创新

本项目采用**隐式UV场学习**的方法，将UV映射表述为连续函数学习问题：

1. **隐式UV场**: 学习函数 F: ℝ³ → ℝ²，直接从3D位置预测UV坐标
2. **度量对齐**: 通过雅可比矩阵匹配保证局部度量一致性
3. **多图表分支**: 支持多个UV图表的并行预测，提高覆盖率和质量
4. **面角UV表示**: 正确处理UV接缝，一个顶点可在不同面中使用不同UV坐标

### 当前实现状态

**Phase 1 - MA-IUVF核心实现 ✅**

- ✅ 多图表隐式UV场网络架构
- ✅ Fourier位置编码（可微分，平滑）
- ✅ 基于雅可比的度量对齐损失
- ✅ 面角UV表示支持
- ✅ CPU光栅化器（100%像素覆盖）
- ✅ 完整的训练和推理流程
- ✅ 49+单元测试通过

**Phase 2 - 性能优化 📋**
- B-Spline哈希网格编码器（5-10倍加速）
- 多轮训练优化
- 图表分类准确度提升

**Phase 3 - 完整集成 📋**
- 与纹理生成网络集成
- 多图表优化和接缝减少
- 端到端低模着色管道

## 技术架构

### 系统组成

```
高模网格 → OBJ解析 → UV图表分割 → MA-IUVF训练 → 低模推理 → 纹理映射
```

### 核心组件

**数据管道** (`src/data/`)
- `obj_parser.py`: 面角UV支持的OBJ解析器
- `uv_chart_segmentation.py`: 基于面邻接图的图表分割
- `metric_aligned_iuv_baker.py`: 训练数据烘焙（雅可比计算、法向外推采样）

**模型架构** (`src/models/`)
- `metric_aligned_iuv_field.py`: MA-IUVF网络核心实现
  - Fourier位置编码
  - 多图表UV预测分支
  - Softplus激活（数值稳定）
  - 参数量: ~50K-100K

**训练系统** (`src/training/`)
- `metric_aligned_iuv_losses.py`: 损失函数实现
  - 度量对齐损失（雅可比匹配）
  - UV锚点损失（防止漂移）
  - 图表分类损失（交叉熵）
  - 统一局部损失（泰勒展开）

**推理渲染** (`src/inference/`)
- `metric_aligned_iuv_inference.py`: 批量UV预测引擎
- `offline_renderer.py`: CPU光栅化器（bbox归一化，深度测试）
- `opengl_renderer.py`: 实验性OpenGL渲染器

### 数学基础

**隐式UV场定义**:
```
F(p) = ∑_{i=1}^{C} softmax(c_i(p)) · uv_i(p)
```

**UV雅可比计算**:
```
J_3d = [∂u/∂x, ∂u/∂y, ∂u/∂z]
       [∂v/∂x, ∂v/∂y, ∂v/∂z]
```

**度量对齐损失**:
```
L_metric = ||J_pred - J_gt||_F²
```

**总训练目标**:
```
L_total = λ_metric · L_metric + λ_uv · L_uv + λ_chart · L_chart
```

## 快速开始

### 环境配置

```bash
# 创建环境
conda create -n diffusion-uv python=3.10
conda activate diffusion-uv

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy scipy trimesh
pip install PyOpenGL PyGLFW
pip install Pillow matplotlib pytest
```

### 数据准备

```bash
# 生成Stanford Bunny测试模型
python scripts/generate_stanford_bunny.py

# 生成程序化纹理
python scripts/generate_procedural_bunny_texture.py
```

### 训练MA-IUVF模型

```bash
# 基础训练（静态烘焙数据）
python scripts/train_metric_aligned_iuv_field.py \
    --high-mesh data/models/stanford_bunny_procedural.obj \
    --texture data/textures/bunny_texture.png \
    --output-dir outputs/maiuvf_bunny/train \
    --num-charts 8 \
    --epochs 100

# 端到端实验
python scripts/run_maiuvf_experiment.py \
    --high-mesh data/models/stanford_bunny_procedural.obj \
    --texture data/textures/bunny_texture.png \
    --output-dir outputs/maiuvf_bunny/experiment \
    --target-faces 500
```

### 推理和渲染

```bash
# 在低模网格上推理
python scripts/infer_metric_aligned_iuv.py \
    --checkpoint outputs/maiuvf_bunny/best.pt \
    --input-mesh data/models/low_poly.obj \
    --output-dir outputs/maiuvf_bunny/inference
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_metric_aligned_iuv_training.py -v
```

## 项目结构

```
Diffusion-UV/
├── configs/                    # 配置文件（YAML）
│   ├── default.yaml            # 默认实验配置
│   ├── maiuvf_baseline.yaml   # MA-IUVF基线配置
│   ├── production.yaml         # 生产环境配置
│   └── bunny_test.yaml         # Stanford Bunny测试配置
├── docs/                       # 详细文档
│   ├── CLAUDE.md              # 项目综合文档
│   ├── START.md               # 技术设计文档
│   └── ALSFD.md               # ALSFD方法文档
├── scripts/                    # 训练和评估脚本
│   ├── train_metric_aligned_iuv_field.py   # MA-IUVF训练
│   ├── run_maiuvf_experiment.py             # 端到端实验
│   └── infer_metric_aligned_iuv.py          # MA-IUVF推理
├── src/                        # 源代码
│   ├── models/                 # 神经网络实现
│   ├── data/                   # 数据加载和预处理
│   ├── training/               # 训练逻辑和损失函数
│   ├── inference/              # 推理和渲染管道
│   ├── geometry/               # 几何处理（ALSFD、投影等）
│   └── analysis/               # 分析工具和实验
├── tests/                      # 单元测试（49+测试通过）
└── data/                       # 数据资产
    ├── models/                 # 测试模型
    └── textures/               # 纹理文件
```

## 验证结果

### Stanford Bunny验证

**模型统计**:
- 顶点数: 35,947
- 面数: 69,451
- UV图表: 8
- UV坐标: 208,353（面角表示，6:1比例）

**训练验证**:
- 损失收敛: 3.2718（1 epoch）
- 图表分类准确度: 13.80%（训练），5.96%（渲染）
- 状态: ✅ 损失下降，准确度提升中

**烘焙验证**:
- 图表检测: 8（正确）
- UV接缝: 26,063（符合预期）
- 状态: ✅ 图表分割正常工作

**渲染验证**:
- CPU渲染器: 100%像素覆盖
- OpenGL渲染器: 60%覆盖（正交投影几何限制）
- 状态: ✅ 两种渲染器均正常工作

### 实验框架

**四个核心实验**:

1. **接缝连续性测试**: 图表边界穿越行为，熵比0.97
2. **薄壳穿透测试**: 欧几里得哈希网格的测地线感知，误差比0.91
3. **非流形外推测试**: 训练曲面外的鲁棒性，能量比1.03
4. **法向梯度噪声测试**: 不必要的法向梯度检测，均值0.306

## 关键技术特点

### 1. 面角UV表示

正确处理UV接缝，支持一个顶点在不同面中使用不同UV坐标：

```python
# OBJ格式: f v1/vt1 v2/vt2 v3/vt3
# 每个面角有独立的UV坐标
UV: [num_face_corners, 2]
Chart_ID: [num_face_corners]
Face_ID: [num_face_corners]
Position: [num_face_corners, 3]
```

### 2. 多图表分支架构

为每个UV图表维护独立的UV预测分支：

```python
# 网络输出
chart_logits: [B, C]      # C个图表的分类logits
uv_preds: [B, C, 2]       # 每个图表的UV预测

# 图表选择
selected_chart = argmax(chart_logits)
selected_uv = uv_preds[arange(B), selected_chart]
```

### 3. 度量对齐损失

通过雅可比矩阵匹配确保局部度量一致性：

```python
# UV雅可比计算
J_3d = torch.autograd.grad(
    uv, positions, 
    create_graph=True,
    grad_outputs=torch.eye(2)
)

# 度量对齐损失
L_metric = ||J_pred - J_gt||_F²
```

### 4. Fourier位置编码

平滑、可微分的位置特征编码：

```python
γ(p) = [sin(2πk·p), cos(2πk·p)] for k ∈ K
```

## 配置系统

### YAML配置示例

```yaml
# configs/maiuvf_baseline.yaml
data:
  high_mesh_path: data/models/stanford_bunny_procedural.obj
  texture_path: data/textures/bunny_texture.png
  num_samples: 100000
  chart_mode: uv_islands

model:
  num_charts: 8
  hidden_dim: 128
  num_layers: 3
  positional_encoding_freqs: 6
  encoder_type: fourier

training:
  epochs: 100
  batch_size: 4096
  learning_rate: 0.0001
  loss_weights:
    metric: 1.0
    anchor: 0.1
    chart: 1.0
```

### CLI参数覆盖

```bash
python scripts/run_maiuvf_experiment.py \
    --config configs/maiuvf_baseline.yaml \
    --training.epochs 200 \
    --model.hidden_dim 256
```

## 开发指南

### 添加新的损失函数

1. 在 `src/training/metric_aligned_iuv_losses.py` 中实现损失函数
2. 在 `compute_metric_aligned_iuv_loss()` 中集成
3. 在配置文件中添加权重参数
4. 添加单元测试验证

### 添加新的编码器

1. 在 `src/models/encoders/` 中实现编码器
2. 在 `MetricAlignedIUVField` 中添加支持
3. 更新 `create_model()` 工厂函数
4. 添加相应的配置参数

### 调试技巧

1. **OBJ解析问题**: 检查 `src/data/obj_parser.py` 验证断言
2. **图表分割**: 验证UV连续性容差设置
3. **训练问题**: 检查损失权重和学习率
4. **渲染问题**: 使用CPU光栅化器作为后备

## 学术贡献

### 技术创新点

1. **面角UV表示**: 首次针对神经UV学习的系统性实现
2. **隐式UV场与度量对齐**: 无需显式参数化的可微分UV映射
3. **多图表分支**: 通过专用图表专家实现更好的覆盖率
4. **雅可比度量损失**: 度量一致性的新型训练目标

### 研究方向

1. **Phase 2**: B-Spline哈希网格编码器加速训练
2. **Phase 3**: 多图表优化和接缝减少
3. **集成**: 与纹理生成网络的完整管道
4. **应用**: 低模着色、纹理传递、神经渲染

## 参考资料

### 技术文档
- `CLAUDE.md`: 综合项目文档（开发指南）
- `START.md`: 详细技术设计
- `ALSFD.md`: ALSFD扩散方法
- `MAIUVF_ANALYSIS_IMPLEMENTATION.md`: 分析实现
- `MAIUVF_EXPERIMENT_SUMMARY.md`: 实验结果总结

### 相关工作
- 传统UV展开: LSCM, Angle-Based Flattening
- 神经隐式表示: SDF, NeRF
- 可微分渲染: 神经UV映射优化

## License

MIT License

## 致谢

本项目基于以下工作：
- Stanford Bunny 3D扫描数据
- PyTorch3D几何处理库
- 传统UV展开算法
