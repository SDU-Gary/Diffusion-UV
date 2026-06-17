# Diffusion-UV

共享纹理约束下低模UV映射的隐式纹理场方案

## 项目说明

本项目实现了一个基于隐式纹理场的低模着色方案，解决了高模和低模共用同一纹理但低模无UV的问题。

### 核心思想

抛弃传统UV映射，学习一个隐式纹理场 \( F: \mathbb{R}^3 \to \mathbb{R}^3 \)，从3D点直接输出颜色。

### 架构

三网络协同系统：

1. **G (Geometry Network)**: 几何感知体积纹理场
   - 预测 SDF 值和低频颜色
   ~ ~0.8M 参数

2. **D (Diffusion Network)**: 条件扩散模型
   - 基于几何条件生成高保真颜色
   - 约 4M 参数

3. **R (Reverse Mapping Network)**: 闭环约束网络
   - 反推几何-纹理联合标签
   - 约 50K 参数

### 训练流程

三阶段渐进训练：

#### 阶段 1 (500-1000 epochs): 训练网络 G
- 优化目标: SDF loss + Eikonal loss + 低频颜色 loss
- 仅训练 G 参数

#### 阶段 2 (200-300 epochs): 训练网络 D
- 固定 G，使用其输出和离线特征
- 标准 DDPM 噪声预测损失

#### 阶段 3 (100-200 epochs): 联合微调
- 同时训练 G + D + R
- 加入反向映射 loss 和熵正则化
- 可选的开环模式

## 项目结构

```
Diffusion-UV/
├── src/
│   ├── models/          # 网络实现
│   ├── data/            # 数据加载和预处理
│   └── training/        # 训练循环和优化器
├── configs/             # 配置文件
├── scripts/             # 训练和评估脚本
├── tests/               # 单元测试
├── cache/               # 缓存的数据和特征
├── logs/                # 训练日志
├── outputs/             # 渲染结果
└── START.md             # 详细设计文档
```

## 快速开始

### 环境配置

```bash
conda create -n uv_mapping python=3.10
conda activate uv_mapping

pip install torch torchvision torchaudio
pip install pytorch3d
pip install open3d trimesh diffusers accelerate wandb mlflow
pip install scikit-learn scipy matplotlib
```

### 配置实验

```python
from src import get_default_config

config = get_default_config()
# 修改配置...
config.save("configs/experiment.yaml")
```

### 训练

```bash
python scripts/train.py --config configs/experiment.yaml
```

### 评估

```bash
python scripts/evaluate.py --config configs/experiment.yaml --checkpoint ./logs/exp/checkpoint.pt
```

## 接口设计

所有核心组件都定义了明确的接口（ABC）和类型系统：

- `IMeshLoader`: Mesh 加载接口
- `IGeometryFeatureExtractor`: 几何特征提取接口
- `INetworkG/D/R`: 网络 G/D/R 接口
- `ILossFunction`: 损失函数接口
- `ITrainer`: 训练器接口
- `IEvaluator`: 评估器接口

详细信息见 `src/interfaces.py`

## 配置系统

配置使用 YAML 格式，支持集中管理和复用：

```yaml
data:
  high_mesh_path: ./data/high.obj
  low_mesh_path: ./data/low.obj
  texture_path: ./data/texture.png
  num_samples_per_epoch: 2000000

network_g:
  hidden_dim: 256
  num_layers: 8
  positional_encoding_freqs: 6

training:
  phase1_epochs: 500
  phase2_epochs: 200
  phase3_epochs: 100
```

## 设计原则

1. **抽象接口优先**: 所有组件通过 ABC 定义清晰契约
2. **类型安全**: 使用 dataclass 和类型注解
3. **配置驱动**: 所有超参数通过配置文件管理
4. **模块化**: 数据、训练、评估相互解耦
5. **可复现性**: 随机数生成器管理

## 参考资料

- START.md: 详细技术设计文档
- torch-ngp: SDF 实现参考
- diffusers: 扩散模型实现参考
- PyTorch3D: 几何处理库

## License

MIT License
# Diffusion-UV
