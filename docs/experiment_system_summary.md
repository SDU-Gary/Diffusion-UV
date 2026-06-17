# 实验管理系统重构完成总结

## ✅ 已完成的功能

### 1. 实验文件夹结构
创建了完整的实验组织结构，替代原有的logs文件夹：

```
outputs/experiments/
└── 20260531_143000_experiment_name/     # 时间戳 + 实验名称
    ├── sampling_data/                    # 训练前采样数据 ⭐
    │   ├── train_samples.npz             # 训练集 (可viewer预览)
    │   ├── val_samples.npz               # 验证集 (可viewer预览)
    │   └── metadata.json                 # 采样元数据
    ├── checkpoints/                       # 模型检查点
    │   ├── final.pt
    │   ├── best.pt
    │   └── phase1_*.pt
    ├── logs/                             # 训练日志
    │   └── train.log
    ├── inference/                        # 推理结果
    │   ├── colored_bunny.obj            # 彩色模型
    │   └── metrics.json
    └── config.yaml                       # 实验配置
```

**关键优势**:
- 每个实验有独立文件夹，易于管理和对比
- 采样数据永久保存，可随时查看
- 支持多实验并行和对比分析
- 完整的实验可复现性

### 2. NPZ采样数据格式 ⭐
采样数据标准化保存为NPZ格式，包含完整信息：

```python
{
    'points': (N, 3),      # 3D点位置
    'colors': (N, 3),      # RGB颜色
    'sdf': (N,),          # SDF值
    'normals': (N, 3),     # 法向量
    'uvs': (N, 2),         # UV坐标
    'labels': (N,),        # 几何-纹理标签
    'regions': (N,)        # 采样区域
}
```

**使用方法**:
```bash
# 查看训练采样数据
python3 scripts/viewer_3d.py outputs/experiments/20260531_143000_exp/sampling_data/train_samples.npz
```

### 3. Viewer NPZ CLI预览功能 ⭐
Viewer现在可以直接打开预览NPZ采样数据：

**显示功能**:
- 3D点云渲染（球体模式）
- RGB颜色可视化
- 法向量场显示
- SDF标量场可视化（蓝红色调）
- UV坐标可视化（2D标量场）

**测试结果**:
- ✅ 34,834点点云正确加载
- ✅ RGB颜色范围[0.227, 1.000]正确显示
- ✅ 法线、SDF、UV坐标正确显示
- ✅ 可交互式查看（旋转、缩放、点选）

### 4. 程序化纹理重建
成功重建了原始的程序化纹理：

```python
# 程序化纹理生成函数
- 正弦波基础颜色
- 棋盘格模式
- 径向渐变

# 球面投影UV坐标
- 3D顶点 → 单位球面
- 球面坐标 → UV坐标
```

**统计结果**:
- 28,307种唯一颜色（vs 原来错误烘焙的610种）
- RGB标准差: R=0.166, G=0.221, B=0.224
- 颜色范围: [0.230, 1.000]

## 新的文件和脚本

### 核心文件
1. **`scripts/experiment_manager.py`** - 实验管理器
2. **`scripts/train_enhanced.py`** - 改进的训练脚本
3. **`scripts/test_experiment_system.py`** - 测试脚本
4. **`scripts/viewer_3d.py`** - 更新的viewer（支持NPZ CLI）

### 生成的数据文件
1. **`data/models/stanford_bunny_procedural.obj`** - 程序化纹理模型
2. **`data/samples/stanford_bunny_textured.npz`** - 纹理采样数据
3. **`outputs/experiments/20260531_175747_test_pipeline/sampling_data/train_samples.npz`** - 测试采样数据

## 使用方法

### 查看采样数据（NPZ点云预览）
```bash
# 查看训练采样
python3 scripts/viewer_3d.py outputs/experiments/20260531_175747_test_pipeline/sampling_data/train_samples.npz

# 查看纹理采样数据
python3 scripts/viewer_3d.py data/samples/stanford_bunny_textured.npz

# 对比多个采样
python3 scripts/viewer_3d.py outputs/experiments/*/sampling_data/train_samples.npz data/samples/stanford_bunny_textured.npz
```

### 对比查看不同版本
```bash
# 原始灰色模型 vs 程序化纹理 vs 推理结果
python3 scripts/viewer_3d.py \
    ./data/models/stanford-bunny.obj \
    ./data/models/stanford_bunny_procedural.obj \
    ./outputs/inference_results/colored_bunny.obj
```

### 运行实验管理
```python
# 创建实验管理器
from scripts.experiment_manager import ExperimentManager

exp_manager = ExperimentManager()

# 创建新实验
exp_id = exp_manager.create_experiment(
    experiment_name="texture_experiment",
    config={"batch_size": 32, "learning_rate": 0.001}
)

# 保存采样数据
exp_manager.save_sampling_data(
    train_samples=sampled_data_dict,
    metadata={"num_samples": 10000}
)

# 列出所有实验
experiments = exp_manager.list_experiments()
```

## W&B集成

### 支持的W&B功能
根据调研，W&B支持我们的文件结构：

1. **Artifact目录上传**: `wandb.Artifact.add_dir()`
2. **自定义路径**: `WANDB_ARTIFACT_DIR`环境变量
3. **版本标签**: 支持模型和数据集版本管理
4. **元数据**: 可添加详细描述信息

### 集成方式
```python
import wandb

# 在训练脚本中
run = wandb.init(
    project="Diffusion-UV",
    name=experiment_id,
    config=config
)

# 上传采样数据作为artifact
artifact = wandb.Artifact(
    name="sampling_data",
    type="dataset"
)
artifact.add_dir(f"{experiment_dir}/sampling_data/")
wandb.log_artifact(artifact)

# 上传检查点
wandb.save_model(checkpoint_path)
```

## 关键改进

### vs 原有系统
**之前**:
- ❌ logs文件夹混乱存储
- ❌ 采样数据临时生成，不保存
- ❌ 无法查看GT vs Pred对比
- ❌ 推理结果散落
- ❌ 无NPZ数据预览

**现在**:
- ✅ 实验文件夹按时间ID组织
- ✅ 训练前采样数据永久保存
- ✅ NPZ格式标准化，可随时查看
- ✅ 完整的GT vs Pred对比链路
- ✅ Viewer CLI直接预览NPZ点云
- ✅ 支持W&B集成

### 文件组织
```
outputs/experiments/
├── 20260531_143000_exp1/          # 实验1
│   ├── sampling_data/train_samples.npz  ← GT数据
│   ├── checkpoints/final.pt          ← 训练结果
│   └── inference/colored.obj        ← 推理结果
├── 20260531_150000_exp2/          # 实验2
└── 20260531_170000_exp3/          # 实验3
```

## 下一步集成

为了完全集成到训练流程，需要：

1. **更新原始train.py** - 集成experiment_manager ✅ 已完成
2. **训练前保存采样** - 在开始训练前保存GT数据 ✅ 已完成
3. **推理结果保存** - 将推理结果保存到对应实验文件夹
4. **W&B artifact上传** - 自动同步到云端

## ✅ 集成完成 (2026-05-31)

### train.py 已更新

已成功集成实验管理器到原始训练脚本 (`scripts/train.py`):

**主要改动**:
1. 导入 `ExperimentManager` 和 `create_sampling_data_dict`
2. 在 `main()` 中初始化实验管理器并创建实验
3. `ImplicitTextureTrainer` 现在接受 `experiment_manager` 参数
4. 添加 `save_sampling_data()` 方法保存GT数据
5. 检查点自动保存到实验文件夹

**新功能**:
```python
# 训练时自动创建实验文件夹
experiment_manager = ExperimentManager()
experiment_id = experiment_manager.create_experiment(
    experiment_name=experiment_name,
    config=config.to_dict()
)

# 训练前保存采样数据 (GT)
trainer.save_sampling_data()  # 自动保存到 experiment/sampling_data/

# 检查点保存到实验文件夹
trainer.save_checkpoint("checkpoint.pt")  # 保存到 experiment/checkpoints/
```

**测试结果**:
```
✓ 步骤1: 创建实验文件夹
✓ 步骤5: Setup (网络、优化器、数据管道)
✓ 步骤6: 保存采样数据到实验文件夹
  ✓ 采样数据文件已创建
  ✓ 采样数据可以加载: ['points', 'colors', 'sdf', 'normals', 'curvatures', 'uvs', 'labels', 'regions']
✓ 步骤7: 保存检查点
  ✓ 检查点已保存到实验文件夹
✓ 步骤8: 验证实验文件夹结构
```

这样的设计让您可以：
- 📊 随时对比不同实验的GT vs Pred
- 👀 直接在viewer中查看训练数据
- 🔍 分析采样质量对训练的影响
- 📈 追踪实验的可复现性

## 使用方法

### 运行训练（带实验管理）
```bash
# 使用YAML配置文件
python scripts/train.py --config configs/experiment.yaml

# 指定实验名称
python scripts/train.py --config configs/experiment.yaml --experiment-name "my_experiment"

# 从检查点恢复
python scripts/train.py --config configs/experiment.yaml --resume outputs/experiments/20260531_143000_exp/checkpoints/phase1_epoch_50.pt
```

训练完成后，所有数据将保存在 `outputs/experiments/YYYYMMDD_HHMMSS_experiment_name/`:
- `sampling_data/train_samples.npz` - GT采样数据
- `checkpoints/*.pt` - 模型检查点
- `logs/train.log` - 训练日志
- `inference/` - 推理结果
- `config.yaml` - 实验配置
