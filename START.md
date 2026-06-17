# 共享纹理约束下低模UV映射的隐式纹理场方案

## START 文档

> **版本**：1.0
> **更新日期**：2026-05-30
> **状态**：设计定稿，待代码实现

---

## 1. 问题定义

### 1.1 背景
在三维资产管线中，存在一个高频但难以自动化的需求：
- 高模 \( \mathcal{H} \) 拥有固定颜色纹理 \( T: \Omega \to \mathbb{R}^3 \) 和已知 UV 映射 \( \phi_H: \mathcal{H} \to \Omega \)
- 低模 \( \mathcal{L} \) 由高模简化得到（几何固定，无 UV）
- **约束**：\( \mathcal{L} \) 必须与 \( \mathcal{H} \) **共用同一张纹理** \( T \)，不能重烘焙或生成新纹理
- **目标**：找到 \( \mathcal{L} \) 的某种着色方式，使其渲染外观尽可能逼近 \( \mathcal{H} \)

### 1.2 核心难点
- 传统 UV 映射优化是**高度病态**的：几何简化破坏了纹理与表面的精确对齐，且优化目标（颜色匹配）与约束（UV 连续性、低失真）本质冲突。
- 直接为低模求解连续 UV 映射 \( \phi_L: \mathcal{L} \to \Omega \) 在数学上**不适定**（ill-posed）。

### 1.3 解决方向
抛弃“低模必须有 UV 映射”的思维，转而采用**隐式纹理场**：
> 学习一个函数 \( F: \mathbb{R}^3 \to \mathbb{R}^3 \)，使得对于任意点 \( x \in \mathcal{L} \)，\( F(x) \) 直接输出颜色。
完全绕开 UV 映射，将问题转化为函数回归（良态问题）。

---

## 2. 解决思路概述

### 2.1 核心思想
将高模表面的纹理颜色“烘焙”到一个**3D 空间中的连续颜色场**中，低模只需查询该场。
该场通过神经网络隐式表示，并引入 **有符号距离场（SDF）** 和 **条件扩散模型** 来同时解决几何偏移和高频纹理两大挑战。

### 2.2 双分支协同架构
我们设计了一个三网络协同系统：

| 网络 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **G** (Geometry Network) | 几何感知体积纹理场 | 3D 点坐标 \( x \) | 有符号距离 \( s(x) \) + 低频基色 \( c_{\text{base}}(x) \) |
| **D** (Diffusion Network) | 条件扩散模型 | 条件向量 \( \mathbf{y} \) (含 \( c_{\text{base}}, s \), 几何特征) + 噪声 \( z \) | 高保真颜色 \( \hat{c}(x) \) |
| **R** (Reverse Mapping Network) | 闭环约束网络 | 点坐标 \( x \) + 最终颜色 \( \hat{c}(x) \) | 几何-纹理联合标签 \( \ell(x) \) |

> 闭环约束强制 \( D \) 生成的纹理必须包含足够几何上下文信息，从根本上对抗幻觉纹理。

---

## 3. 架构详细设计

### 3.1 网络 G：几何感知体积纹理场

**目标**：为任意 3D 点预测 SDF 值和低频颜色。

**结构**（PyTorch 风格描述）：

```
Input: x (3D) → Positional Encoding (L=6) → [36-dim]
       ↓
MLP: Linear(36 → 256) + ReLU
     Linear(256 → 256) + ReLU
     Linear(256 → 256) + ReLU
     Linear(256 → 256) + ReLU   ← skip connection from first layer
     Linear(256+36 → 256) + ReLU
     Linear(256 → 256) + ReLU
     Linear(256 → 256) + ReLU
     Linear(256 → 256) + ReLU
       ↓
Split into two heads:
  - SDF head: Linear(256 → 1) + tanh   → s(x)
  - Color head: Linear(256 → 3) + sigmoid → c_base(x)
```

**关键参数**：
- 位置编码频率数 \( L_{\text{pos}} = 6 \)（扩展后 36 维）
- 总参数量 ≈ 0.8M

### 3.2 网络 D：条件扩散模型

**目标**：在给定条件 \( \mathbf{y} \) 下，将噪声逐步去噪为高保真颜色。

**条件向量 \( \mathbf{y} \)** 组成：

| 特征 | 符号 | 维数 | 来源 |
|------|------|------|------|
| 低频基色 | \( c_{\text{base}} \) | 3 | 网络 G 输出 |
| 有符号距离 | \( s \) | 1 | 网络 G 输出 |
| 主曲率 | \( \kappa_1, \kappa_2 \) | 2 | 从 SDF 的 Hessian 计算 |
| 法线 | \( \mathbf{n} \) | 3 | SDF 梯度归一化 |
| 边界距离 | \( d_{\text{boundary}} \) | 1 | 测地距离到 UV 图表边界 |
| 全局形状编码 | \( \mathbf{z}_{\text{global}} \) | 32 | PointNet++ 对低模编码 |

总维数 \( d_y = 3+1+2+3+1+32 = 42 \)

**扩散模型结构**（基于 `diffusers` 的轻量 UNet）：

- 输入：噪声颜色 \( \epsilon \) (3 维)
- 条件注入：使用 **FiLM** 在每个残差块中调制特征图
- 时间步编码：Sinusoidal embedding (512 维)
- 网络容量：约 4M 参数（下采样层数 3，通道基数 64）

**扩散调度**：
- 训练步数 \( T = 1000 \)，线性噪声调度
- 推理使用 DDIM，\( T_{\text{inf}} = 20 \) 步

### 3.3 网络 R：闭环约束网络

**目标**：从最终颜色反推几何-纹理联合标签，提供监督信号防止幻觉。

**结构**：

```
Input: concat(x (3), c (3)) → [6-dim]
       ↓
Positional Encoding (L=6) → [6*2*6 = 72-dim]
       ↓
MLP: Linear(72 → 256) + ReLU
     Linear(256 → 128) + ReLU
     Linear(128 → 64) + ReLU
     Linear(64 → K)     (no activation)
       ↓
Output: logits → softmax → probability over K classes
```

- \( K \)：几何-纹理联合类别数（通过 K-means 聚类得到，典型值 16~32）
- 参数量 ≈ 50K

---

## 4. 几何上下文特征提取

所有特征在训练前离线计算并缓存，以节省训练时间。

| 特征 | 计算方法 | 工具 |
|------|----------|------|
| 主曲率 \( \kappa_1, \kappa_2 \) | 对 SDF 的 Hessian 矩阵特征值分解 | PyTorch3D / 自定义 CUDA 核 |
| 法线 \( \mathbf{n} \) | SDF 梯度归一化 | 自动微分 |
| 边界距离 \( d_{\text{boundary}} \) | 点在 UV 空间到最近图表边界（3D 映射回测地距离） | Open3D + trimesh |
| 全局形状编码 \( \mathbf{z}_{\text{global}} \) | PointNet++ 对低模点云编码 | 官方 PointNet++ 实现 |

---

## 5. 训练流程

### 5.1 数据准备

**采样策略**（每个 epoch 采样固定点数，如 2M 点）：

| 区域 | 比例 | 采样方法 |
|------|------|----------|
| 高模表面 (\( s=0 \)) | 40% | 三角形面积均匀采样 |
| 表面附近 (\( |s|<\epsilon \)) | 40% | 法线方向高斯偏移 (\(\sigma=0.01 \times \text{包围盒大小}\)) |
| 外部 (\( s>\epsilon \)) | 10% | 包围盒内均匀采样 |
| 内部 (\( s<-\epsilon \)) | 10% | 包围盒内均匀采样 |

**每个样本存储**：\( x, s_{\text{gt}}, c_{\text{gt}}, c_{\text{lowpass}}, \kappa_1, \kappa_2, \mathbf{n}, d_{\text{boundary}}, \ell_{\text{gt}} \)
- \( c_{\text{gt}} \)：表面点从纹理采样，非表面点使用最近表面点颜色
- \( c_{\text{lowpass}} \)：\( c_{\text{gt}} \) 经高斯滤波（\( \sigma=5 \) 像素）
- \( \ell_{\text{gt}} \)：通过 K-means 对表面点的 (\( \kappa_1, \kappa_2, c_{\text{gt}} \)) 聚类得到

### 5.2 三阶段渐进训练

#### **阶段 1：训练网络 G**（500~1000 epochs）
- 优化目标：\( \mathcal{L}_{\text{Geo}} = \lambda_{\text{SDF}}\|s-s_{\text{gt}}\|_1 + \lambda_{\text{Eikonal}}(\|\nabla s\|_2-1)^2 + \lambda_{\text{Color}}\|c_{\text{base}}-c_{\text{lowpass}}\|_2^2 \)
- 优化器：Adam，lr=5e-4，batch size=65536
- 仅更新 G 参数

#### **阶段 2：训练网络 D**（200~300 epochs）
- 固定 G，使用其输出 \( c_{\text{base}}, s \) 并加上离线特征构造 \( \mathbf{y} \)
- 标准 DDPM 噪声预测损失：
  \[
  \mathcal{L}_{\text{Diff}} = \mathbb{E}_{t,\epsilon} \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}c_{\text{gt}}+\sqrt{1-\bar{\alpha}_t}\epsilon, t, \mathbf{y})\|_2^2
  \]
- 优化器：AdamW，lr=1e-4，batch size=32768

#### **阶段 3：联合微调 + 闭环约束**（100~200 epochs）
- 同时训练 G、D、R，加入：
  - 反向损失：\( \mathcal{L}_{\text{Reverse}} = \text{CrossEntropy}(\hat{\ell}, \ell_{\text{gt}}) \)
  - 熵正则化：\( \mathcal{L}_{\text{Entropy}} = -\mathbb{E}_{x}[ \sum_c p_D(c|\mathbf{y}) \log p_D(c|\mathbf{y}) ] \)
- 总损失：\( \mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Geo}} + \lambda_{\text{Diff}}\mathcal{L}_{\text{Diff}} + \lambda_{\text{Reverse}}\mathcal{L}_{\text{Reverse}} + \lambda_{\text{Entropy}}\mathcal{L}_{\text{Entropy}} \)
- 超参数：\( \lambda_{\text{Reverse}} \) 从 0.1 逐渐升至 0.5；\( \lambda_{\text{Entropy}} \) 从 0 渐增至 0.05
- 使用梯度归一化处理冲突

### 5.3 推理过程（应用于低模 \( \mathcal{L} \)）

1. 对于低模每个顶点 \( v_i \)，计算 \( s(v_i), c_{\text{base}}(v_i) \)（网络 G）
2. 提取几何特征（曲率、法线等）或复用预计算的最近高模点特征
3. 构造条件 \( \mathbf{y}_i \)，运行扩散模型（确定性模式 \( z=0 \)，20 步 DDIM）得到颜色 \( \hat{c}_i \)
4. 低模三角形内部通过重心插值颜色（顶点颜色 → 片段着色）
5. （可选）若要求纹理贴图而非顶点颜色，可将顶点 UV 设置为临时连续参数化，再将颜色烘焙成纹理。

---

## 6. 起步策略与代码复用

### 6.1 推荐的开源项目

| 组件 | 项目 | 用途 |
|------|------|------|
| SDF + 颜色场 | [torch-ngp (SDF module)](https://github.com/ashawkey/torch-ngp) | 快速搭建网络 G，含跳跃连接和位置编码 |
| 扩散模型 | [diffusers](https://github.com/huggingface/diffusers) | 条件扩散训练框架，支持 FiLM |
| 3D 几何处理 | [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | 法线、曲率、网格采样 |
| 特征提取 | [Open3D](https://github.com/isl-org/Open3D) | 测地距离、点云操作 |
| 实验管理 | [Weights & Biases](https://wandb.ai/) + [MLflow](https://mlflow.org/) | 监控、记录、可视化 |

### 6.2 环境搭建（conda）

```bash
conda create -n uv_mapping python=3.10
conda activate uv_mapping

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt201/download.html
pip install open3d trimesh diffusers accelerate wandb mlflow tensorboard
pip install scikit-learn scipy matplotlib imageio nvdiffrast

git clone https://github.com/ashawkey/torch-ngp   # 提取 SDF 模块代码
git clone https://github.com/cvmi-lab/Point-UV-Diffusion.git  # 参考扩散设计
```

### 6.3 实现路线图（建议按顺序）

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 第 1 周 | 环境配置，数据采样管线，实现网络 G 并单独训练（SDF+低频颜色） | 5 天 |
| 第 2 周 | 实现几何特征提取（曲率、法线、边界距离），缓存为 .npz | 3 天 |
| 第 3 周 | 基于 diffusers 实现条件扩散模型 D，阶段 2 训练 | 5 天 |
| 第 4 周 | 实现网络 R，阶段 3 联合训练，调参 | 5 天 |
| 第 5 周 | 评估、可视化、消融实验，撰写报告 | 5 天 |

### 6.4 数据增强与调试技巧

- **坐标抖动**（高斯噪声 \( \sigma=0.001 \)）提升鲁棒性
- 使用**梯度裁剪**（norm=1.0）避免梯度爆炸
- 阶段 3 初期只开 \( \mathcal{L}_{\text{Reverse}} \) 不加熵正则化，待稳定后再加入
- 定期（每 10 epoch）在验证集低模上渲染图像，上传到 W&B 人工检查

---

## 7. 评估指标

| 指标 | 计算方法 | 理想方向 |
|------|----------|----------|
| PSNR | 渲染图像与高模参考图像 | ↑ |
| SSIM | 结构相似性 | ↑ |
| RMSE | 颜色误差均方根 | ↓ |
| 几何标签准确率 | R 网络预测 \( \ell \) 的准确率 | ↑ |
| 模式覆盖率（熵） | 对多次采样颜色的分布熵 | 适中不坍塌 |

**可视化工具**：
- 使用 Blender 的 `bpy` 模块批量渲染低模多视角图像
- 生成颜色误差热图（顶点差异映射到网格）

---

## 8. 潜在风险与应对

| 风险 | 缓解措施 |
|------|----------|
| 几何偏移过大（结构性简化） | 本方案仅保证微小偏移，若低模有大量结构丢失，需回归手动重拓扑 |
| 扩散模型训练不稳定 | 使用 EMA（指数移动平均）更新参数，减小学习率 |
| 闭环约束导致模式崩塌 | 从较小的 \( \lambda_{\text{Reverse}} \) 开始，监控 \( \mathcal{L}_{\text{Reverse}} \) 和熵值 |
| 训练时间过长 | 使用混合精度训练，缓存所有几何特征离线 |

---

## 9. 后续扩展方向

- 将扩散模型替换为 **一致性模型**（Consistency Model），实现单步生成，提升推理速度。
- 引入 **跨模型微调**：在同一几何类别上预训练 G 的 SDF 分支，再针对不同纹理微调 D。
- 支持 **纹理编辑**：允许用户修改低模颜色后，通过逆映射网络 \( R \) 反向微调高模纹理 \( T \)。

---

## 10. 总结

本方案从第一性原理出发，彻底放弃传统 UV 映射，采用**隐式纹理场 + 条件扩散 + 闭环约束**的三网络协同架构，在数学上避开了病态问题。通过分阶段渐进训练和几何上下文注入，能够同时应对**几何偏移**和**高频纹理保持**两大挑战。起步策略充分复用现有开源代码，可在一个月内完成原型实现。
