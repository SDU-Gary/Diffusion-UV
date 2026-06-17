"""
SDF网络诊断脚本 - 分析训练失败的根本原因
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append('/home/kyrie/Diffusion-UV')

from src.models.sdf_network import SDFNetwork
from src.data.gpu_constant_baker import load_mesh_constants
from src.data.sdf_data_generator import SDFDataGenerator

print("=" * 80)
print("SDF网络诊断分析")
print("=" * 80)

# 加载模型
device = "cuda"
sdf_net = SDFNetwork(
    num_levels=8,
    log2_hashmap_size=12,
    base_res=8,
    max_res=128,
    hidden_dim=32,
    num_layers=2,
    cuda_backend="torch",
).to(device)

checkpoint = torch.load("outputs/sdf_pretrain_100epochs_fixed/best.pt", map_location=device)
sdf_net.load_state_dict(checkpoint['model_state_dict'])
sdf_net.eval()

print(f"✓ SDF网络加载成功（epoch {checkpoint['epoch']}）")

# 加载mesh数据
constants, metadata = load_mesh_constants("data/models/bunny_mesh_constants.pt", map_location=device)
bbox_min = metadata["bbox_min"]
bbox_max = metadata["bbox_max"]

print("\n" + "=" * 80)
print("诊断1：检查SDF值分布")
print("=" * 80)

# 在bounding box中采样
num_samples = 10000
samples = torch.rand(num_samples, 3, device=device)
bbox_min_t = torch.tensor(bbox_min, device=device)
bbox_max_t = torch.tensor(bbox_max, device=device)
samples = bbox_min_t + samples * (bbox_max_t - bbox_min_t)

with torch.no_grad():
    sdf_vals = sdf_net(samples).cpu().numpy()

print(f"SDF值统计（{num_samples}个随机点）:")
print(f"  Mean: {sdf_vals.mean():.6f}")
print(f"  Std:  {sdf_vals.std():.6f}")
print(f"  Min:  {sdf_vals.min():.6f}")
print(f"  Max:  {sdf_vals.max():.6f}")

# 理想情况下，SDF值应该在bounding box范围内（约0.25）
if abs(sdf_vals.mean()).item() < 0.1:
    print("  ✓ SDF均值接近0（合理）")
else:
    print(f"  ❌ SDF均值偏离0: {sdf_vals.mean():.6f}")

print("\n" + "=" * 80)
print("诊断2：检查梯度场")
print("=" * 80)

# 测试几个关键点
test_points = torch.tensor([
    [0.5, 0.5, 0.5],  # 中心
    [0.3, 0.5, 0.5],  # 偏离中心
    [0.5, 0.5, 0.25], # 表面附近
    [0.1, 0.1, 0.1],  # 角落
], device=device)

test_points_req = test_points.clone().detach()
test_points_req.requires_grad_(True)

sdf_vals_test = sdf_net(test_points_req)
grads_test = torch.autograd.grad(
    outputs=sdf_vals_test.sum(),
    inputs=test_points_req,
    create_graph=False,
)[0]

grad_norms = torch.norm(grads_test, dim=-1)

print("测试点的梯度分析:")
for i, (pt, grad, grad_norm) in enumerate(zip(test_points, grads_test, grad_norms)):
    print(f"  点{i+1} {pt.cpu().numpy()}:")
    print(f"    SDF: {sdf_vals_test[i].item():.6f}")
    print(f"    梯度: {grad.cpu().numpy()}")
    print(f"    梯度模长: {grad_norm.item():.6f}", end="")
    if abs(grad_norm.item() - 1.0) < 0.1:
        print(" ✓")
    else:
        print(" ❌ (应该接近1.0)")

print("\n" + "=" * 80)
print("诊断3：与mesh法线对比")
print("=" * 80)

# 在表面采样少量点
num_surface_samples = 1000

from src.data.sdf_data_generator import SDFDataGenerator
data_gen = SDFDataGenerator(
    mesh_constants_path="data/models/bunny_mesh_constants.pt",
    surface_batch_size=num_surface_samples,
    off_surface_batch_size=num_surface_samples,
    device=device,
)

batch = data_gen.next_batch()
surface_pos = batch["surface_pos"]

# 获取mesh法线
face_idx = torch.multinomial(
    constants["face_probs"], num_surface_samples, replacement=True
)
sel_normals = constants["face_normals"][face_idx]

# Barycentric插值
u = torch.rand(num_surface_samples, device=device)
v = torch.rand(num_surface_samples, device=device)
is_over = (u + v) > 1.0
u = torch.where(is_over, 1.0 - u, u)
v = torch.where(is_over, 1.0 - v, v)
w = 1.0 - u - v
bary = torch.stack([u, v, w], dim=-1)

mesh_normals = torch.bmm(bary.unsqueeze(1), sel_normals).squeeze(1)
mesh_normals = torch.nn.functional.normalize(mesh_normals, dim=-1, eps=1e-6)

# 获取SDF法线
surface_pos_req = surface_pos.clone().detach()
surface_pos_req.requires_grad_(True)

sdf_surf = sdf_net(surface_pos_req)
sdf_normals = torch.autograd.grad(
    outputs=sdf_surf.sum(),
    inputs=surface_pos_req,
    create_graph=False,
)[0]

sdf_normals = torch.nn.functional.normalize(sdf_normals, dim=-1, eps=1e-6)

# 计算余弦相似度
cosine_sim = torch.sum(sdf_normals * mesh_normals, dim=-1).cpu().numpy()

print(f"表面点法线对比（{num_surface_samples}个样本）:")
print(f"  余弦相似度: {cosine_sim.mean():.6f} ± {cosine_sim.std():.6f}")
print(f"  最大值: {cosine_sim.max():.6f}")
print(f"  最小值: {cosine_sim.min():.6f}")

if cosine_sim.mean() > 0.9:
    print("  ✓ 法线方向基本一致")
elif cosine_sim.mean() > 0.5:
    print("  ⚠️ 法线方向部分一致")
else:
    print("  ❌ 法线方向几乎正交（严重问题）")

print("\n" + "=" * 80)
print("诊断4：检查是否过拟合")
print("=" * 80)

# 在训练集和测试集上分别评估
print("（暂略，需要单独的test set）")

print("\n" + "=" * 80)
print("根本原因推断")
print("=" * 80)

print("\n基于诊断结果，可能的问题：")
print("\n1. ❌ SDF值分布异常")
print("   如果SDF值全部接近某个常数，说明网络坍塌")

print("\n2. ❌ 梯度场异常")
print("   如果梯度模长普遍<0.1或>2.0，说明网络没有学习Eikonal约束")

print("\n3. ❌ 法线方向错误")
print("   如果余弦相似度<0.1，说明SDF法线和mesh法线几乎正交")

print("\n4. ❌ 可能的根本原因：")
print("   a) 网络容量不足（只有67K参数）")
print("   b) B-Spline Hash Grid初始化问题")
print("   c) Surface样本和Off-surface样本不平衡")
print("   d) Eikonal loss权重不够高")
print("   e) 训练epoch数不够（可能需要100+ epochs）")

print("\n5. 🔧 建议的解决方案：")
print("\n   方案A: 大幅增加lambda_eikonal权重（0.1 → 1.0或更高）")
print("   方案B: 增加网络容量（hidden_dim: 32 → 128）")
print("   方案C: 延长训练时间（50 → 200 epochs）")
print("   方案D: 检查B-Spline Hash Grid是否有实现问题")
print("   方案E: 使用更简单的Fourier编码代替Hash Grid")

print("\n" + "=" * 80)
