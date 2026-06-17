"""
实验一：切空间正交性断言测试

目的：验证 tangent space projection 是否真的把法向分量剔除了
"""

import torch
import sys
sys.path.append('/home/kyrie/Diffusion-UV')

from src.training.metric_aligned_iuv_losses import project_to_tangent_space
from src.models.metric_aligned_iuv_field import MetricAlignedIUVField

def test_orthogonality():
    """测试投影后的Jacobian是否与法线正交"""

    print("=" * 80)
    print("实验一：切空间正交性断言测试")
    print("=" * 80)

    # 创建测试数据
    batch_size = 100

    # 模拟一个简单的Jacobian
    jacobian = torch.randn(batch_size, 2, 3)  # [B, 2, 3]

    # 模拟单位法线
    normals = torch.randn(batch_size, 3)
    normals = torch.nn.functional.normalize(normals, dim=-1)

    print(f"\n输入形状:")
    print(f"  Jacobian: {jacobian.shape}")
    print(f"  Normals: {normals.shape}")

    # 使用当前的投影实现
    print(f"\n执行 tangent space projection...")
    j_tangent = project_to_tangent_space(jacobian, normals)

    print(f"输出形状: {j_tangent.shape}")

    # 计算原始Jacobian与法线的内积（应该非零）
    noise_u_original = torch.sum(jacobian[:, 0, :] * normals, dim=-1)
    noise_v_original = torch.sum(jacobian[:, 1, :] * normals, dim=-1)

    print(f"\n原始 Jacobian 与法线的内积（投影前）:")
    print(f"  noise_u (J[0]·n): mean={noise_u_original.mean().item():.6f}, std={noise_u_original.std().item():.6f}")
    print(f"  noise_v (J[1]·n): mean={noise_v_original.mean().item():.6f}, std={noise_v_original.std().item():.6f}")
    print(f"  绝对值均值: |noise_u|={noise_u_original.abs().mean().item():.6f}, |noise_v|={noise_v_original.abs().mean().item():.6f}")

    # 检查投影后的正交性
    # j_tangent是 [B, 2, 2]，需要重新投影回3D空间来检查正交性
    # 实际上，投影后的Jacobian应该满足 J_tangent @ n ≈ 0

    # 重建3D切空间Jacobian（用tangent和bitangent基）
    primary_tangent = jacobian[:, 0, :]
    normal_component = torch.sum(primary_tangent * normals, dim=-1, keepdim=True) * normals
    tangent = primary_tangent - normal_component
    tangent = torch.nn.functional.normalize(tangent, dim=-1, eps=1e-6)

    bitangent = torch.cross(normals, tangent, dim=-1)
    bitangent = torch.nn.functional.normalize(bitangent, dim=-1, eps=1e-6)

    # 将[B, 2, 2]的j_tangent重新投影到3D空间
    j_tangent_3d_u = j_tangent[:, 0, 0].unsqueeze(-1) * tangent + j_tangent[:, 0, 1].unsqueeze(-1) * bitangent
    j_tangent_3d_v = j_tangent[:, 1, 0].unsqueeze(-1) * tangent + j_tangent[:, 1, 1].unsqueeze(-1) * bitangent

    # 计算投影后的法向噪声
    noise_u_projected = torch.sum(j_tangent_3d_u * normals, dim=-1)
    noise_v_projected = torch.sum(j_tangent_3d_v * normals, dim=-1)

    print(f"\n投影后 Jacobian 与法线的内积（应该接近0）:")
    print(f"  noise_u (J_tangent_u·n): mean={noise_u_projected.mean().item():.6f}, std={noise_u_projected.std().item():.6f}")
    print(f"  noise_v (J_tangent_v·n): mean={noise_v_projected.mean().item():.6f}, std={noise_v_projected.std().item():.6f}")
    print(f"  绝对值均值: |noise_u|={noise_u_projected.abs().mean().item():.6f}, |noise_v|={noise_v_projected.abs().mean().item():.6f}")

    # 硬性断言
    print(f"\n{'='*80}")
    print(f"正交性断言测试结果:")

    tolerance = 1e-4

    if noise_u_projected.abs().max().item() < tolerance and noise_v_projected.abs().max().item() < tolerance:
        print(f"  ✅ PASS - 投影后的Jacobian与法线正交（max |J·n| < {tolerance}）")
        return True
    else:
        print(f"  ❌ FAIL - 投影后的Jacobian仍与法线有非零内积！")
        print(f"       max |noise_u| = {noise_u_projected.abs().max().item():.6f}")
        print(f"       max |noise_v| = {noise_v_projected.abs().max().item():.6f}")
        print(f"       这证明当前的tangent space projection实现有数学错误！")
        return False

if __name__ == "__main__":
    success = test_orthogonality()
    sys.exit(0 if success else 1)
