"""
实验一（重做）：切空间正交性断言测试 - 正确的投影矩阵实现
"""

import torch
import sys
sys.path.append('/home/kyrie/Diffusion-UV')

from src.training.metric_aligned_iuv_losses import project_to_tangent_space

def test_orthogonality_correct():
    """测试正确的投影矩阵实现是否保证正交性"""

    print("=" * 80)
    print("实验一（重做）：切空间正交性断言测试")
    print("使用正确的投影矩阵实现: P = I - n·n^T")
    print("=" * 80)

    # 创建测试数据
    batch_size = 100
    jacobian = torch.randn(batch_size, 2, 3)  # [B, 2, 3]
    normals = torch.randn(batch_size, 3)
    normals = torch.nn.functional.normalize(normals, dim=-1)

    print(f"\n输入形状:")
    print(f"  Jacobian: {jacobian.shape}")
    print(f"  Normals: {normals.shape}")

    # 使用正确的投影矩阵实现
    print(f"\n执行 tangent space projection (投影矩阵法)...")
    j_tangent = project_to_tangent_space(jacobian, normals)
    print(f"输出形状: {j_tangent.shape}")  # 应该是 [B, 2, 3]

    # 计算原始Jacobian与法线的内积
    noise_u_original = torch.sum(jacobian[:, 0, :] * normals, dim=-1)
    noise_v_original = torch.sum(jacobian[:, 1, :] * normals, dim=-1)

    print(f"\n原始 Jacobian 与法线的内积（投影前）:")
    print(f"  noise_u (J[0]·n): mean={noise_u_original.mean().item():.6f}, std={noise_u_original.std().item():.6f}")
    print(f"  noise_v (J[1]·n): mean={noise_v_original.mean().item():.6f}, std={noise_v_original.std().item():.6f}")
    print(f"  绝对值均值: |noise_u|={noise_u_original.abs().mean().item():.6f}, |noise_v|={noise_v_original.abs().mean().item():.6f}")

    # 检查投影后的正交性
    noise_u_projected = torch.sum(j_tangent[:, 0, :] * normals, dim=-1)
    noise_v_projected = torch.sum(j_tangent[:, 1, :] * normals, dim=-1)

    print(f"\n投影后 Jacobian 与法线的内积（应该接近0）:")
    print(f"  noise_u (J_tangent[0]·n): mean={noise_u_projected.mean().item():.6f}, std={noise_u_projected.std().item():.6f}")
    print(f"  noise_v (J_tangent[1]·n): mean={noise_v_projected.mean().item():.6f}, std={noise_v_projected.std().item():.6f}")
    print(f"  绝对值均值: |noise_u|={noise_u_projected.abs().mean().item():.6f}, |noise_v|={noise_v_projected.abs().mean().item():.6f}")
    print(f"  最大值: max|noise_u|={noise_u_projected.abs().max().item():.6f}, max|noise_v|={noise_v_projected.abs().max().item():.6f}")

    # 硬性断言
    print(f"\n{'='*80}")
    print(f"正交性断言测试结果:")

    tolerance = 1e-5

    if noise_u_projected.abs().max().item() < tolerance and noise_v_projected.abs().max().item() < tolerance:
        print(f"  ✅ PASS - 投影后的Jacobian与法线正交")
        print(f"       max |J·n| < {tolerance}")
        return True
    else:
        print(f"  ❌ FAIL - 投影后的Jacobian仍与法线有非零内积！")
        print(f"       max |noise_u| = {noise_u_projected.abs().max().item():.6f}")
        print(f"       max |noise_v| = {noise_v_projected.abs().max().item():.6f}")
        return False

if __name__ == "__main__":
    success = test_orthogonality_correct()
    sys.exit(0 if success else 1)
