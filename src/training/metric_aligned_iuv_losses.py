"""
MA-IUVF 训练Loss

核心组件：
1. Metric alignment loss（雅可比匹配）
2. UV anchor loss
3. Chart classification loss
"""

import torch
import torch.nn.functional as F
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def gather_chart_uvs(uv_preds: torch.Tensor, chart_id: torch.Tensor) -> torch.Tensor:
    """
    根据chart ID收集对应的UV预测

    Args:
        uv_preds: [B, C, 2] 所有chart的UV预测
        chart_id: [B] 目标chart ID

    Returns:
        selected_uv: [B, 2] 选中chart的UV
    """
    B = uv_preds.shape[0]
    range_indices = torch.arange(B, device=uv_preds.device)

    # 高级索引：[B, C, 2] -> [B, 2]
    selected_uv = uv_preds[range_indices, chart_id]

    return selected_uv


def compute_uv_jacobian(selected_uv: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    计算UV相对于位置的空间雅可比

    使用autograd计算：
    J = [d(u)/d(x), d(u)/d(y), d(u)/d(z)]
        [d(v)/d(x), d(v)/d(y), d(v)/d(z)]

    Args:
        selected_uv: [B, 2] UV坐标（requires_grad从pos传递）
        pos: [B, 3] 3D位置（requires_grad=True）

    Returns:
        jacobian: [B, 2, 3] UV雅可比矩阵
    """
    # 确保pos需要梯度
    if not pos.requires_grad:
        raise ValueError("pos必须设置requires_grad=True")

    # 计算u通道的梯度
    grad_u = torch.autograd.grad(
        outputs=selected_uv[:, 0].sum(),
        inputs=pos,
        create_graph=True,
        retain_graph=True,
    )[0]  # [B, 3]

    # 计算v通道的梯度
    grad_v = torch.autograd.grad(
        outputs=selected_uv[:, 1].sum(),
        inputs=pos,
        create_graph=True,
        retain_graph=True,
    )[0]  # [B, 3]

    # 组装雅可比矩阵
    jacobian = torch.stack([grad_u, grad_v], dim=1)  # [B, 2, 3]

    return jacobian


def compute_metric_loss(
    j_pred: torch.Tensor,
    j_gt: torch.Tensor,
) -> torch.Tensor:
    """
    计算metric alignment loss

    L = ||J_pred - J_gt||_F^2

    Args:
        j_pred: [B, 2, 3] 预测雅可比
        j_gt: [B, 2, 3] GT雅可比

    Returns:
        loss: [1] metric loss
    """
    diff = j_pred - j_gt
    loss = torch.mean(diff ** 2)

    return loss


def project_to_tangent_space(
    jacobian: torch.Tensor,  # [B, 2, 3]
    normals: torch.Tensor,    # [B, 3]
) -> torch.Tensor:
    """
    Project full Jacobian to tangent plane using projection matrix

    Mathematical formulation:
    - Projection matrix: P = I - n·n^T
    - J_tangent = J · P

    This directly projects the entire Jacobian to the tangent plane,
    without using Jacobian entries to construct the tangent basis.

    Args:
        jacobian: [B, 2, 3] UV Jacobian in full 3D space
        normals: [B, 3] Unit normal vectors at sample points

    Returns:
        jacobian_tangent: [B, 2, 3] Projected Jacobian in tangent plane
    """
    B = jacobian.shape[0]

    # Build projection matrix: P = I - n·n^T
    # normals: [B, 3] -> [B, 3, 1]
    # normals: [B, 3] -> [B, 1, 3]
    # n·n^T: [B, 3, 3]
    normal_outer = torch.bmm(
        normals.unsqueeze(2),      # [B, 3, 1]
        normals.unsqueeze(1)       # [B, 1, 3]
    )  # [B, 3, 3]

    # P = I - n·n^T: [B, 3, 3]
    identity = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0).expand(B, -1, -1)
    P = identity - normal_outer  # [B, 3, 3]

    # Project Jacobian: J_tangent = J · P
    # [B, 2, 3] @ [B, 3, 3] = [B, 2, 3]
    jacobian_tangent = torch.bmm(jacobian, P)

    return jacobian_tangent


def compute_tangent_space_metric_loss(
    j_pred: torch.Tensor,      # [B, 2, 3]
    normals: torch.Tensor,      # [B, 3]
    j_gt: torch.Tensor,         # [B, 2, 3]
) -> torch.Tensor:
    """
    Compute metric alignment loss in tangent space only

    Key difference from compute_metric_loss:
    - Projects J_pred to tangent plane before computing loss
    - Ensures network doesn't learn gradients in normal direction
    - j_gt is already in tangent space (from mesh UVs)

    Args:
        j_pred: [B, 2, 3] Predicted full Jacobian
        normals: [B, 3] Unit normal vectors (from mesh or SDF)
        j_gt: [B, 2, 3] Ground truth Jacobian (already in tangent space)

    Returns:
        loss: Scalar tangent space metric loss
    """
    # Project both Jacobians to tangent space
    j_pred_tangent = project_to_tangent_space(j_pred, normals)  # [B, 2, 3]
    j_gt_tangent = project_to_tangent_space(j_gt, normals)        # [B, 2, 3]

    # Compute loss in tangent space
    diff = j_pred_tangent - j_gt_tangent  # [B, 2, 3]
    loss = torch.mean(diff ** 2)

    return loss


def compute_anchor_loss(
    uv_pred: torch.Tensor,
    uv_anchor: torch.Tensor,
) -> torch.Tensor:
    """
    计算UV anchor loss

    L = ||uv_pred - uv_anchor||^2

    Args:
        uv_pred: [B, 2] 预测UV
        uv_anchor: [B, 2] 锚点UV

    Returns:
        loss: [1] anchor loss
    """
    loss = F.mse_loss(uv_pred, uv_anchor)
    return loss


def compute_chart_com_loss(
    uv_pred: torch.Tensor,
    uv_anchor: torch.Tensor,
    chart_id: torch.Tensor,
    num_charts: int,
) -> torch.Tensor:
    """
    计算 chart-wise Center of Mass anchor loss。

    对每个当前 batch 中出现的 chart k，分别计算预测 UV 与 GT UV 的
    几何中心，并对齐二者：

        L_com(k) = ||mean(UV_pred_k) - mean(UV_gt_k)||^2

    最终 loss 是所有出现 chart 的平均值。该项只约束每个 chart 的
    宏观平移常数，不做逐点拉扯。

    Args:
        uv_pred: [B, 2] 目标 chart 分支的预测 UV
        uv_anchor: [B, 2] GT UV
        chart_id: [B] 每个样本所属 chart
        num_charts: chart 总数 C

    Returns:
        loss: [1] 质心对齐 loss
    """
    if uv_pred.shape != uv_anchor.shape:
        raise ValueError(f"uv_pred/uv_anchor shape mismatch: {uv_pred.shape} vs {uv_anchor.shape}")
    if uv_pred.ndim != 2 or uv_pred.shape[-1] != 2:
        raise ValueError(f"uv_pred must have shape [B, 2], got {uv_pred.shape}")
    if chart_id.ndim != 1 or chart_id.shape[0] != uv_pred.shape[0]:
        raise ValueError(f"chart_id must have shape [B], got {chart_id.shape}")
    if num_charts < 1:
        raise ValueError("num_charts must be >= 1")

    chart_id = chart_id.long()
    sums_pred = uv_pred.new_zeros((num_charts, 2))
    sums_gt = uv_anchor.new_zeros((num_charts, 2))
    counts = uv_pred.new_zeros((num_charts, 1))

    sums_pred.index_add_(0, chart_id, uv_pred)
    sums_gt.index_add_(0, chart_id, uv_anchor)
    counts.index_add_(0, chart_id, torch.ones((uv_pred.shape[0], 1), device=uv_pred.device, dtype=uv_pred.dtype))

    present = counts.squeeze(-1) > 0
    if not bool(present.any()):
        return uv_pred.sum() * 0.0

    means_pred = sums_pred / counts.clamp_min(1.0)
    means_gt = sums_gt / counts.clamp_min(1.0)
    delta = means_pred[present] - means_gt[present]  # [K_present, 2]
    return (delta ** 2).sum(dim=-1).mean()


def compute_classification_loss(
    logits: torch.Tensor,
    chart_id: torch.Tensor,
) -> torch.Tensor:
    """
    计算chart分类loss

    Args:
        logits: [B, C] 分类logits
        chart_id: [B] 目标chart ID

    Returns:
        loss: [1] 分类loss
    """
    loss = F.cross_entropy(logits, chart_id)
    return loss


def compute_unified_local_loss(
    selected_uv: torch.Tensor,      # [B, 2] 当前点的UV预测
    uv_anchor: torch.Tensor,         # [B, 2] 当前点的GT UV（绝对定位锚点）
    pos: torch.Tensor,                 # [B, 3] 当前点的3D位置
    pos_neighbors: torch.Tensor,       # [B, N, 3] 邻域点的3D位置
    j_gt: torch.Tensor,                # [B, 2, 3] GT雅可比矩阵
    model_forward_fn,                  # 模型前向传播函数
    chart_id: torch.Tensor,            # [B] chart ID
) -> torch.Tensor:
    """
    计算局部坐标差分损失（Unified Local Loss）

    核心思想：不直接约束一阶导数（具有平移不变性），而是约束局部邻域内的
    UV预测应该与基于 GT UV 的一阶泰勒展开一致。

    数学公式：
    L_unified = ||û(x + Δx) - (u_gt(x) + J_gt · Δx)||²

    其中：
    - û(x + Δx): 邻域点的UV预测（网络输出）
    - u_gt(x): 当前点的GT UV（绝对定位锚点，消除平移不变性！）
    - J_gt: GT雅可比矩阵
    - Δx: 邻域偏移向量

    CRITICAL: 使用 u_gt(x) 而非 û(x) 作为泰勒展开基点，彻底消除平移不变性！
    如果网络产生全局平移漂移 C，则：
    - û(x + Δx) = u_gt(x + Δx) + C
    - u_taylor = u_gt(x) + J_gt · Δx ≈ u_gt(x + Δx)
    - diff = C ≠ 0 ✅ (能够检测到平移漂移)

    Args:
        selected_uv: [B, 2] 当前点的UV预测（未使用，保留用于兼容）
        uv_anchor: [B, 2] 当前点的GT UV（绝对定位锚点）
        pos: [B, 3] 当前点的3D位置
        pos_neighbors: [B, N, 3] 邻域点的3D位置（N个邻居）
        j_gt: [B, 2, 3] GT雅可比矩阵
        model_forward_fn: 模型前向传播函数（用于获取邻域点UV）
        chart_id: [B] 每个样本的chart ID

    Returns:
        loss: [1] 局部坐标差分loss
    """
    B, N, _ = pos_neighbors.shape

    # 1. 获取邻域点的UV预测
    # Flatten pos_neighbors for batch forward pass
    pos_neighbors_flat = pos_neighbors.view(-1, 3)  # [B*N, 3]

    # Forward pass for neighbors
    # Note: This requires a model that can batch process positions
    uv_neighbors_all = model_forward_fn(pos_neighbors_flat)  # [B*N, C, 2]

    # Gather UVs based on chart_id (repeat chart_id for N neighbors)
    chart_id_expanded = chart_id.unsqueeze(1).repeat(1, N).view(-1)  # [B*N]
    range_indices = torch.arange(B * N, device=pos_neighbors_flat.device)

    # Select UV for each neighbor
    uv_neighbors = uv_neighbors_all[range_indices, chart_id_expanded]  # [B*N, 2]
    uv_neighbors = uv_neighbors.view(B, N, 2)  # [B, N, 2]

    # 2. 计算一阶泰勒展开预测（使用 GT UV 作为基点，消除平移不变性！）
    # u_pred_taylor = u_gt(x) + J_gt · Δx
    # Δx = pos_neighbors - pos (需要广播)
    pos_expanded = pos.unsqueeze(1)  # [B, 1, 3]
    delta_x = pos_neighbors - pos_expanded  # [B, N, 3]

    # J_gt · Δx: [B, 2, 3] @ [B, N, 3] -> [B, 2, N]
    # 使用 einsum进行批量矩阵乘法
    jacobian_delta = torch.einsum('bij,bnj->bin', j_gt, delta_x)  # [B, 2, N]

    # u_taylor = u_gt(x) + J_gt · Δx （CRITICAL: 使用 uv_anchor 而非 selected_uv！）
    # uv_anchor: [B, 2] -> [B, 2, 1]
    # jacobian_delta: [B, 2, N] -> [B, 2, N]
    uv_anchor_expanded = uv_anchor.unsqueeze(2)  # [B, 2, 1]
    uv_taylor = uv_anchor_expanded + jacobian_delta  # [B, 2, N]

    # 3. 计算预测差异（修正维度对齐！）
    # L = ||uv_neighbors - uv_taylor||²
    # uv_neighbors: [B, N, 2]
    # uv_taylor: [B, 2, N] -> [B, N, 2] (permute(0, 2, 1) NOT permute(2, 0, 1))
    diff = uv_neighbors - uv_taylor.permute(0, 2, 1)  # [B, N, 2]
    loss = torch.mean(diff ** 2)

    return loss


def compute_metric_aligned_iuv_loss(
    model_output,
    pos: torch.Tensor,
    j_3d_gt: torch.Tensor,
    uv_anchor: torch.Tensor,
    chart_id: torch.Tensor,
    normals: torch.Tensor = None,  # NEW: optional normals
    metric_weight: float = 1.0,
    anchor_weight: float = 1e-4,
    com_weight: float = 0.0,
    cls_weight: float = 1.0,
    # NEW: Unified Local Loss Parameters
    unified_weight: float = 0.0,
    unified_num_neighbors: int = 4,
    unified_epsilon: float = 0.01,
    # ========================================
    # ARCHIVED: Tangent Space Parameter
    # use_tangent_space: bool = False,  # Flag for tangent space projection
    # ========================================
) -> Dict[str, torch.Tensor]:
    """
    计算完整的MA-IUVF loss

    Args:
        model_output: MetricAlignedIUVOutput
        pos: [B, 3] 3D位置（requires_grad=True）
        j_3d_gt: [B, 2, 3] GT雅可比
        uv_anchor: [B, 2] 锚点UV
        chart_id: [B] 目标chart ID
        normals: [B, 3] Normal vectors (required if use_tangent_space=True)
        metric_weight: metric loss权重
        anchor_weight: anchor loss权重
        com_weight: chart-wise 质心对齐 loss 权重
        cls_weight: 分类loss权重
        unified_weight: unified local loss权重
        unified_num_neighbors: 邻域点数量
        unified_epsilon: 邻域扰动大小

    Returns:
        loss_dict: {
            "total": 总loss,
            "metric": metric loss,
            "anchor": anchor loss,
            "anchor_weighted": 加权anchor loss,
            "com": chart-wise 质心对齐 loss,
            "com_weighted": 加权质心 loss,
            "cls": 分类loss,
            "unified": unified local loss,
        }
    """
    # 1. 收集目标chart的UV
    selected_uv = gather_chart_uvs(model_output.uv_preds, chart_id)  # [B, 2]

    # 2. 计算空间雅可比
    j_pred = compute_uv_jacobian(selected_uv, pos)  # [B, 2, 3]

    # 3. 计算metric loss - use original full space approach
    metric_loss = compute_metric_loss(j_pred, j_3d_gt)

    # 4. 计算其他loss
    anchor_loss = compute_anchor_loss(selected_uv, uv_anchor)
    com_loss = compute_chart_com_loss(
        selected_uv,
        uv_anchor,
        chart_id,
        num_charts=model_output.uv_preds.shape[1],
    )
    cls_loss = compute_classification_loss(model_output.logits, chart_id)

    # 5. NEW: 计算Unified Local Loss
    unified_loss = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
    if unified_weight > 0:
        # 生成邻域点：通过添加小的随机扰动
        B = pos.shape[0]
        N = unified_num_neighbors

        # 生成随机扰动（在单位球面上）
        noise = torch.randn(B, N, 3, device=pos.device, dtype=pos.dtype)
        noise = noise / (torch.norm(noise, dim=-1, keepdim=True) + 1e-8)  # 归一化
        pos_neighbors = pos.unsqueeze(1) + noise * unified_epsilon  # [B, N, 3]

        # 定义模型前向传播函数（用于获取邻域点UV）
        def model_forward(positions):
            # 临时禁用梯度位置要求，用于邻域点前向传播
            temp_pos = positions.clone().detach().requires_grad_(False)
            temp_output = model_output.model(temp_pos)
            return temp_output.uv_preds

        try:
            unified_loss = compute_unified_local_loss(
                selected_uv=selected_uv,
                uv_anchor=uv_anchor,  # CRITICAL: 传入 GT UV 作为泰勒基点
                pos=pos,
                pos_neighbors=pos_neighbors,
                j_gt=j_3d_gt,
                model_forward_fn=model_forward,
                chart_id=chart_id,
            )
        except Exception as e:
            # 如果unified loss计算失败，使用0
            logger.warning(f"Unified local loss计算失败，使用0: {e}")
            unified_loss = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

    # 6. 组合
    total_loss = (
        metric_weight * metric_loss +
        anchor_weight * anchor_loss +
        com_weight * com_loss +
        cls_weight * cls_loss +
        unified_weight * unified_loss
    )

    loss_dict = {
        "total": total_loss,
        "metric": metric_loss,
        "anchor": anchor_loss,
        "anchor_weighted": anchor_weight * anchor_loss,
        "com": com_loss,
        "com_weighted": com_weight * com_loss,
        "cls": cls_loss,
        "unified": unified_loss,
    }

    return loss_dict


def validate_jacobian_math():
    """
    验证雅可比计算的数学正确性

    测试用例：
    - 平面三角形：v0=(0,0,0), v1=(1,0,0), v2=(0,1,0)
    - UV：uv0=(0,0), uv1=(2,0), uv2=(0,3)
    - 预期J_3d = [[2,0,0], [0,3,0]]
    """
    logger.info("验证雅可比数学...")

    # 构造测试三角形
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)

    uvs = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 3.0],
    ], dtype=torch.float32)

    # 计算雅可比
    from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker

    baker = MetricAlignedIUVBaker.__new__(MetricAlignedIUVBaker)
    j_3d_gt = baker._compute_triangle_jacobian(vertices, uvs)

    # 预期结果
    expected_j = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ], dtype=torch.float32)

    # 验证
    diff = (j_3d_gt - expected_j).abs()
    max_diff = diff.max().item()

    if max_diff < 1e-5:
        logger.info(f"✓ 雅可比数学正确 (max_diff={max_diff:.2e})")
        return True
    else:
        logger.error(f"✗ 雅可比数学错误 (max_diff={max_diff:.2e})")
        logger.error(f"预期:\n{expected_j}")
        logger.error(f"实际:\n{j_3d_gt}")
        return False


def validate_normal_zero_grad():
    """
    验证法向零梯度特性

    对于平面三角形，J_3d @ normal ≈ [0, 0]
    """
    logger.info("验证法向零梯度...")

    # 平面三角形
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)

    uvs = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 3.0],
    ], dtype=torch.float32)

    # 计算雅可比
    from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker

    baker = MetricAlignedIUVBaker.__new__(MetricAlignedIUVBaker)
    j_3d_gt = baker._compute_triangle_jacobian(vertices, uvs)

    # 法向
    normal = torch.tensor([0.0, 0.0, 1.0])

    # J @ n
    result = j_3d_gt @ normal

    # 应该接近[0, 0]
    max_abs = result.abs().max().item()

    if max_abs < 1e-5:
        logger.info(f"✓ 法向零梯度正确 (max_abs={max_abs:.2e})")
        return True
    else:
        logger.error(f"✗ 法向零梯度错误 (max_abs={max_abs:.2e})")
        logger.error(f"J @ n = {result}")
        return False
