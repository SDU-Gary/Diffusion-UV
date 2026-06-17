import torch

from src.models.encoders import BSplineHashGrid
from src.models.metric_aligned_iuv_field import create_model
from src.training.metric_aligned_iuv_losses import compute_metric_aligned_iuv_loss


def test_bspline_hash_grid_torch_path_shape_and_gradients():
    encoder = BSplineHashGrid(
        num_levels=2,
        features_per_level=2,
        log2_hashmap_size=5,
        base_res=4,
        max_res=8,
        cuda_backend="torch",
        normalize_positions=False,
    )
    positions = torch.rand(6, 3, requires_grad=True)

    features = encoder(positions)
    loss = features.sum()
    loss.backward()

    assert features.shape == (6, 4)
    assert positions.grad is not None
    assert encoder.hash_table.grad is not None
    assert not torch.isnan(positions.grad).any()
    assert not torch.isnan(encoder.hash_table.grad).any()


def test_bspline_hash_model_supports_metric_loss_backward():
    model = create_model(
        num_charts=2,
        hidden_dim=16,
        num_layers=2,
        encoder_type="bspline_hash",
        hash_num_levels=2,
        hash_features_per_level=2,
        hash_log2_size=5,
        hash_base_res=4,
        hash_max_res=8,
        hash_cuda_backend="torch",
        bbox_min=[0.0, 0.0, 0.0],
        bbox_max=[1.0, 1.0, 1.0],
        activation="silu",
    )
    positions = torch.rand(8, 3, requires_grad=True)
    output = model(positions)

    loss_dict = compute_metric_aligned_iuv_loss(
        model_output=output,
        pos=positions,
        j_3d_gt=torch.randn(8, 2, 3),
        uv_anchor=torch.rand(8, 2),
        chart_id=torch.randint(0, 2, (8,)),
        metric_weight=0.01,
        anchor_weight=1.0,
        cls_weight=1.0,
    )
    loss_dict["total"].backward()

    assert positions.grad is not None
    assert model.grid_encoder.hash_table.grad is not None
    assert not torch.isnan(model.grid_encoder.hash_table.grad).any()
