import math
import tempfile
from pathlib import Path

import torch

from src.data.gpu_constant_baker import (
    MetricAlignedIUVGPUConstantBaker,
    load_mesh_constants,
)
from src.data.gpu_dataset import GPUDynamicSampleGenerator
from src.models.metric_aligned_iuv_field import create_model
from scripts.train_metric_aligned_iuv_field import compute_loss_schedule, train_epoch_dynamic


def write_square_obj(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "f 1/1 2/2 3/3",
                "f 1/1 3/3 4/4",
                "",
            ]
        )
    )


def make_constants(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "square.obj"
    constants_path = tmp_path / "mesh_constants.pt"
    write_square_obj(mesh_path)
    baker = MetricAlignedIUVGPUConstantBaker(str(mesh_path), seed=7)
    baker.save(
        output_path=str(constants_path),
        extrusion_sigma_ratio=0.01,
        chart_mode="face_component",
    )
    return constants_path


def test_gpu_constant_baker_schema(tmp_path: Path):
    constants_path = make_constants(tmp_path)
    constants, metadata = load_mesh_constants(str(constants_path), map_location="cpu")

    assert constants["face_vertices"].shape == (2, 3, 3)
    assert constants["face_normals"].shape == (2, 3, 3)
    assert constants["face_uvs"].shape == (2, 3, 2)
    assert constants["face_j_3d_gt"].shape == (2, 2, 3)
    assert constants["face_chart_id"].shape == (2,)
    assert constants["face_probs"].shape == (2,)
    assert torch.isclose(constants["face_probs"].sum(), torch.tensor(1.0))
    assert torch.isfinite(constants["face_j_3d_gt"]).all()
    assert metadata["data_kind"] == "mesh_constants"
    assert metadata["num_charts"] == 1
    assert len(metadata["face_chart_id"]) == 2


def test_dynamic_generator_batch_shapes_and_fresh_samples(tmp_path: Path):
    constants_path = make_constants(tmp_path)
    gen = GPUDynamicSampleGenerator(
        str(constants_path),
        batch_size=64,
        sigma=0.0,
        device="cpu",
        seed=123,
    )

    batch_a = gen.next_batch()
    batch_b = gen.next_batch()

    assert batch_a["pos"].shape == (64, 3)
    assert batch_a["j_3d_gt"].shape == (64, 2, 3)
    assert batch_a["uv_anchor"].shape == (64, 2)
    assert batch_a["chart_id"].shape == (64,)
    assert torch.isfinite(batch_a["pos"]).all()
    assert torch.isfinite(batch_a["uv_anchor"]).all()
    assert torch.allclose(batch_a["pos"][:, 2], torch.zeros(64), atol=1e-6)
    assert not torch.allclose(batch_a["pos"], batch_b["pos"])


def test_dynamic_training_epoch_runs(tmp_path: Path):
    constants_path = make_constants(tmp_path)
    gen = GPUDynamicSampleGenerator(
        str(constants_path),
        batch_size=16,
        sigma=0.0,
        device="cpu",
        seed=321,
    )
    model = create_model(
        num_charts=1,
        hidden_dim=16,
        num_layers=2,
        encoder_type="fourier",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = train_epoch_dynamic(
        model=model,
        generator=gen,
        optimizer=optimizer,
        device=torch.device("cpu"),
        metric_weight=0.01,
        anchor_weight=1.0,
        cls_weight=1.0,
        steps_per_epoch=2,
    )

    for key in ["loss", "metric", "anchor", "com", "cls", "cls_acc"]:
        assert key in metrics
        assert math.isfinite(metrics[key])


def test_two_stage_loss_schedule_boundaries():
    phase_a = compute_loss_schedule(
        epoch=30,
        total_epochs=100,
        base_metric_weight=0.01,
        base_anchor_weight=1.0,
        base_cls_weight=1.0,
        schedule_name="two_stage",
        phase_a_epochs=30,
        target_metric_weight=1.0,
        target_anchor_weight=0.01,
        target_cls_weight=0.1,
    )
    final = compute_loss_schedule(
        epoch=100,
        total_epochs=100,
        base_metric_weight=0.01,
        base_anchor_weight=1.0,
        base_cls_weight=1.0,
        schedule_name="two_stage",
        phase_a_epochs=30,
        target_metric_weight=1.0,
        target_anchor_weight=0.01,
        target_cls_weight=0.1,
    )

    assert phase_a["metric_weight"] == 0.01
    assert phase_a["anchor_weight"] == 1.0
    assert phase_a["com_weight"] == 0.0
    assert phase_a["cls_weight"] == 1.0
    assert math.isclose(final["metric_weight"], 1.0, rel_tol=1e-6)
    assert math.isclose(final["anchor_weight"], 0.01, rel_tol=1e-6)
    assert math.isclose(final["com_weight"], 0.0, rel_tol=1e-6)
    assert math.isclose(final["cls_weight"], 0.1, rel_tol=1e-6)


def test_fixed_loss_schedule():
    weights = compute_loss_schedule(
        epoch=75,
        total_epochs=100,
        base_metric_weight=0.02,
        base_anchor_weight=0.5,
        base_cls_weight=0.3,
        schedule_name="fixed",
    )
    assert weights == {
        "metric_weight": 0.02,
        "anchor_weight": 0.5,
        "com_weight": 0.0,
        "cls_weight": 0.3,
    }
