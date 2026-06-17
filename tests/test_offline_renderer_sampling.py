import numpy as np

from src.inference.offline_renderer import OfflineRenderer


def test_offline_renderer_uses_bilinear_texture_sampling():
    texture = np.array(
        [
            [[0, 0, 0], [100, 0, 0]],
            [[0, 100, 0], [0, 0, 100]],
        ],
        dtype=np.uint8,
    )
    renderer = OfflineRenderer(
        mesh_vertices=np.zeros((3, 3), dtype=np.float32),
        mesh_faces=np.array([[0, 1, 2]], dtype=np.int32),
        texture_image=texture,
        resolution=(1, 1),
    )

    valid_mask = np.array([[True]])
    uv_center = np.array([[0.5, 0.5]], dtype=np.float32)

    sampled = renderer._sample_texture(uv_center, valid_mask)

    # Center UV must average all four texels. Nearest-neighbor sampling would
    # return one corner color instead.
    np.testing.assert_array_equal(sampled[0, 0], np.array([25, 25, 25], dtype=np.uint8))
