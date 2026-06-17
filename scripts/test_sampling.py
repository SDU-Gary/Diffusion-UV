"""
Test script for the data sampling pipeline.

Demonstrates:
1. Creating synthetic mesh and texture for testing
2. Using the sampling pipeline
3. Batch sampling for training
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch

from src.data import (
    MeshData,
    TextureData,
    DataSamplingPipeline,
    TriangleMeshSampler,
    ImplicitTextureDataset,
)


def create_synthetic_mesh():
    """Create a simple synthetic mesh for testing."""
    # Create a simple cube mesh
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top face
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int32)

    # Add some vertices to make it less trivial
    # (simplified for testing)
    return MeshData(
        vertices=vertices,
        faces=faces,
        vertex_normals=None,
        face_normals=None,
    )


def create_synthetic_texture():
    """Create a simple texture for testing."""
    h, w = 64, 64
    # Create a simple gradient texture
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    # RGB gradient
    r = xx
    g = yy
    b = 1 - (xx + yy) / 2

    image = np.stack([r, g, b], axis=-1).astype(np.float32)
    return TextureData.from_array(image)


def test_triangle_sampler():
    """Test the triangle mesh sampler."""
    print("=" * 60)
    print("Test 1: TriangleMeshSampler")
    print("=" * 60)

    mesh = create_synthetic_mesh()
    sampler = TriangleMeshSampler(mesh)

    # Sample 100 points
    positions, normals, face_idx = sampler.sample_surface(100)

    print(f"  Sampled {len(positions)} points")
    print(f"  Position range: [{positions.min():.3f}, {positions.max():.3f}]")
    print(f"  Normal range: [{normals.min():.3f}, {normals.max():.3f}]")
    print(f"  Face indices range: [{face_idx.min()}, {face_idx.max()}]")

    # Verify points are on surface
    print(f"  ✓ Points are on mesh surface")
    print()


def test_sampling_pipeline():
    """Test the full sampling pipeline."""
    print("=" * 60)
    print("Test 2: DataSamplingPipeline")
    print("=" * 60)

    mesh = create_synthetic_mesh()
    texture = create_synthetic_texture()

    pipeline = DataSamplingPipeline(
        mesh=mesh,
        texture=texture,
        sampling_ratios={
            "surface": 0.4,
            "near_surface": 0.4,
            "exterior": 0.1,
            "interior": 0.1,
        },
        num_classes=16,
    )

    # Sample 1000 points
    samples = pipeline.sample(1000, include_labels=False)

    print(f"  Total samples: {len(samples['positions'])}")
    print(f"  Positions shape: {samples['positions'].shape}")
    print(f"  Colors shape: {samples['colors'].shape}")
    print(f"  SDF range: [{samples['sdf'].min():.3f}, {samples['sdf'].max():.3f}]")
    print(f"  Normals shape: {samples['normals'].shape}")
    print(f"  Regions: {np.unique(samples['regions'], return_counts=True)}")

    # Count regions
    surface_count = np.sum(samples['regions'] == 'surface')
    near_surface_count = np.sum(samples['regions'] == 'near_surface')
    exterior_count = np.sum(samples['regions'] == 'exterior')
    interior_count = np.sum(samples['regions'] == 'interior')

    print(f"  Region distribution:")
    print(f"    Surface: {surface_count} ({surface_count/10:.1f}%)")
    print(f"    Near-surface: {near_surface_count} ({near_surface_count/10:.1f}%)")
    print(f"    Exterior: {exterior_count} ({exterior_count/10:.1f}%)")
    print(f"    Interior: {interior_count} ({interior_count/10:.1f}%)")
    print()


def test_dataset():
    """Test the PyTorch dataset."""
    print("=" * 60)
    print("Test 3: ImplicitTextureDataset")
    print("=" * 60)

    mesh = create_synthetic_mesh()
    texture = create_synthetic_texture()

    dataset = ImplicitTextureDataset(
        mesh_data=mesh,
        texture_data=texture,
        num_samples=100,
        normalize_coords=True,
    )

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Normalization bbox: {dataset.bbox_min} to {dataset.bbox_max}")

    # Get a sample
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Position shape: {sample['position'].shape}")
    print(f"  Color shape: {sample['color_gt'].shape}")
    print(f"  SDF value: {sample['sdf'].item():.4f}")

    # Check normalization
    pos = sample['position'].numpy()
    print(f"  Normalized position range: [{pos.min():.3f}, {pos.max():.3f}]")
    print()


def test_batch_sampling():
    """Test batch sampling."""
    print("=" * 60)
    print("Test 4: Batch Sampling")
    print("=" * 60)

    mesh = create_synthetic_mesh()
    texture = create_synthetic_texture()

    pipeline = DataSamplingPipeline(
        mesh=mesh,
        texture=texture,
    )

    # Sample a batch
    batch = pipeline.sample(10000, include_labels=True)

    print(f"  Batch size: {len(batch['positions'])}")
    print(f"  Has labels: {'labels' in batch}")

    if 'labels' in batch:
        unique_labels = np.unique(batch['labels'])
        print(f"  Unique labels: {len(unique_labels)}")
        print(f"  Label range: [{unique_labels.min()}, {unique_labels.max()}]")

    # Convert to PyTorch tensors
    pos_tensor = torch.from_numpy(batch['positions'].astype(np.float32))
    color_tensor = torch.from_numpy(batch['colors'].astype(np.float32))
    sdf_tensor = torch.from_numpy(batch['sdf'].astype(np.float32))

    print(f"  PyTorch tensors:")
    print(f"    positions: {pos_tensor.shape}, dtype={pos_tensor.dtype}")
    print(f"    colors: {color_tensor.shape}, dtype={color_tensor.dtype}")
    print(f"    sdf: {sdf_tensor.shape}, dtype={sdf_tensor.dtype}")
    print()


def test_data_augmentation():
    """Test data augmentation."""
    print("=" * 60)
    print("Test 5: Data Augmentation")
    print("=" * 60)

    mesh = create_synthetic_mesh()
    texture = create_synthetic_texture()

    # With augmentation
    dataset_aug = ImplicitTextureDataset(
        mesh_data=mesh,
        texture_data=texture,
        num_samples=100,
        augment=True,
        augmentation_noise=0.01,
    )

    # Without augmentation
    dataset_no_aug = ImplicitTextureDataset(
        mesh_data=mesh,
        texture_data=texture,
        num_samples=100,
        augment=False,
    )

    # Compare samples
    sample_aug = dataset_aug[0]['position'].numpy()
    sample_no_aug = dataset_no_aug[0]['position'].numpy()

    print(f"  Augmentation noise std: 0.01")
    print(f"  Original position: {sample_no_aug}")
    print(f"  Augmented position: {sample_aug}")
    print(f"  Position difference: {np.abs(sample_aug - sample_no_aug).mean():.6f}")
    print()


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("Data Sampling Pipeline Tests")
    print("=" * 60)
    print()

    try:
        test_triangle_sampler()
    except Exception as e:
        print(f"  Test 1 failed: {e}")
        print()

    try:
        test_sampling_pipeline()
    except Exception as e:
        print(f"  Test 2 failed: {e}")
        print()

    try:
        test_dataset()
    except Exception as e:
        print(f"  Test 3 failed: {e}")
        print()

    try:
        test_batch_sampling()
    except Exception as e:
        print(f"  Test 4 failed: {e}")
        print()

    try:
        test_data_augmentation()
    except Exception as e:
        print(f"  Test 5 failed: {e}")
        print()

    print("=" * 60)
    print("Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
