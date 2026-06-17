#!/usr/bin/env python3
"""
Test script for Polyscope-based 3D viewer
"""

import sys
sys.path.append('scripts')

from viewer_3d import Viewer3D
from pathlib import Path
import numpy as np

def test_mesh_loading():
    """Test mesh file loading."""
    print("Testing mesh file loading...")

    mesh_file = "outputs/inference_results/colored_bunny.obj"
    if not Path(mesh_file).exists():
        print(f"Mesh file not found: {mesh_file}")
        print("Run inference first: ./scripts/run_inference_pipeline.sh")
        return False

    try:
        # 验证推理输出的颜色数据
        import trimesh
        mesh = trimesh.load(mesh_file)
        colors = mesh.visual.vertex_colors

        print(f"  验证颜色数据:")
        print(f"    顶点数: {len(colors)}")
        print(f"    唯一颜色: {len(np.unique(colors, axis=0))}")

        mean_r, mean_g, mean_b = colors[:, :3].mean(axis=0)
        print(f"    平均RGB: ({mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f})")

        if abs(mean_r - mean_g) > 5 or abs(mean_g - mean_b) > 5:
            print(f"    ✓ 有颜色差异（非灰色）")
        else:
            print(f"    ⚠ 接近灰色")

        viewer = Viewer3D(title="Test Viewer")

        # Test loading mesh
        if viewer.load_mesh_file(mesh_file):
            print("✓ Mesh loaded successfully with RGB colors")

            # Check mesh properties
            mesh_name = Path(mesh_file).stem
            if mesh_name in viewer.meshes:
                ps_mesh = viewer.meshes[mesh_name]
                print(f"  Registered mesh: {mesh_name}")

            return True
        else:
            print("✗ Failed to load mesh")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_npz_format():
    """Test NPZ sampling data format."""
    print("\nTesting NPZ format...")

    import numpy as np

    # Create test NPZ file
    test_npz = "/tmp/test_samples.npz"

    # Generate test data
    npoints = 1000
    points = np.random.randn(npoints, 3).astype(np.float32)
    colors = np.random.rand(npoints, 3).astype(np.float32)
    sdf = np.random.randn(npoints).astype(np.float32)

    # Save NPZ
    np.savez(test_npz, points=points, colors=colors, sdf=sdf)
    print(f"✓ Created test NPZ file: {test_npz}")

    # Test loading
    try:
        viewer = Viewer3D(title="NPZ Test")

        if viewer.load_sampling_data(test_npz):
            print("✓ NPZ loaded successfully")

            # Check point cloud
            cloud_name = Path(test_npz).stem
            if cloud_name in viewer.point_clouds:
                print(f"  Registered point cloud: {cloud_name}")

            return True
        else:
            print("✗ Failed to load NPZ")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Polyscope Viewer Tests")
    print("=" * 60)

    results = []

    # Test 1: Mesh loading
    results.append(("Mesh Loading", test_mesh_loading()))

    # Test 2: NPZ format
    results.append(("NPZ Format", test_npz_format()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed! ✓" if all_passed else "Some tests failed ✗"))

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
