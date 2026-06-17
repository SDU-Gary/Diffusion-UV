"""
Visualize Stanford Bunny UV and Texture.

This script:
1. Loads the bunny mesh
2. Generates spherical UV coordinates
3. Creates a procedural texture
4. Visualizes UV layout and texture side by side
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import os

def spherical_uv(vertices):
    """
    Generate spherical UV coordinates for mesh vertices.

    Args:
        vertices: (N, 3) vertex positions

    Returns:
        uvs: (N, 2) UV coordinates in [0, 1] range
    """
    # Center vertices
    centroid = vertices.mean(axis=0)
    v_centered = vertices - centroid

    # Normalize to unit sphere
    norms = np.linalg.norm(v_centered, axis=1, keepdims=True)
    v_normalized = v_centered / (norms + 1e-8)

    # Spherical coordinates
    x, y, z = v_normalized[:, 0], v_normalized[:, 1], v_normalized[:, 2]

    # Theta: angle from XZ plane (elevation)
    theta = np.arcsin(np.clip(y, -1, 1))

    # Phi: angle in XZ plane (azimuth)
    phi = np.arctan2(z, x)

    # Convert to UV [0, 1]
    u = (phi / (2 * np.pi) + 0.5).astype(np.float32)
    v = (theta / np.pi + 0.5).astype(np.float32)

    return np.stack([u, v], axis=1)


def create_procedural_texture(width=512, height=512):
    """
    Create a visually interesting procedural texture.

    Returns:
        texture: (H, W, 3) RGB texture
    """
    u = np.linspace(0, 1, width)
    v = np.linspace(0, 1, height)
    U, V = np.meshgrid(u, v)

    # Base colors
    r = 0.7 + 0.3 * np.sin(U * 2 * np.pi * 4)
    g = 0.5 + 0.3 * np.sin(V * 2 * np.pi * 3)
    b = 0.6 + 0.3 * np.sin((U + V) * 2 * np.pi * 2)

    # Add checkerboard pattern
    checker = ((np.floor(U * 8) + np.floor(V * 8)) % 2).astype(float) * 0.15
    r = r + checker
    g = g + checker * 0.8
    b = b + checker * 0.6

    # Add radial gradient
    center_u, center_v = 0.5, 0.5
    dist = np.sqrt((U - center_u)**2 + (V - center_v)**2)
    radial = np.exp(-dist * 3) * 0.3
    r = r + radial
    g = g + radial * 0.7
    b = b + radial * 0.4

    # Clamp
    texture = np.stack([r, g, b], axis=2)
    texture = np.clip(texture, 0, 1).astype(np.float32)

    return texture


def visualize_uv_and_texture(mesh_path, output_dir):
    """Main visualization function."""

    # Load mesh
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)

    vertices = mesh.vertices
    faces = mesh.faces

    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")

    # Generate UV coordinates
    print("Generating spherical UV coordinates...")
    uvs = spherical_uv(vertices)

    print(f"  UV range: U=[{uvs[:,0].min():.3f}, {uvs[:,0].max():.3f}], "
          f"V=[{uvs[:,1].min():.3f}, {uvs[:,1].max():.3f}]")

    # Create procedural texture
    print("Creating procedural texture...")
    texture = create_procedural_texture(512, 512)

    # Get face UVs (average of vertex UVs)
    face_uvs = uvs[faces].mean(axis=1)

    # Downsample for visualization (scatter too slow for 35k points)
    vis_idx = np.random.choice(len(vertices), min(10000, len(vertices)), replace=False)

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 1. UV Layout (bottom left) - show vertex UVs
    ax1 = fig.add_subplot(2, 3, 1)
    scatter = ax1.scatter(uvs[vis_idx, 0], uvs[vis_idx, 1], c=uvs[vis_idx, 0], cmap='viridis',
                          s=1, alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('U')
    ax1.set_ylabel('V')
    ax1.set_title('UV Layout (colored by U)')
    ax1.set_aspect('equal')
    ax1.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
    plt.colorbar(scatter, ax=ax1, label='U value')

    # 2. UV Layout colored by V (bottom middle-left)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(uvs[vis_idx, 0], uvs[vis_idx, 1], c=uvs[vis_idx, 1], cmap='plasma',
                           s=1, alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('U')
    ax2.set_ylabel('V')
    ax2.set_title('UV Layout (colored by V)')
    ax2.set_aspect('equal')
    ax2.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
    plt.colorbar(scatter2, ax=ax2, label='V value')

    # 3. Texture Image (bottom middle)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(texture)
    ax3.set_title('Procedural Texture')
    ax3.set_xlabel('U (0 to 1)')
    ax3.set_ylabel('V (0 to 1)')

    # 4. 3D Mesh with UV coloring (top left)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    mesh_obj = ax4.scatter(vertices[vis_idx, 0], vertices[vis_idx, 1], vertices[vis_idx, 2],
                           c=uvs[vis_idx, 0], cmap='viridis', s=1, alpha=0.5)
    ax4.set_title('3D Mesh (colored by U)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    # 5. 3D Mesh colored by V (top middle-left)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.scatter(vertices[vis_idx, 0], vertices[vis_idx, 1], vertices[vis_idx, 2],
                c=uvs[vis_idx, 1], cmap='plasma', s=1, alpha=0.5)
    ax5.set_title('3D Mesh (colored by V)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')

    # 6. UV Coverage Heatmap (top right)
    ax6 = fig.add_subplot(2, 3, 6)
    # Create 2D histogram of UV coverage
    hist, xedges, yedges = np.histogram2d(uvs[:, 0], uvs[:, 1], bins=64, range=[[0, 1], [0, 1]])
    im = ax6.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', aspect='equal')
    ax6.set_xlabel('U')
    ax6.set_ylabel('V')
    ax6.set_title('UV Coverage Density')
    plt.colorbar(im, ax=ax6, label='Sample Count')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'bunny_uv_texture_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")

    # Also save UV coordinates and texture for use
    np.savez(os.path.join(output_dir, 'bunny_uvs.npz'),
             uvs=uvs, faces=faces)
    plt.imsave(os.path.join(output_dir, 'bunny_texture.png'), texture)
    print(f"Saved UV coordinates: {output_dir}/bunny_uvs.npz")
    print(f"Saved texture: {output_dir}/bunny_texture.png")

    # Create standalone texture visualization
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Full texture
    axes[0].imshow(texture)
    axes[0].set_title('Procedural Texture (512x512)')
    axes[0].set_xlabel('U pixel')
    axes[0].set_ylabel('V pixel')

    # UV wireframe overlay
    axes[1].imshow(texture)
    for i in range(0, 512, 32):
        axes[1].axhline(i, color='white', linewidth=0.5, alpha=0.5)
        axes[1].axvline(i, color='white', linewidth=0.5, alpha=0.5)
    axes[1].set_title('Texture with UV Grid (32x32)')
    axes[1].set_xlabel('U pixel')
    axes[1].set_ylabel('V pixel')

    # Sample some face centers in UV space
    sample_idx = np.random.choice(len(face_uvs), min(500, len(face_uvs)), replace=False)
    sample_face_uvs = face_uvs[sample_idx]

    axes[2].imshow(texture)
    axes[2].scatter(sample_face_uvs[:, 0] * 512, sample_face_uvs[:, 1] * 512,
                    c='cyan', s=2, alpha=0.7, label='Face centers')
    axes[2].set_title('Texture with Face UV Samples')
    axes[2].set_xlabel('U pixel')
    axes[2].set_ylabel('V pixel')
    axes[2].legend()

    plt.tight_layout()
    texture_viz_path = os.path.join(output_dir, 'bunny_texture_detail.png')
    plt.savefig(texture_viz_path, dpi=150, bbox_inches='tight')
    print(f"Saved texture detail: {texture_viz_path}")

    plt.close('all')
    print("Done!")

    return uvs, texture


if __name__ == '__main__':
    mesh_path = 'data/models/stanford-bunny.obj'
    output_dir = 'data/textures'

    os.makedirs(output_dir, exist_ok=True)

    uvs, texture = visualize_uv_and_texture(mesh_path, output_dir)

    print("\nSummary:")
    print(f"  UV coordinates shape: {uvs.shape}")
    print(f"  Texture shape: {texture.shape}")
    print(f"  UV coordinate range: [{uvs.min():.3f}, {uvs.max():.3f}]")
