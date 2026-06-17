# 3D Viewer System (Polyscope-based)

Desktop 3D viewer for Diffusion-UV project using Polyscope rendering engine.

## Features

- **Mesh Viewing**: Load and view 3D mesh files (OBJ, PLY, STL, GLTF, GLB, etc.)
- **Texture Viewing**: Display OBJ+MTL diffuse textures and GLTF/GLB base-color textures when UVs are present
- **Point Cloud Viewing**: Load and view sampling data (NPZ format)
- **Interactive Controls**: Rotate, zoom, pan with mouse
- **Color Visualization**: View texture maps, vertex colors, face colors, and point cloud colors
- **Scalar/Vector Fields**: Visualize SDF values, normals, etc.
- **High Performance**: Native OpenGL rendering through Polyscope

## Installation

Install required dependencies:

```bash
pip install polyscope trimesh numpy scipy
```

## Usage

### View Mesh Files

```bash
# View single mesh
python3 scripts/viewer_3d.py outputs/inference_results/colored_bunny.obj

# View textured OBJ. Open the OBJ, not the MTL; the OBJ references its MTL.
python3 scripts/viewer_3d.py data/models/stanford_bunny_textured.obj

# View GLTF/GLB files, including multi-node scenes when trimesh can load them
python3 scripts/viewer_3d.py scene.gltf model.glb

# View multiple meshes
python3 scripts/viewer_3d.py mesh1.obj mesh2.obj

# Using named arguments
python3 scripts/viewer_3d.py --mesh mesh1.obj --mesh mesh2.obj
```

### View Sampling Data

```bash
# View point cloud samples
python3 scripts/viewer_3d.py outputs/bunny_samples.npz

# View multiple NPZ files
python3 scripts/viewer_3d.py samples1.npz samples2.npz

# Using named arguments
python3 scripts/viewer_3d.py --samples samples.npz
```

### Combined Viewing

```bash
# View both mesh and samples
python3 scripts/viewer_3d.py colored_bunny.obj bunny_samples.npz
```

## Export Sampling Data

Export mesh samples to NPZ format for viewing:

```bash
# Basic export
python3 scripts/export_samples.py data/models/stanford-bunny.obj outputs/bunny_samples.npz

# With custom sample count
python3 scripts/export_samples.py mesh.obj samples.npz --num-samples 50000

# Include all features
python3 scripts/export_samples.py mesh.obj samples.npz --colors --normals --sdf
```

## Viewer Controls

### Mouse Controls

- **Left Click + Drag**: Rotate view
- **Right Click + Drag**: Pan view
- **Scroll Wheel**: Zoom in/out
- **Ctrl+Click**: Pick/select point

### Polyscope UI Features

- **Scene panel**: Toggle visibility of registered structures
- **Quantities panel**: Enable/disable color quantities and scalar fields
- **Options panel**: Adjust rendering settings (ground plane, lighting, etc.)
- **Screenshot**: Save current view to image file
- **Inspector**: Click points to inspect quantities and values

## NPZ File Format

Sampling data files should follow this format:

```python
{
    'points': (N, 3) array,      # Required: 3D positions
    'colors': (N, 3) array,      # Optional: RGB colors [0-1]
    'normals': (N, 3) array,     # Optional: Normal vectors
    'sdf': (N,) array,           # Optional: SDF values
}
```

## Examples

### Complete Inference Pipeline

```bash
# 1. Run inference
./scripts/run_inference_pipeline.sh

# 2. View colored mesh
python3 scripts/viewer_3d.py outputs/inference_results/colored_bunny.obj

# 3. Export samples from original mesh
python3 scripts/export_samples.py data/models/stanford-bunny.obj outputs/samples.npz --num-samples 10000

# 4. View samples
python3 scripts/viewer_3d.py outputs/samples.npz
```

### Quick Testing

```bash
# Test viewer functionality
python3 scripts/test_viewer.py

# Test with empty viewer
python3 scripts/viewer_3d.py
```

## Supported File Formats

### Mesh Files
- OBJ (.obj), including MTL diffuse textures via `map_Kd`
- PLY (.ply)
- STL (.stl)
- GLTF (.gltf, .glb)
- DAE (.dae)
- 3MF (.3mf)
- OFF (.off)
- VTK (.vtk)
- VTP (.vtp)

MTL files are material libraries and are not standalone geometry files. Open the
OBJ file which references the MTL.

### Sampling Data
- NPZ (.npz) - NumPy compressed array format

## Troubleshooting

### Viewer doesn't open
- Ensure polyscope is installed: `pip install polyscope`
- Check if display/GUI is available

### Texture or colors not showing
- For textured OBJ files, verify the OBJ references an MTL file and the MTL has a valid `map_Kd` path.
- For GLTF/GLB files, verify the asset has UVs and a base-color texture supported by trimesh.
- For vertex or face colors, verify the mesh contains color attributes.
- In the Polyscope UI, check the mesh Quantities panel and enable `texture`, `vertex_colors`, `face_colors`, or `uv` as needed.
- For NPZ files, check that the file contains a `colors` array in `[0, 1]` or `[0, 255]`.

### Performance issues
- Reduce sample count for large point clouds
- Use wireframe mode for complex meshes
- Close other applications using GPU

## Technical Notes

### Texture And Color Rendering

The viewer registers mesh visuals in this priority order:

1. UV texture image, when UVs and a material texture are available.
2. Vertex or face colors, when color attributes are available.
3. Material base color, when only a diffuse/base color is available.

For UV-only meshes with no texture image, the viewer registers a UV
parameterization quantity so seams and UV layout can still be inspected from the
Polyscope UI.

### Performance

Polyscope provides excellent performance for:
- Large meshes (100K+ vertices)
- Dense point clouds (1M+ points)
- Real-time interactive manipulation
- GPU-accelerated rendering
