#!/usr/bin/env python3
"""
Desktop 3D Viewer for Diffusion-UV (Polyscope-based)

A unified viewer for:
- 3D mesh files (OBJ, PLY, STL, etc.)
- Sampling data (NPZ format)
- Vertex colors and point clouds

Usage:
    python scripts/viewer_3d.py                    # Show empty viewer
    python scripts/viewer_3d.py mesh.obj          # Load mesh file
    python scripts/viewer_3d.py samples.npz       # Load sampling data
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, Tuple
import numpy as np
import logging

# Polyscope import
try:
    import polyscope as ps
    POLYSCOPE_AVAILABLE = True
except ImportError:
    POLYSCOPE_AVAILABLE = False
    print("Error: polyscope not available. Install with: pip install polyscope")

# Mesh processing
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Install with: pip install trimesh")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Viewer3D:
    """
    Desktop 3D Viewer based on Polyscope.

    Features:
    - Load mesh files (OBJ, PLY, STL, etc.)
    - Load textured mesh files (OBJ+MTL, GLTF, GLB, etc.)
    - Load sampling data (NPZ format)
    - Interactive camera controls (rotate, zoom, pan)
    - Texture, vertex color, and face color visualization
    - Point cloud visualization
    """

    def __init__(self, title: str = "Diffusion-UV 3D Viewer"):
        """Initialize Polyscope viewer."""
        if not POLYSCOPE_AVAILABLE:
            raise ImportError("polyscope is required for 3D viewing. Install with: pip install polyscope")

        # Initialize polyscope
        ps.init()

        # Track registered structures
        self.meshes = {}
        self.point_clouds = {}

        logger.info(f"Initialized Polyscope viewer: {title}")

    def load_mesh_file(self, file_path: str, name: Optional[str] = None) -> bool:
        """
        Load a 3D mesh file.

        Args:
            file_path: Path to mesh file (OBJ, PLY, STL, etc.)
            name: Optional name for the mesh

        Returns:
            True if loaded successfully
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        name = name or file_path.stem

        try:
            if not TRIMESH_AVAILABLE:
                logger.error("Cannot load mesh - trimesh not available")
                return False

            # Load through a Scene so OBJ, GLTF, GLB, and multi-node files share
            # one path. scene.dump(concatenate=False) applies node transforms
            # while preserving per-geometry visuals in trimesh.
            scene_or_mesh = trimesh.load(file_path, force="scene", process=False)
            meshes = list(self._iter_scene_meshes(scene_or_mesh))

            loaded = 0
            for i, mesh in enumerate(meshes):
                mesh_name = self._mesh_display_name(name, mesh, i, len(meshes))
                loaded += self._register_trimesh_mesh(mesh, mesh_name)

            if loaded == 0:
                logger.warning(f"No renderable triangle meshes found in: {file_path}")
                return False

            logger.info(f"Loaded mesh file: {file_path} ({loaded} surface mesh item(s))")
            return True

        except Exception as e:
            logger.error(f"Error loading mesh: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _iter_scene_meshes(self, scene_or_mesh: Any) -> Iterable[Any]:
        """Yield renderable Trimesh objects from a loaded mesh or scene."""
        if isinstance(scene_or_mesh, trimesh.Trimesh):
            yield scene_or_mesh
            return

        if isinstance(scene_or_mesh, trimesh.Scene):
            for geom in scene_or_mesh.dump(concatenate=False):
                if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                    yield geom
            return

        logger.warning(f"Unsupported loaded geometry type: {type(scene_or_mesh).__name__}")

    def _mesh_display_name(self, base_name: str, mesh: Any, index: int, total: int) -> str:
        """Build a stable display name for a geometry inside a file."""
        mesh_name = mesh.metadata.get("name") or mesh.metadata.get("node") if hasattr(mesh, "metadata") else None
        if mesh_name and str(mesh_name) != base_name:
            candidate = f"{base_name}:{mesh_name}"
        else:
            candidate = base_name

        if total > 1 and index > 0:
            candidate = f"{candidate}_{index}"

        return str(candidate)

    def _register_trimesh_mesh(self, mesh: Any, name: str) -> int:
        """
        Register a trimesh object in Polyscope.

        Multi-material textured meshes are split by material so each subset can
        bind its own texture image. Single-material meshes are registered as one
        item.
        """
        split = self._split_by_material(mesh, name)
        registered = 0

        for submesh, subname, material in split:
            vertices = np.asarray(submesh.vertices, dtype=np.float32)
            faces = np.asarray(submesh.faces, dtype=np.int32)

            if vertices.size == 0 or faces.size == 0:
                continue

            logger.info(f"Loading mesh: {subname}")
            logger.info(f"  Vertices: {len(vertices)}")
            logger.info(f"  Faces: {len(faces)}")

            ps_mesh = ps.register_surface_mesh(
                subname,
                vertices,
                faces,
                smooth_shade=True
            )

            texture_added = self._add_texture_quantity(ps_mesh, submesh, material)
            color_added = self._add_color_quantities(ps_mesh, submesh, enabled=not texture_added)

            if not texture_added and not color_added:
                self._apply_material_base_color(ps_mesh, material)

            self.meshes[subname] = ps_mesh
            logger.info(f"Registered mesh: {subname}")
            registered += 1

        return registered

    def _split_by_material(self, mesh: Any, name: str) -> Iterable[Tuple[Any, str, Any]]:
        """Split multi-material meshes into material-specific submeshes."""
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None)
        face_materials = getattr(visual, "face_materials", None)

        if (
            material is not None
            and material.__class__.__name__ == "MultiMaterial"
            and face_materials is not None
            and len(face_materials) == len(mesh.faces)
        ):
            uv = getattr(visual, "uv", None)
            unique_ids = np.unique(face_materials)

            for material_id in unique_ids:
                mask = face_materials == material_id
                if not np.any(mask):
                    continue

                submesh = self._make_face_subset(mesh, mask, uv)
                sub_material = material.get(int(material_id))
                subname = f"{name}:mat_{int(material_id)}"
                mat_name = getattr(sub_material, "name", None)
                if mat_name:
                    subname = f"{name}:{mat_name}"

                yield submesh, subname, sub_material
            return

        yield mesh, name, material

    def _make_face_subset(self, mesh: Any, face_mask: np.ndarray, uv: Optional[np.ndarray]) -> Any:
        """Create a compact submesh for a face mask and preserve UVs when possible."""
        selected_faces = np.asarray(mesh.faces[face_mask], dtype=np.int64)
        used_vertices = np.unique(selected_faces.reshape(-1))
        remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
        remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)

        sub_vertices = np.asarray(mesh.vertices[used_vertices], dtype=np.float64)
        sub_faces = remap[selected_faces]

        submesh = trimesh.Trimesh(
            vertices=sub_vertices,
            faces=sub_faces,
            process=False
        )

        if uv is not None:
            uv = np.asarray(uv, dtype=np.float32)
            if len(uv) == len(mesh.vertices):
                sub_uv = uv[used_vertices]
            elif len(uv) == len(mesh.faces) * 3:
                sub_uv = uv.reshape(len(mesh.faces), 3, 2)[face_mask].reshape(-1, 2)
            else:
                sub_uv = None

            if sub_uv is not None:
                submesh.visual = trimesh.visual.TextureVisuals(uv=sub_uv)

        return submesh

    def _add_texture_quantity(self, ps_mesh: Any, mesh: Any, material: Any) -> bool:
        """Add a real UV texture quantity when the mesh has UVs and an image."""
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None)
        if uv is None:
            return False

        uv_values, defined_on = self._prepare_uvs(mesh, uv)
        if uv_values is None:
            logger.warning("  UVs present but shape does not match vertices or corners; skipping texture")
            return False

        image = self._material_image(material)
        param_name = "uv"
        ps_mesh.add_parameterization_quantity(
            param_name,
            uv_values,
            defined_on=defined_on,
            coords_type="unit",
            enabled=image is None
        )
        logger.info(f"  UV: {defined_on} parameterization ({len(uv_values)} coords)")

        if image is None:
            logger.info("  UV visualization enabled; no texture image found")
            return False

        texture = self._image_to_float_rgb(image)
        if texture is None:
            logger.warning("  Texture image could not be converted to RGB array")
            return False

        ps_mesh.add_color_quantity(
            "texture",
            texture,
            defined_on="texture",
            param_name=param_name,
            image_origin="upper_left",
            enabled=True
        )
        logger.info(f"  Texture: {texture.shape[1]}x{texture.shape[0]} RGB")
        return True

    def _prepare_uvs(self, mesh: Any, uv: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Return UVs and Polyscope domain name: vertices or corners."""
        uv = np.asarray(uv, dtype=np.float32)
        if uv.ndim != 2 or uv.shape[1] != 2:
            return None, None

        if len(uv) == len(mesh.vertices):
            return uv, "vertices"

        corner_count = len(mesh.faces) * 3
        if len(uv) == corner_count:
            return uv.reshape(corner_count, 2), "corners"

        return None, None

    def _material_image(self, material: Any) -> Any:
        """Extract a diffuse/base-color texture image from common trimesh materials."""
        if material is None:
            return None

        for attr in ("image", "baseColorTexture"):
            image = getattr(material, attr, None)
            if image is not None:
                return image

        to_simple = getattr(material, "to_simple", None)
        if callable(to_simple):
            try:
                simple = to_simple()
                image = getattr(simple, "image", None)
                if image is not None:
                    return image
            except Exception:
                pass

        return None

    def _image_to_float_rgb(self, image: Any) -> Optional[np.ndarray]:
        """Convert PIL/numpy texture images to HxWx3 float32 RGB in [0, 1]."""
        try:
            if hasattr(image, "convert"):
                arr = np.asarray(image.convert("RGB"), dtype=np.float32)
            else:
                arr = np.asarray(image, dtype=np.float32)

            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
            else:
                return None

            if arr.max(initial=0.0) > 1.0:
                arr = arr / 255.0

            return np.clip(arr, 0.0, 1.0).astype(np.float32)
        except Exception:
            return None

    def _add_color_quantities(self, ps_mesh: Any, mesh: Any, enabled: bool = True) -> bool:
        """Add vertex or face color quantities when available."""
        visual = getattr(mesh, "visual", None)
        added = False

        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None:
            colors = self._normalize_colors(vertex_colors)
            if colors is not None and len(colors) == len(mesh.vertices):
                ps_mesh.add_color_quantity("vertex_colors", colors, defined_on="vertices", enabled=enabled)
                logger.info(f"  Vertex colors: RGB range [{colors.min():.3f}, {colors.max():.3f}]")
                added = True

        face_colors = getattr(visual, "face_colors", None)
        if face_colors is not None:
            colors = self._normalize_colors(face_colors)
            if colors is not None and len(colors) == len(mesh.faces):
                ps_mesh.add_color_quantity("face_colors", colors, defined_on="faces", enabled=enabled and not added)
                logger.info(f"  Face colors: RGB range [{colors.min():.3f}, {colors.max():.3f}]")
                added = True

        return added

    def _normalize_colors(self, colors: Any) -> Optional[np.ndarray]:
        """Normalize Nx3/Nx4 color arrays to float32 Nx3 in [0, 1]."""
        arr = np.asarray(colors)
        if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
            return None

        arr = arr[:, :3].astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr = arr / 255.0

        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _apply_material_base_color(self, ps_mesh: Any, material: Any) -> None:
        """Use material diffuse/base color as the Polyscope base color."""
        color = self._material_base_color(material)
        if color is not None:
            ps_mesh.set_color(color.tolist())
            logger.info(f"  Material base color: [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]")

    def _material_base_color(self, material: Any) -> Optional[np.ndarray]:
        """Extract an RGB base color from common material types."""
        if material is None:
            return None

        for attr in ("main_color", "diffuse", "baseColorFactor"):
            color = getattr(material, attr, None)
            if color is not None:
                arr = np.asarray(color, dtype=np.float32).reshape(-1)
                if len(arr) >= 3:
                    arr = arr[:3]
                    if arr.max(initial=0.0) > 1.0:
                        arr = arr / 255.0
                    return np.clip(arr, 0.0, 1.0).astype(np.float32)

        return None

    def load_sampling_data(self, file_path: str, name: Optional[str] = None) -> bool:
        """
        Load sampling data from NPZ file and display as colored point cloud.

        Expected NPZ format:
        - points: (N, 3) point positions (REQUIRED)
        - colors: (N, 3) optional RGB colors
        - normals: (N, 3) optional normals
        - sdf: (N,) optional SDF values
        - uvs: (N, 2) optional UV coordinates

        Args:
            file_path: Path to NPZ file
            name: Optional name for the point cloud

        Returns:
            True if loaded successfully
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        name = name or file_path.stem

        try:
            # Load NPZ data
            data = np.load(file_path)

            if 'points' not in data:
                logger.error("NPZ file must contain 'points' array")
                return False

            points = data['points'].astype(np.float32)

            logger.info(f"Loading sampling data: {name}")
            logger.info(f"  Points: {len(points)}")

            # Register point cloud in polyscope with sphere rendering
            ps_cloud = ps.register_point_cloud(name, points, point_render_mode='sphere')

            # Add colors if present
            if 'colors' in data:
                colors = data['colors']

                # Normalize to [0,1] if needed
                if colors.max() > 1.0:
                    colors = colors / 255.0

                colors = colors.astype(np.float32)

                # Add RGB color quantity for point cloud
                ps_cloud.add_color_quantity("colors", colors)

                logger.info(f"  Colors: RGB range [{colors.min():.3f}, {colors.max():.3f}]")
                logger.info(f"  RGB color visualization enabled")

                # Set base color to mean color
                mean_color = colors.mean(axis=0).tolist()
                ps_cloud.set_color(mean_color)

            # Add normals if present
            if 'normals' in data:
                normals = data['normals'].astype(np.float32)
                ps_cloud.add_vector_quantity("normals", normals)
                logger.info(f"  Normals: ✓")

            # Add SDF values if present
            if 'sdf' in data:
                sdf = data['sdf'].astype(np.float32)
                ps_cloud.add_scalar_quantity("sdf", sdf, cmap='coolwarm')
                logger.info(f"  SDF range: [{sdf.min():.3f}, {sdf.max():.3f}]")

            # Add UV coordinates if present
            if 'uvs' in data:
                uvs = data['uvs'].astype(np.float32)
                # Display UV as a 2D scalar (using U coordinate for visualization)
                ps_cloud.add_scalar_quantity("uv_u", uvs[:, 0], cmap='viridis')
                ps_cloud.add_scalar_quantity("uv_v", uvs[:, 1], cmap='plasma')
                logger.info(f"  UV coordinates: ✓")

            # Display metadata if present
            if 'metadata' in data:
                metadata = data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
                logger.info(f"  Metadata: {list(metadata.keys()) if isinstance(metadata, dict) else 'present'}")

            self.point_clouds[name] = ps_cloud
            logger.info(f"✓ Registered point cloud: {name} ({len(points)} points)")
            return True

        except Exception as e:
            logger.error(f"Error loading sampling data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def show(self):
        """Display the viewer."""
        logger.info("Opening Polyscope viewer...")
        logger.info("Controls:")
        logger.info("  Left-click + drag: Rotate")
        logger.info("  Right-click + drag: Pan")
        logger.info("  Scroll: Zoom")
        logger.info("  Ctrl+click: Pick point")
        ps.show()


def main():
    parser = argparse.ArgumentParser(
        description="Desktop 3D Viewer for Diffusion-UV (Polyscope-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Empty viewer
  python scripts/viewer_3d.py

  # Load mesh file
  python scripts/viewer_3d.py outputs/inference_results/colored_bunny.obj

  # Load textured OBJ with its referenced MTL/texture
  python scripts/viewer_3d.py data/models/stanford_bunny_textured.obj

  # Load GLTF/GLB scene or mesh
  python scripts/viewer_3d.py scene.gltf model.glb

  # Load sampling data
  python scripts/viewer_3d.py outputs/bunny_samples.npz

  # Load multiple files
  python scripts/viewer_3d.py mesh1.obj mesh2.obj samples.npz
        """
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Files to view (mesh files or NPZ sampling data)"
    )

    parser.add_argument(
        "--mesh",
        action="append",
        help="Load mesh file (can be used multiple times)"
    )

    parser.add_argument(
        "--samples",
        action="append",
        help="Load sampling data NPZ file (can be used multiple times)"
    )

    parser.add_argument(
        "--title",
        default="Diffusion-UV 3D Viewer",
        help="Window title"
    )

    args = parser.parse_args()

    # Check dependencies
    if not POLYSCOPE_AVAILABLE:
        logger.error("Error: polyscope is required. Install with:")
        logger.error("  pip install polyscope")
        return 1

    if not TRIMESH_AVAILABLE:
        logger.warning("Warning: trimesh not available. Install with: pip install trimesh")
        logger.warning("Mesh loading will not work without trimesh")

    # Create viewer
    try:
        viewer = Viewer3D(title=args.title)
    except Exception as e:
        logger.error(f"Failed to initialize viewer: {e}")
        return 1

    # Collect files to load
    files_to_load = []

    # Positional arguments
    files_to_load.extend(args.files or [])

    # Named arguments
    if args.mesh:
        files_to_load.extend(args.mesh)
    if args.samples:
        files_to_load.extend(args.samples)

    # If no files provided
    if not files_to_load:
        logger.info("No files specified. Launching empty viewer...")
        logger.info("Supported formats:")
        logger.info("  Mesh: OBJ(+MTL), PLY, STL, GLTF, GLB, etc.")
        logger.info("  Sampling: NPZ with 'points' array")
    else:
        logger.info(f"Loading {len(files_to_load)} file(s)...")

    # Load files
    loaded_count = 0
    for file_path in files_to_load:
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        # Determine file type
        suffix = file_path.suffix.lower()

        if suffix == '.npz':
            # Sampling data
            if viewer.load_sampling_data(str(file_path)):
                loaded_count += 1

        elif suffix in ['.obj', '.ply', '.stl', '.gltf', '.glb', '.off', '.vtk', '.vtp', '.dae', '.3mf']:
            # Mesh file
            if viewer.load_mesh_file(str(file_path)):
                loaded_count += 1

        elif suffix == '.mtl':
            logger.warning("MTL files are material libraries, not standalone meshes.")
            logger.warning("  Open the OBJ file which references this MTL instead.")

        else:
            logger.warning(f"Unsupported file type: {suffix}")
            logger.warning(f"  Supported: OBJ(+MTL), PLY, STL, GLTF, GLB, OFF, VTK, VTP, DAE, 3MF, NPZ")

    # Show viewer
    if loaded_count > 0:
        logger.info(f"\n✓ Loaded {loaded_count} file(s)")
    else:
        logger.info("\nNo files loaded. Showing empty viewer...")

    viewer.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
