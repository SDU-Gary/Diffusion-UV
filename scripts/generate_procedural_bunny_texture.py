#!/usr/bin/env python3
"""
Generate a procedural textured Stanford bunny asset for MA-IUVF experiments.

The script starts from an OBJ without UVs, partitions the mesh into a fixed
number of connected charts, builds a face-corner UV atlas, writes OBJ/MTL/PNG
files, and runs a lightweight validation through the project's OBJ parser and
UV chart splitter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE_CHART_COLORS = [
    (222, 72, 82),
    (69, 150, 225),
    (72, 178, 110),
    (232, 179, 63),
    (157, 104, 219),
    (53, 188, 184),
    (235, 111, 36),
    (118, 168, 54),
    (236, 92, 167),
    (83, 111, 216),
    (188, 155, 54),
    (78, 178, 170),
]


@dataclass
class GeneratedAsset:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    face_uv_indices: np.ndarray
    face_chart_ids: np.ndarray
    chart_info: List[Dict]


def load_mesh(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load OBJ geometry without changing index order."""
    mesh = trimesh.load(str(mesh_path), process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) != 1:
            raise ValueError(f"Expected one geometry in scene, got {len(mesh.geometry)}")
        mesh = next(iter(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("This generator expects a triangulated mesh")
    return vertices, faces


def build_face_adjacency(faces: np.ndarray) -> List[List[int]]:
    """Build face adjacency from shared mesh edges."""
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_to_faces[tuple(sorted((int(a), int(b))))].append(face_idx)

    adjacency = [set() for _ in range(len(faces))]
    for touching_faces in edge_to_faces.values():
        if len(touching_faces) == 2:
            f0, f1 = touching_faces
            adjacency[f0].add(f1)
            adjacency[f1].add(f0)

    return [sorted(neighbors) for neighbors in adjacency]


def choose_partition_seeds(face_centroids: np.ndarray, num_charts: int) -> List[int]:
    """Choose deterministic farthest-point seeds in face-centroid space."""
    if num_charts < 1:
        raise ValueError("num_charts must be >= 1")
    if num_charts > len(face_centroids):
        raise ValueError("num_charts cannot exceed number of faces")

    mesh_center = face_centroids.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(face_centroids - mesh_center, axis=1)))
    seeds = [first]
    min_dist2 = np.sum((face_centroids - face_centroids[first]) ** 2, axis=1)

    for _ in range(1, num_charts):
        next_seed = int(np.argmax(min_dist2))
        seeds.append(next_seed)
        dist2 = np.sum((face_centroids - face_centroids[next_seed]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

    return seeds


def partition_faces_connected(vertices: np.ndarray, faces: np.ndarray, num_charts: int) -> np.ndarray:
    """Partition a connected mesh into a fixed number of connected face regions."""
    centroids = vertices[faces].mean(axis=1)
    adjacency = build_face_adjacency(faces)
    seeds = choose_partition_seeds(centroids, num_charts)

    chart_ids = np.full(len(faces), -1, dtype=np.int32)
    queues: List[deque[int]] = []
    for chart_id, seed in enumerate(seeds):
        chart_ids[seed] = chart_id
        queues.append(deque([seed]))

    remaining = len(faces) - len(seeds)
    active = True
    while remaining > 0 and active:
        active = False
        for chart_id, queue in enumerate(queues):
            if not queue:
                continue
            active = True
            face_idx = queue.popleft()
            for neighbor in adjacency[face_idx]:
                if chart_ids[neighbor] == -1:
                    chart_ids[neighbor] = chart_id
                    queue.append(neighbor)
                    remaining -= 1

    if remaining > 0:
        # Handles rare non-manifold disconnected leftovers by assigning each
        # leftover face to the nearest seed in Euclidean centroid space.
        seed_centroids = centroids[seeds]
        leftover = np.flatnonzero(chart_ids == -1)
        dist2 = ((centroids[leftover, None, :] - seed_centroids[None, :, :]) ** 2).sum(axis=2)
        chart_ids[leftover] = np.argmin(dist2, axis=1).astype(np.int32)

    return chart_ids


def atlas_shape(num_charts: int) -> Tuple[int, int]:
    """Choose a compact atlas grid."""
    cols = int(np.ceil(np.sqrt(num_charts * 1.35)))
    rows = int(np.ceil(num_charts / cols))
    return cols, rows


def tile_for_chart(chart_id: int, atlas_cols: int, atlas_rows: int) -> Tuple[int, int]:
    """Return bottom-left-origin tile coordinates."""
    return chart_id % atlas_cols, chart_id // atlas_cols


def project_vertices_pca(vertices: np.ndarray, used_vertices: np.ndarray) -> np.ndarray:
    """Project vertices to the first two PCA axes of a chart."""
    chart_vertices = vertices[used_vertices]
    center = chart_vertices.mean(axis=0)
    centered = chart_vertices - center

    # Vt contains orthonormal principal axes sorted by variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T

    projected = (vertices - center) @ basis
    return projected.astype(np.float32)


def normalize_to_tile(
    projected: np.ndarray,
    tile_xy: Tuple[int, int],
    atlas_cols: int,
    atlas_rows: int,
    padding: float,
) -> np.ndarray:
    """Normalize projected 2D coordinates into one atlas tile."""
    mn = projected.min(axis=0)
    mx = projected.max(axis=0)
    span = np.maximum(mx - mn, 1e-8)
    local = (projected - mn) / span

    # Keep aspect ratio inside the tile to avoid extreme atlas distortion.
    aspect = span[0] / span[1]
    if aspect > 1.0:
        local[:, 1] = (local[:, 1] - 0.5) / aspect + 0.5
    else:
        local[:, 0] = (local[:, 0] - 0.5) * aspect + 0.5

    local = np.clip(local, 0.0, 1.0)
    local = padding + local * (1.0 - 2.0 * padding)

    tile_w = 1.0 / atlas_cols
    tile_h = 1.0 / atlas_rows
    tile_x, tile_y = tile_xy

    out = np.empty_like(local, dtype=np.float32)
    out[:, 0] = (tile_x + local[:, 0]) * tile_w
    out[:, 1] = (tile_y + local[:, 1]) * tile_h
    return out


def build_connected_chart_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_charts: int,
    padding: float,
) -> GeneratedAsset:
    """Create face-corner UVs with connected chart partitions."""
    face_chart_ids = partition_faces_connected(vertices, faces, num_charts)
    atlas_cols, atlas_rows = atlas_shape(num_charts)

    per_chart_vertex_uvs: Dict[int, np.ndarray] = {}
    chart_info: List[Dict] = []

    for chart_id in range(num_charts):
        face_mask = face_chart_ids == chart_id
        if not face_mask.any():
            raise RuntimeError(f"Chart {chart_id} has no faces")

        used_vertices = np.unique(faces[face_mask].reshape(-1))
        tile_xy = tile_for_chart(chart_id, atlas_cols, atlas_rows)
        projected_all = project_vertices_pca(vertices, used_vertices)
        projected_used = projected_all[used_vertices]
        mn = projected_used.min(axis=0)
        mx = projected_used.max(axis=0)
        projected_norm = normalize_to_tile(projected_all, tile_xy, atlas_cols, atlas_rows, padding)

        per_chart_vertex_uvs[chart_id] = projected_norm
        chart_info.append({
            "id": chart_id,
            "name": f"chart_{chart_id:02d}",
            "tile": [int(tile_xy[0]), int(tile_xy[1])],
            "num_faces": int(face_mask.sum()),
            "num_vertices": int(len(used_vertices)),
            "projected_min": mn.astype(float).tolist(),
            "projected_max": mx.astype(float).tolist(),
        })

    uvs: List[np.ndarray] = []
    face_uv_indices = np.empty_like(faces, dtype=np.int32)

    # One vt per face corner. Coordinates remain continuous inside a chart,
    # while seams are explicit at chart boundaries.
    vt_index = 0
    for face_idx, face in enumerate(faces):
        chart_id = int(face_chart_ids[face_idx])
        chart_uvs = per_chart_vertex_uvs[chart_id]
        for corner_idx, vertex_idx in enumerate(face):
            uvs.append(chart_uvs[int(vertex_idx)])
            face_uv_indices[face_idx, corner_idx] = vt_index
            vt_index += 1

    return GeneratedAsset(
        vertices=vertices,
        faces=faces,
        uvs=np.asarray(uvs, dtype=np.float32),
        face_uv_indices=face_uv_indices,
        face_chart_ids=face_chart_ids,
        chart_info=chart_info,
    )


def chart_color(chart_id: int) -> Tuple[int, int, int]:
    if chart_id < len(BASE_CHART_COLORS):
        return BASE_CHART_COLORS[chart_id]

    # Deterministic fallback color for larger experiments.
    hue = (chart_id * 0.61803398875) % 1.0
    angle = hue * np.pi * 2.0
    return (
        int(145 + 85 * np.sin(angle)),
        int(145 + 85 * np.sin(angle + 2.1)),
        int(145 + 85 * np.sin(angle + 4.2)),
    )


def make_procedural_texture(
    width: int,
    height: int,
    num_charts: int,
    atlas_cols: int,
    atlas_rows: int,
    chart_info: Sequence[Dict],
) -> Image.Image:
    """Create a high-frequency atlas texture with per-chart identity cues."""
    tex = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(tex)

    tile_w = width // atlas_cols
    tile_h = height // atlas_rows

    for chart_id in range(num_charts):
        name = chart_info[chart_id]["name"]
        tile_xy = tile_for_chart(chart_id, atlas_cols, atlas_rows)
        tx, ty = tile_xy
        x0 = tx * tile_w
        y0 = (atlas_rows - 1 - ty) * tile_h
        x1 = width if tx == atlas_cols - 1 else (tx + 1) * tile_w
        y1 = height if ty == 0 else (atlas_rows - ty) * tile_h

        base = np.array(chart_color(chart_id), dtype=np.float32)
        tw = x1 - x0
        th = y1 - y0

        tile = np.zeros((th, tw, 3), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, tw, endpoint=False, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, th, endpoint=False, dtype=np.float32)
        u, v = np.meshgrid(xs, ys)

        checker = ((np.floor(u * 14) + np.floor(v * 14)) % 2).astype(np.float32)
        fine = 0.5 + 0.5 * np.sin((u * 41.0 + v * 17.0 + chart_id * 0.37) * np.pi * 2.0)
        ring = np.sin(np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 70.0)

        color = base[None, None, :] * (0.55 + 0.25 * checker[..., None])
        color += np.array([42, 42, 42], dtype=np.float32)[None, None, :] * fine[..., None]
        color += np.array([30, 30, 30], dtype=np.float32)[None, None, :] * ring[..., None]

        # Directional ramps make mirrored/flipped UVs visible.
        color[..., 0] += 70.0 * u
        color[..., 1] += 70.0 * v
        color[..., 2] += 40.0 * (1.0 - u)

        tile_img = Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), mode="RGB")
        tex.paste(tile_img, (x0, y0))

        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 255), width=max(2, width // 512))
        draw.line([x0, y0, x1, y1], fill=(255, 255, 255), width=max(1, width // 768))
        draw.line([x0, y1, x1, y0], fill=(0, 0, 0), width=max(1, width // 768))
        draw.text((x0 + 8, y0 + 8), f"C{chart_id} {name}", fill=(255, 255, 255))

    return tex


def write_mtl(mtl_path: Path, texture_path: Path, obj_path: Path) -> None:
    rel_texture = Path(os.path.relpath(texture_path.resolve(), obj_path.parent.resolve()))
    content = "\n".join([
        "# Procedural Stanford bunny material",
        "newmtl bunny_procedural_material",
        "Ka 1.0 1.0 1.0",
        "Kd 1.0 1.0 1.0",
        "Ks 0.0 0.0 0.0",
        "Ns 10.0",
        "d 1.0",
        f"map_Kd {rel_texture.as_posix()}",
        "",
    ])
    mtl_path.write_text(content)


def write_obj(obj_path: Path, mtl_path: Path, asset: GeneratedAsset) -> None:
    with obj_path.open("w") as f:
        f.write("# Stanford bunny with procedural multi-chart face-corner UV atlas\n")
        f.write("# Generated by scripts/generate_procedural_bunny_texture.py\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl bunny_procedural_material\n")

        for v in asset.vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        for uv in asset.uvs:
            f.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")

        for face, face_vt in zip(asset.faces, asset.face_uv_indices):
            f.write(
                "f"
                f" {int(face[0]) + 1}/{int(face_vt[0]) + 1}"
                f" {int(face[1]) + 1}/{int(face_vt[1]) + 1}"
                f" {int(face[2]) + 1}/{int(face_vt[2]) + 1}\n"
            )


def write_uv_debug_image(path: Path, texture: Image.Image, asset: GeneratedAsset, max_faces: int = 12000) -> None:
    debug = texture.copy()
    draw = ImageDraw.Draw(debug)
    width, height = debug.size
    rng = np.random.default_rng(1234)
    face_indices = np.arange(len(asset.faces))
    if len(face_indices) > max_faces:
        face_indices = rng.choice(face_indices, size=max_faces, replace=False)

    for face_idx in face_indices:
        uv = asset.uvs[asset.face_uv_indices[face_idx]]
        pts = [(float(p[0]) * (width - 1), (1.0 - float(p[1])) * (height - 1)) for p in uv]
        draw.line([pts[0], pts[1], pts[2], pts[0]], fill=(255, 255, 255), width=1)

    debug.save(path)


def validate_outputs(obj_path: Path) -> Dict:
    from src.data.obj_parser import verify_face_corner_uvs
    from src.data.uv_chart_segmentation import compute_uv_charts
    import torch

    result = verify_face_corner_uvs(str(obj_path))
    data = result["data"]
    face_chart_id, chart_info = compute_uv_charts(
        torch.from_numpy(data["face_vertex_indices"]),
        torch.from_numpy(data["face_uv_indices"]),
        torch.from_numpy(data["face_uvs"]),
    )

    return {
        "obj_stats": result["stats"],
        "chart_stats": {
            "num_charts": int(chart_info["num_charts"]),
            "chart_sizes": [int(x) for x in chart_info["chart_sizes"]],
            "num_uv_seams": int(len(chart_info["uv_seams"])),
        },
        "face_chart_ids_min": int(face_chart_id.min().item()),
        "face_chart_ids_max": int(face_chart_id.max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a procedural multi-chart textured bunny OBJ for MA-IUVF training."
    )
    parser.add_argument("--input-mesh", default="data/models/stanford-bunny.obj")
    parser.add_argument("--output-obj", default="data/models/stanford_bunny_procedural.obj")
    parser.add_argument("--texture", default="data/textures/stanford_bunny_procedural_texture.png")
    parser.add_argument("--metadata", default="data/textures/stanford_bunny_procedural_metadata.json")
    parser.add_argument("--uv-debug", default="data/textures/stanford_bunny_procedural_uv_debug.png")
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--num-charts", type=int, default=8)
    parser.add_argument("--tile-padding", type=float, default=0.055)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_mesh = Path(args.input_mesh)
    output_obj = Path(args.output_obj)
    texture_path = Path(args.texture)
    metadata_path = Path(args.metadata)
    uv_debug_path = Path(args.uv_debug)
    output_mtl = output_obj.with_suffix(".mtl")

    if not input_mesh.exists():
        raise FileNotFoundError(input_mesh)

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    uv_debug_path.parent.mkdir(parents=True, exist_ok=True)

    vertices, faces = load_mesh(input_mesh)
    asset = build_connected_chart_atlas(
        vertices,
        faces,
        num_charts=args.num_charts,
        padding=args.tile_padding,
    )
    atlas_cols, atlas_rows = atlas_shape(args.num_charts)
    texture = make_procedural_texture(
        args.texture_size,
        args.texture_size,
        num_charts=args.num_charts,
        atlas_cols=atlas_cols,
        atlas_rows=atlas_rows,
        chart_info=asset.chart_info,
    )

    texture.save(texture_path)
    write_mtl(output_mtl, texture_path, output_obj)
    write_obj(output_obj, output_mtl, asset)
    write_uv_debug_image(uv_debug_path, texture, asset)

    validation = None
    if not args.skip_validation:
        validation = validate_outputs(output_obj)

    metadata = {
        "input_mesh": str(input_mesh),
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "texture": str(texture_path),
        "uv_debug": str(uv_debug_path),
        "texture_size": int(args.texture_size),
        "tile_padding": float(args.tile_padding),
        "uv_convention": "bottom_left_origin",
        "atlas": {
            "type": "connected_face_partition_pca",
            "num_charts_requested": int(args.num_charts),
            "cols": int(atlas_cols),
            "rows": int(atlas_rows),
            "charts": asset.chart_info,
        },
        "mesh": {
            "num_vertices": int(len(asset.vertices)),
            "num_faces": int(len(asset.faces)),
            "num_uvs": int(len(asset.uvs)),
        },
        "validation": validation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Generated procedural bunny training asset:")
    print(f"  OBJ:      {output_obj}")
    print(f"  MTL:      {output_mtl}")
    print(f"  Texture:  {texture_path}")
    print(f"  UV debug: {uv_debug_path}")
    print(f"  Metadata: {metadata_path}")
    if validation:
        print("Validation:")
        print(f"  OBJ vertices/faces/uvs: {validation['obj_stats']['num_vertices']} / "
              f"{validation['obj_stats']['num_faces']} / {validation['obj_stats']['num_uvs']}")
        print(f"  UV seam vertices: {validation['obj_stats']['num_uv_seam_vertices']}")
        print(f"  Charts: {validation['chart_stats']['num_charts']}")
        print(f"  Chart sizes: {validation['chart_stats']['chart_sizes']}")
        print(f"  UV seams: {validation['chart_stats']['num_uv_seams']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
