"""
Render a high-poly OBJ with its original face-corner UVs.

This script produces the screen-space reference image used by MA-IUVF
Phase 1 experiments. It does not query any neural field and does not
generate a new texture. It simply rasterizes the high-poly mesh, interpolates
the face-corner UVs, and samples the original texture.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.obj_parser import parse_obj_file
from src.data.uv_chart_segmentation import compute_uv_charts
from src.inference.offline_renderer import CPUTexturizer

logger = logging.getLogger(__name__)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    return value


def load_face_chart_id(
    obj_data: Dict[str, np.ndarray],
    baked_data_path: Optional[str],
    compute_charts: bool,
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Load or compute per-face chart IDs."""
    if baked_data_path:
        baked = torch.load(baked_data_path, map_location="cpu", weights_only=False)
        metadata = baked.get("metadata", {})
        face_chart_id = metadata.get("face_chart_id")
        if face_chart_id is not None:
            face_chart_id = np.asarray(face_chart_id, dtype=np.int32)
            if len(face_chart_id) == len(obj_data["faces"]):
                return face_chart_id, metadata.get("chart_stats")
            logger.warning(
                "Ignoring baked face_chart_id because length %d != num_faces %d",
                len(face_chart_id),
                len(obj_data["faces"]),
            )

    if not compute_charts:
        return None, None

    logger.info("Computing UV charts for reference metadata")
    face_chart_id, chart_info = compute_uv_charts(
        torch.from_numpy(obj_data["face_vertex_indices"]).long(),
        torch.from_numpy(obj_data["face_uv_indices"]).long(),
        torch.from_numpy(obj_data["face_uvs"]).float(),
    )
    return face_chart_id.cpu().numpy().astype(np.int32), chart_info


def rasterize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: Tuple[int, int],
    backend: str,
    view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Rasterize and return world positions, face IDs, barycentric coords."""
    if backend in {"opengl", "auto"}:
        try:
            from src.inference.opengl_renderer import render_with_opengl_gbuffer

            logger.info("Rasterizing reference with OpenGL G-buffer")
            world_pos, face_ids, bary = render_with_opengl_gbuffer(
                vertices, faces, resolution, view_bounds=view_bounds
            )
            return world_pos, face_ids, bary, "opengl"
        except Exception as exc:
            if backend == "opengl":
                raise
            logger.warning("OpenGL rasterization failed, falling back to CPU: %s", exc)

    logger.info("Rasterizing reference with CPU rasterizer")
    rasterizer = CPUTexturizer(vertices, faces, resolution, view_bounds=view_bounds)
    world_pos, face_ids, bary = rasterizer.rasterize()
    return world_pos, face_ids, bary, "cpu"


def bilinear_sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample RGB texture at UV coordinates in bottom-left convention."""
    tex_h, tex_w = texture.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)

    x = u * (tex_w - 1)
    y = (1.0 - v) * (tex_h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tex_w - 1)
    y1 = np.clip(y0 + 1, 0, tex_h - 1)

    wx = (x - x0).reshape(-1, 1)
    wy = (y - y0).reshape(-1, 1)

    c00 = texture[y0, x0].astype(np.float32)
    c10 = texture[y0, x1].astype(np.float32)
    c01 = texture[y1, x0].astype(np.float32)
    c11 = texture[y1, x1].astype(np.float32)

    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    color = c0 * (1.0 - wy) + c1 * wy
    return np.clip(color, 0, 255).astype(np.uint8)


def nearest_sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    tex_h, tex_w = texture.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.rint(u * (tex_w - 1)).astype(np.int32)
    y = np.rint((1.0 - v) * (tex_h - 1)).astype(np.int32)
    return texture[y, x].astype(np.uint8)


def render_reference(
    mesh_path: str,
    texture_path: str,
    output_dir: str,
    resolution,
    backend: str,
    sampling: str,
    baked_data: Optional[str],
    compute_charts_flag: bool,
    prefix: str,
    view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    obj_data = parse_obj_file(mesh_path)
    texture = np.asarray(Image.open(texture_path).convert("RGB"))
    if isinstance(resolution, (tuple, list)):
        width, height = int(resolution[0]), int(resolution[1])
    else:
        width = height = int(resolution)

    world_pos, face_ids, bary, actual_backend = rasterize_mesh(
        obj_data["vertices"], obj_data["faces"], (width, height), backend, view_bounds=view_bounds
    )
    valid_mask = face_ids >= 0
    valid_face_ids = face_ids[valid_mask]
    valid_bary = bary[valid_mask]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    uv_image = np.zeros((height, width, 2), dtype=np.float32)
    chart_image = np.full((height, width), -1, dtype=np.int32)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    if valid_face_ids.size:
        face_uvs = obj_data["face_uvs"][valid_face_ids]  # [N, 3, 2]
        valid_uv = (face_uvs * valid_bary[:, :, None]).sum(axis=1)
        if sampling == "nearest":
            colors = nearest_sample_texture(texture, valid_uv)
        else:
            colors = bilinear_sample_texture(texture, valid_uv)
        image[valid_mask] = colors
        uv_image[valid_mask] = valid_uv
        depth[valid_mask] = world_pos[valid_mask, 2]

    face_chart_id, chart_stats = load_face_chart_id(
        obj_data, baked_data, compute_charts_flag
    )
    chart_distribution = {}
    if face_chart_id is not None and valid_face_ids.size:
        valid_chart_ids = face_chart_id[valid_face_ids]
        chart_image[valid_mask] = valid_chart_ids
        unique, counts = np.unique(valid_chart_ids, return_counts=True)
        chart_distribution = {int(k): int(v) for k, v in zip(unique, counts)}

    image_path = output / f"{prefix}.png"
    mask_path = output / f"{prefix}_mask.png"
    face_id_path = output / f"{prefix}_face_id.npy"
    chart_id_path = output / f"{prefix}_chart_id.npy"
    depth_path = output / f"{prefix}_depth.npy"
    uv_path = output / f"{prefix}_uv.npy"
    info_path = output / f"{prefix}_info.json"

    Image.fromarray(image).save(image_path)
    Image.fromarray((valid_mask.astype(np.uint8) * 255)).save(mask_path)
    np.save(face_id_path, face_ids.astype(np.int32))
    np.save(chart_id_path, chart_image)
    np.save(depth_path, depth)
    np.save(uv_path, uv_image)

    info = {
        "mesh_path": str(mesh_path),
        "texture_path": str(texture_path),
        "resolution": [width, height],
        "requested_backend": backend,
        "actual_backend": actual_backend,
        "sampling": sampling,
        "valid_pixels": int(valid_mask.sum()),
        "total_pixels": int(valid_mask.size),
        "coverage": float(valid_mask.mean()),
        "num_vertices": int(len(obj_data["vertices"])),
        "num_faces": int(len(obj_data["faces"])),
        "num_uvs": int(len(obj_data["uvs"])),
        "chart_distribution": chart_distribution,
        "chart_stats": chart_stats,
        "outputs": {
            "image": str(image_path),
            "mask": str(mask_path),
            "face_id": str(face_id_path),
            "chart_id": str(chart_id_path),
            "depth": str(depth_path),
            "uv": str(uv_path),
        },
    }
    with open(info_path, "w") as f:
        json.dump(_to_jsonable(info), f, indent=2)

    logger.info("Saved reference image: %s", image_path)
    logger.info("Reference coverage: %.2f%%", info["coverage"] * 100.0)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a high-poly UV reference image for MA-IUVF Phase 1."
    )
    parser.add_argument("--mesh", required=True, help="High-poly OBJ with UVs")
    parser.add_argument("--texture", required=True, help="Original texture image")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--width", type=int, help="Render width; overrides --resolution when paired with --height")
    parser.add_argument("--height", type=int, help="Render height; overrides --resolution when paired with --width")
    parser.add_argument("--backend", choices=["cpu", "opengl", "auto"], default="cpu")
    parser.add_argument("--sampling", choices=["bilinear", "nearest"], default="bilinear")
    parser.add_argument("--baked-data", help="Optional baked .pt file with face_chart_id")
    parser.add_argument(
        "--skip-chart-id",
        action="store_true",
        help="Do not compute chart IDs when --baked-data is absent",
    )
    parser.add_argument("--prefix", default="reference")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    resolution = (args.width, args.height) if args.width is not None else args.resolution

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    render_reference(
        mesh_path=args.mesh,
        texture_path=args.texture,
        output_dir=args.output_dir,
        resolution=resolution,
        backend=args.backend,
        sampling=args.sampling,
        baked_data=args.baked_data,
        compute_charts_flag=not args.skip_chart_id,
        prefix=args.prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
