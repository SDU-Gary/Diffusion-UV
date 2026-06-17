"""
Low-poly MA-IUVF inference test pipeline.

Pipeline:
1. prepare a low-poly mesh by loading an existing low mesh or simplifying the
   input high mesh;
2. query the MA-IUVF checkpoint at each low-mesh face corner and export a
   textured OBJ;
3. rasterize the low mesh, query MA-IUVF per visible pixel, sample the texture,
   and save render/debug buffers.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.mesh_simplification import MeshSimplifier
from src.inference.metric_aligned_iuv_inference import MetricAlignedIUVInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh, concatenating scene geometry when needed."""
    # Keep this aligned with scripts/inference.py and MeshSimplifier. For OBJ
    # files with independent face-corner UVs, process=True lets trimesh expose
    # the merged geometry instead of counting texture-expanded corners as
    # distinct geometry vertices.
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        try:
            merged = loaded.dump(concatenate=True)
            if isinstance(merged, trimesh.Trimesh):
                return merged
        except Exception:
            pass

        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        return trimesh.util.concatenate(meshes)

    raise TypeError(f"Unsupported mesh object from {path}: {type(loaded)!r}")


def mesh_stats(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Return JSON-serializable mesh statistics."""
    return {
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "bbox_min": np.asarray(mesh.bounds[0], dtype=np.float64).tolist(),
        "bbox_max": np.asarray(mesh.bounds[1], dtype=np.float64).tolist(),
    }


def prepare_low_mesh(
    input_mesh_path: str,
    low_mesh_path: Optional[str],
    target_faces: Optional[int],
    face_ratio: Optional[float],
    simplify: bool,
) -> Tuple[trimesh.Trimesh, Dict]:
    """Prepare low-poly mesh using the old implicit-color-field simplification style."""
    high_mesh = load_mesh(input_mesh_path)
    high_stats = mesh_stats(high_mesh)

    if low_mesh_path:
        logger.info("Using provided low mesh: %s", low_mesh_path)
        low_mesh = load_mesh(low_mesh_path)
        return low_mesh, {
            "mode": "provided_low_mesh",
            "input_mesh_stats": high_stats,
            "low_mesh_stats": mesh_stats(low_mesh),
            "low_mesh_path": str(low_mesh_path),
        }

    if not simplify:
        logger.info("Skipping simplification; using input mesh directly.")
        return high_mesh, {
            "mode": "no_simplify",
            "input_mesh_stats": high_stats,
            "low_mesh_stats": high_stats,
            "low_mesh_path": None,
        }

    if target_faces is not None:
        target = int(target_faces)
        target_desc = f"{target} faces"
    elif face_ratio is not None:
        target = int(len(high_mesh.faces) * float(face_ratio))
        target_desc = f"{face_ratio:.4f} ratio -> {target} faces"
    else:
        target = int(len(high_mesh.faces) * 0.05)
        target_desc = f"default 0.05 ratio -> {target} faces"

    if target < 4:
        raise ValueError(f"target face count is too small: {target}")

    logger.info("Simplifying mesh: %s", target_desc)

    # Keep the same conservative path used by scripts/inference.py: export the
    # loaded high mesh into a temporary OBJ, then let MeshSimplifier operate on
    # that geometry-only file.
    temp_dir = Path(tempfile.mkdtemp(prefix="maiuvf_lowpoly_"))
    temp_mesh_path = temp_dir / "high_mesh.obj"
    try:
        high_mesh.export(str(temp_mesh_path))
        simplifier = MeshSimplifier(str(temp_mesh_path))
        start = time.time()
        low_mesh = simplifier.simplify_by_count(target, method="quadric", aggression=10)
        elapsed = time.time() - start
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    low_stats = mesh_stats(low_mesh)
    actual_ratio = low_stats["num_faces"] / max(1, high_stats["num_faces"])
    return low_mesh, {
        "mode": "simplified",
        "target_faces": int(target),
        "target_face_ratio": float(face_ratio) if face_ratio is not None else None,
        "input_mesh_stats": high_stats,
        "low_mesh_stats": low_stats,
        "actual_face_ratio": float(actual_ratio),
        "compression_percent": float((1.0 - actual_ratio) * 100.0),
        "elapsed_seconds": float(elapsed),
        "low_mesh_path": None,
    }


def predict_corner_uvs(
    inference: MetricAlignedIUVInference,
    mesh: trimesh.Trimesh,
    batch_size: int,
    uv_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Predict one UV coordinate per low-mesh face corner."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)  # [V, 3]
    faces = np.asarray(mesh.faces, dtype=np.int64)          # [F, 3]
    corner_positions = vertices[faces].reshape(-1, 3)       # [F*3, 3]

    logger.info("Predicting low-mesh corner UVs: %d corners", len(corner_positions))
    logits, uv_preds = inference.predict(corner_positions, batch_size=batch_size)
    selected_uvs, chart_ids = inference.select_uvs(logits, uv_preds, mode=uv_mode)

    num_charts = int(inference.metadata["num_charts"])
    chart_distribution = np.bincount(chart_ids, minlength=num_charts)
    return selected_uvs.astype(np.float32), chart_ids.astype(np.int64), logits.astype(np.float32), {
        "num_corners": int(len(corner_positions)),
        "uv_mode": uv_mode,
        "num_charts": num_charts,
        "corner_chart_distribution": chart_distribution.astype(int).tolist(),
    }


def export_textured_obj(
    mesh: trimesh.Trimesh,
    corner_uvs: np.ndarray,
    texture_path: Optional[str],
    output_obj_path: Path,
    copy_texture: bool,
) -> Dict[str, Optional[str]]:
    """Export a low mesh OBJ where each face corner owns its own vt index."""
    output_obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = output_obj_path.with_suffix(".mtl")

    texture_rel = None
    copied_texture_path = None
    if texture_path:
        texture_abs = Path(texture_path).resolve()
        if copy_texture:
            copied = output_obj_path.parent / texture_abs.name
            if texture_abs != copied.resolve():
                shutil.copy2(texture_abs, copied)
            texture_rel = copied.name
            copied_texture_path = str(copied)
        else:
            try:
                texture_rel = os.path.relpath(texture_abs, output_obj_path.parent.resolve())
            except (OSError, ValueError):
                texture_rel = str(texture_abs)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(corner_uvs) != len(faces) * 3:
        raise ValueError(f"corner_uvs shape mismatch: got {len(corner_uvs)}, expected {len(faces) * 3}")

    with open(output_obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for uv in corner_uvs:
            f.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        f.write("usemtl material_0\n")
        vt_idx = 1
        for face in faces:
            f.write(
                f"f {face[0] + 1}/{vt_idx} "
                f"{face[1] + 1}/{vt_idx + 1} "
                f"{face[2] + 1}/{vt_idx + 2}\n"
            )
            vt_idx += 3

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        if texture_rel:
            f.write(f"map_Kd {texture_rel}\n")

    return {
        "obj": str(output_obj_path),
        "mtl": str(mtl_path),
        "copied_texture": copied_texture_path,
        "mtl_map_kd": texture_rel,
    }


def save_predictions_npz(
    output_path: Path,
    mesh: trimesh.Trimesh,
    corner_uvs: np.ndarray,
    chart_ids: np.ndarray,
    logits: np.ndarray,
) -> str:
    """Save corner-level MA-IUVF predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    corner_positions = vertices[faces].reshape(-1, 3)
    np.savez_compressed(
        output_path,
        corner_positions=corner_positions,
        corner_uvs=corner_uvs,
        chart_ids=chart_ids,
        logits=logits,
        faces=faces,
        vertices=vertices,
    )
    return str(output_path)


def render_low_mesh(
    inference: MetricAlignedIUVInference,
    mesh: trimesh.Trimesh,
    texture_path: str,
    output_dir: Path,
    resolution: Tuple[int, int],
    raster_backend: str,
    save_prediction_buffers: bool,
    view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict:
    """Rasterize the low mesh and render it by per-pixel MA-IUVF texture sampling."""
    import src.inference.offline_renderer as offline_renderer_module
    from src.inference.offline_renderer import OfflineRenderer

    if raster_backend == "cpu":
        offline_renderer_module.OPENGL_AVAILABLE = False
    elif raster_backend == "opengl" and not offline_renderer_module.OPENGL_AVAILABLE:
        raise RuntimeError("OpenGL raster backend requested, but OpenGL renderer is not available")

    texture = np.asarray(Image.open(texture_path).convert("RGB"))
    renderer = OfflineRenderer(
        mesh_vertices=np.asarray(mesh.vertices, dtype=np.float32),
        mesh_faces=np.asarray(mesh.faces, dtype=np.int64),
        texture_image=texture,
        resolution=resolution,
        view_bounds=view_bounds,
    )

    baker_metadata = inference.metadata.get("baker_metadata")
    if baker_metadata is not None and "face_chart_id" in baker_metadata:
        if len(baker_metadata["face_chart_id"]) != len(mesh.faces):
            # A simplified low mesh no longer has high-mesh face ids; passing
            # high-mesh chart metadata would make chart-accuracy statistics
            # meaningless.
            baker_metadata = None

    image, render_info = renderer.render_with_maiuvf(
        model=inference.model,
        device=inference.device,
        baker_metadata=baker_metadata,
    )

    render_path = output_dir / "render.png"
    renderer.save_render(image, str(render_path))
    render_info["outputs"] = {**render_info.get("outputs", {}), "render": str(render_path)}
    render_info["raster_backend"] = raster_backend
    render_info["resolution"] = [int(resolution[0]), int(resolution[1])]

    if save_prediction_buffers:
        buffer_paths = renderer.save_prediction_buffers(str(output_dir), prefix="pred")
        render_info["outputs"].update(buffer_paths)

    info_path = output_dir / "render_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(render_info, f, indent=2)
    render_info["outputs"]["render_info"] = str(info_path)
    return render_info


def render_high_reference(
    input_mesh_path: str,
    texture_path: str,
    output_dir: Path,
    resolution: Tuple[int, int],
    backend: str,
    sampling: str,
    view_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """Render the original high-poly UV reference with the Phase 1 renderer."""
    from scripts.render_high_uv_reference import render_reference

    reference_dir = output_dir / "reference"
    return render_reference(
        mesh_path=input_mesh_path,
        texture_path=texture_path,
        output_dir=str(reference_dir),
        resolution=resolution,
        backend=backend,
        sampling=sampling,
        baked_data=None,
        compute_charts_flag=False,
        prefix="high_reference",
        view_bounds=view_bounds,
    )


def make_high_low_compare(
    high_image_path: str,
    low_image_path: str,
    output_path: Path,
    low_faces: int,
) -> str:
    """Create a labeled high/low side-by-side comparison image, without Err map."""
    high = Image.open(high_image_path).convert("RGB")
    low = Image.open(low_image_path).convert("RGB")
    if low.size != high.size:
        low = low.resize(high.size, Image.BILINEAR)

    width, height = high.size
    label_h = 36
    canvas = Image.new("RGB", (width * 2, height + label_h), (20, 20, 20))
    canvas.paste(high, (0, label_h))
    canvas.paste(low, (width, label_h))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "High-poly UV reference", fill=(240, 240, 240))
    draw.text((width + 12, 10), f"Low-poly MA-IUVF prediction ({low_faces} faces)", fill=(240, 240, 240))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MA-IUVF low-poly inference test: simplify -> predict -> export OBJ -> render.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/infer_lowpoly_metric_aligned_iuv.py \\
      --checkpoint outputs/maiuvf_phase1/bspline_hash_dynamic_anchor1_metric0p01/run_000_samples300000_sigma0p01_epochs100/train/best.pt \\
      --input-mesh data/models/stanford_bunny_procedural.obj \\
      --texture data/textures/stanford_bunny_procedural_texture.png \\
      --output-dir outputs/maiuvf_lowpoly_tests/bunny_5pct \\
      --face-ratio 0.05 \\
      --render-resolution 512
        """,
    )
    parser.add_argument("--checkpoint", required=True, help="MA-IUVF checkpoint path")
    parser.add_argument("--input-mesh", required=True, help="High mesh path used as simplification source")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--texture", help="Texture path; falls back to checkpoint metadata")
    parser.add_argument("--low-mesh", help="Use an existing low mesh instead of simplifying input mesh")

    simplify_group = parser.add_mutually_exclusive_group()
    simplify_group.add_argument("--target-faces", type=int, help="Target low-mesh face count")
    simplify_group.add_argument("--face-ratio", type=float, help="Low-mesh face ratio, e.g. 0.05")
    simplify_group.add_argument("--no-simplify", dest="simplify", action="store_false", help="Use input mesh directly")
    parser.set_defaults(simplify=True)

    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--batch-size", type=int, default=8192, help="MA-IUVF prediction batch size")
    parser.add_argument("--uv-mode", default="argmax", choices=["argmax", "sample"], help="Chart/UV selection mode")
    parser.add_argument("--render-resolution", type=int, default=512, help="Square render resolution")
    parser.add_argument("--render-width", type=int, help="Render width; overrides --render-resolution when paired with --render-height")
    parser.add_argument("--render-height", type=int, help="Render height; overrides --render-resolution when paired with --render-width")
    parser.add_argument(
        "--raster-backend",
        default="cpu",
        choices=["cpu", "auto", "opengl"],
        help="Rasterizer backend for render.png; CPU is the stable default",
    )
    parser.add_argument("--skip-render", action="store_true", help="Only export low mesh and textured OBJ")
    parser.add_argument("--skip-reference", action="store_true", help="Do not render high-poly UV reference or high/low comparison")
    parser.add_argument(
        "--reference-backend",
        default="same",
        choices=["same", "cpu", "auto", "opengl"],
        help="Backend for high-poly UV reference render; 'same' uses --raster-backend",
    )
    parser.add_argument("--reference-sampling", default="bilinear", choices=["bilinear", "nearest"])
    parser.add_argument("--no-copy-texture", dest="copy_texture", action="store_false", help="Do not copy texture beside OBJ")
    parser.set_defaults(copy_texture=True)
    parser.add_argument("--no-export-npz", dest="export_npz", action="store_false", help="Do not save corner prediction NPZ")
    parser.set_defaults(export_npz=True)
    parser.add_argument(
        "--no-save-prediction-buffers",
        dest="save_prediction_buffers",
        action="store_false",
        help="Do not save per-pixel pred_uv/pred_chart_id/pred_face_id buffers",
    )
    parser.set_defaults(save_prediction_buffers=True)

    args = parser.parse_args()
    if (args.render_width is None) != (args.render_height is None):
        parser.error("--render-width and --render-height must be provided together")
    render_resolution = (
        (int(args.render_width), int(args.render_height))
        if args.render_width is not None
        else (int(args.render_resolution), int(args.render_resolution))
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()
    inference = MetricAlignedIUVInference(
        checkpoint_path=args.checkpoint,
        texture_path=args.texture,
        device=args.device,
    )
    texture_path = args.texture or inference.texture_path

    low_mesh, simplification_info = prepare_low_mesh(
        input_mesh_path=args.input_mesh,
        low_mesh_path=args.low_mesh,
        target_faces=args.target_faces,
        face_ratio=args.face_ratio,
        simplify=args.simplify,
    )
    view_bounds = (
        np.asarray(simplification_info["input_mesh_stats"]["bbox_min"], dtype=np.float32),
        np.asarray(simplification_info["input_mesh_stats"]["bbox_max"], dtype=np.float32),
    )

    low_mesh_path = output_dir / "low_mesh.obj"
    low_mesh.export(str(low_mesh_path))
    simplification_info["low_mesh_path"] = str(low_mesh_path)
    logger.info("Saved low mesh: %s", low_mesh_path)

    corner_uvs, chart_ids, logits, prediction_info = predict_corner_uvs(
        inference=inference,
        mesh=low_mesh,
        batch_size=args.batch_size,
        uv_mode=args.uv_mode,
    )

    textured_obj_outputs = export_textured_obj(
        mesh=low_mesh,
        corner_uvs=corner_uvs,
        texture_path=texture_path,
        output_obj_path=output_dir / "low_maiuvf_textured.obj",
        copy_texture=args.copy_texture,
    )
    logger.info("Saved textured OBJ: %s", textured_obj_outputs["obj"])

    npz_path = None
    if args.export_npz:
        npz_path = save_predictions_npz(
            output_path=output_dir / "low_maiuvf_predictions.npz",
            mesh=low_mesh,
            corner_uvs=corner_uvs,
            chart_ids=chart_ids,
            logits=logits,
        )
        logger.info("Saved prediction NPZ: %s", npz_path)

    render_info = None
    reference_info = None
    compare_path = None
    if not args.skip_render:
        if not texture_path:
            raise ValueError("Rendering requires --texture or texture_path in checkpoint metadata")
        render_info = render_low_mesh(
            inference=inference,
            mesh=low_mesh,
            texture_path=texture_path,
            output_dir=output_dir,
            resolution=render_resolution,
            raster_backend=args.raster_backend,
            save_prediction_buffers=args.save_prediction_buffers,
            view_bounds=view_bounds,
        )
        if not args.skip_reference:
            reference_backend = args.raster_backend if args.reference_backend == "same" else args.reference_backend
            reference_info = render_high_reference(
                input_mesh_path=args.input_mesh,
                texture_path=texture_path,
                output_dir=output_dir,
                resolution=render_resolution,
                backend=reference_backend,
                sampling=args.reference_sampling,
                view_bounds=view_bounds,
            )
            # Also copy the single high reference image to the top-level output
            # directory for easier inspection.
            high_ref_src = Path(reference_info["outputs"]["image"])
            high_ref_top = output_dir / "high_reference.png"
            shutil.copy2(high_ref_src, high_ref_top)
            compare_path = make_high_low_compare(
                high_image_path=str(high_ref_top),
                low_image_path=render_info["outputs"]["render"],
                output_path=output_dir / "high_low_compare.png",
                low_faces=len(low_mesh.faces),
            )

    metadata = {
        "checkpoint_path": str(args.checkpoint),
        "input_mesh_path": str(args.input_mesh),
        "provided_low_mesh_path": str(args.low_mesh) if args.low_mesh else None,
        "texture_path": str(texture_path) if texture_path else None,
        "device": str(inference.device),
        "batch_size": int(args.batch_size),
        "simplification": simplification_info,
        "prediction": prediction_info,
        "render": render_info,
        "reference": reference_info,
        "outputs": {
            **textured_obj_outputs,
            "low_mesh": str(low_mesh_path),
            "predictions_npz": npz_path,
            "high_reference": str(output_dir / "high_reference.png") if reference_info is not None else None,
            "high_low_compare": compare_path,
            "metadata": str(output_dir / "metadata.json"),
        },
        "elapsed_seconds": float(time.time() - start_total),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Low-poly MA-IUVF inference complete.")
    logger.info("Output directory: %s", output_dir)
    logger.info("Textured OBJ: %s", textured_obj_outputs["obj"])
    if render_info is not None:
        logger.info("Render: %s", render_info["outputs"].get("render"))
        logger.info("Coverage: %.2f%%", 100.0 * render_info.get("coverage", 0.0))
    if compare_path is not None:
        logger.info("High/low compare: %s", compare_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
