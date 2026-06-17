#!/usr/bin/env python
"""Visualize chart-wise UV translation drift in MA-IUVF predictions.

This diagnostic estimates one robust median UV offset per chart from pixels
whose predicted chart matches the reference chart. It then visualizes:
- chart-level offset vectors,
- per-pixel UV offset direction/magnitude,
- residual error after subtracting chart offsets,
- texture renders compensated by chart offsets.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    return value


def load_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    if path.exists():
        return np.asarray(Image.open(path).convert("L")) > 0
    return np.ones(shape, dtype=bool)


def compute_chart_boundary(chart: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(valid, dtype=bool)
    neighbor_pairs = [
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ]
    for a, b in neighbor_pairs:
        both = valid[a] & valid[b]
        diff = chart[a] != chart[b]
        boundary[a] |= both & diff
    return boundary & valid


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0 = radius + dy
            x0 = radius + dx
            out |= padded[y0 : y0 + h, x0 : x0 + w]
    return out


def delta_to_rgb(delta: np.ndarray, valid: np.ndarray, scale: float) -> np.ndarray:
    """Encode 2D delta as HSV: hue=direction, value=magnitude."""
    rgb = np.zeros((*delta.shape[:2], 3), dtype=np.uint8)
    mag = np.linalg.norm(delta, axis=-1)
    angle = np.arctan2(delta[..., 1], delta[..., 0])
    hue = (angle + math.pi) / (2.0 * math.pi)
    value = np.clip(mag / max(scale, 1e-8), 0.0, 1.0)
    sat = np.ones_like(value)

    flat_hsv = np.stack([hue.reshape(-1), sat.reshape(-1), value.reshape(-1)], axis=-1)
    flat_rgb = np.array([colorsys.hsv_to_rgb(*hsv) for hsv in flat_hsv], dtype=np.float32)
    rgb[:] = (flat_rgb.reshape(*delta.shape[:2], 3) * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def colorize_l2(error: np.ndarray, valid: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    vals = error[valid & np.isfinite(error)]
    scale = float(np.percentile(vals, percentile)) if vals.size else 1.0
    scale = max(scale, 1e-8)
    norm = np.clip(error / scale, 0.0, 1.0)
    cmap = plt.get_cmap("magma")
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def bilinear_sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
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
    return np.clip(c0 * (1.0 - wy) + c1 * wy, 0, 255).astype(np.uint8)


def render_from_uv(texture: np.ndarray, uv_img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = valid.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if valid.any():
        out[valid] = bilinear_sample_texture(texture, uv_img[valid])
    return out


def image_metrics(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> Dict:
    ref = reference.astype(np.float32) / 255.0
    pred = prediction.astype(np.float32) / 255.0
    diff = ref[valid] - pred[valid]
    mse = float(np.mean(diff * diff)) if diff.size else None
    psnr = float(-10.0 * math.log10(max(mse, 1e-12))) if mse is not None else None
    mae = float(np.mean(np.abs(diff))) if diff.size else None
    return {"mse": mse, "psnr": psnr, "mae": mae}


def summarize(vals: np.ndarray) -> Dict:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def draw_chart_offset_map(
    ref_chart: np.ndarray,
    valid: np.ndarray,
    chart_offsets: Dict[int, np.ndarray],
    chart_stats: Dict[int, Dict],
    output_path: Path,
):
    h, w = valid.shape
    offset_values = np.array(list(chart_offsets.values()), dtype=np.float32)
    max_mag = float(np.percentile(np.linalg.norm(offset_values, axis=1), 95)) if len(offset_values) else 1.0
    max_mag = max(max_mag, 1e-8)

    offset_img = np.zeros((h, w, 2), dtype=np.float32)
    for chart_id, offset in chart_offsets.items():
        offset_img[(ref_chart == chart_id) & valid] = offset
    rgb = delta_to_rgb(offset_img, valid, max_mag)
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    arrow_scale = 850.0
    for chart_id, offset in chart_offsets.items():
        mask = (ref_chart == chart_id) & valid
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        dx = float(offset[0] * arrow_scale)
        dy = float(-offset[1] * arrow_scale)
        mag = float(np.linalg.norm(offset))
        color = (255, 255, 255)
        draw.line((cx, cy, cx + dx, cy + dy), fill=color, width=3)
        # Arrow head.
        angle = math.atan2(dy, dx)
        for a in (angle + 2.55, angle - 2.55):
            draw.line(
                (cx + dx, cy + dy, cx + dx - 12 * math.cos(a), cy + dy - 12 * math.sin(a)),
                fill=color,
                width=3,
            )
        label = f"C{chart_id} du={offset[0]:+.4f} dv={offset[1]:+.4f} |d|={mag:.4f}"
        bbox = draw.textbbox((0, 0), label, font=small_font)
        tx = min(max(4, cx + 6), w - (bbox[2] - bbox[0]) - 6)
        ty = min(max(4, cy + 6), h - (bbox[3] - bbox[1]) - 6)
        draw.rectangle((tx - 3, ty - 2, tx + bbox[2] - bbox[0] + 3, ty + bbox[3] - bbox[1] + 2), fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 255, 255), font=small_font)

    title = "Per-chart median UV offset (color+arrow), estimated from non-seam correct pixels"
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.rectangle((6, 6, 12 + bbox[2] - bbox[0], 30), fill=(0, 0, 0))
    draw.text((9, 9), title, fill=(255, 255, 255), font=font)
    image.save(output_path)


def make_compare_strip(
    reference: np.ndarray,
    original: np.ndarray,
    corrected_gt: np.ndarray,
    corrected_pred: np.ndarray,
    output_path: Path,
    metrics: Dict,
):
    panels = [
        ("Reference", reference),
        (f"Original\\nPSNR {metrics['original']['psnr']:.2f}", original),
        (f"GT-chart offset corrected\\nPSNR {metrics['gt_chart_corrected']['psnr']:.2f}", corrected_gt),
        (f"Pred-chart offset corrected\\nPSNR {metrics['pred_chart_corrected']['psnr']:.2f}", corrected_pred),
    ]
    h, w = reference.shape[:2]
    header = 64
    canvas = Image.new("RGB", (w * len(panels), h + header), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for i, (title, arr) in enumerate(panels):
        x = i * w
        canvas.paste(Image.fromarray(arr), (x, header))
        for j, line in enumerate(title.split("\\n")):
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((x + (w - (bbox[2] - bbox[0])) / 2, 10 + j * 24), line, fill=(245, 245, 245), font=font)
        if i > 0:
            draw.line((x, 0, x, h + header), fill=(60, 64, 72), width=2)
    canvas.save(output_path, quality=95)


def analyze_offsets(
    reference_dir: Path,
    prediction_dir: Path,
    texture_path: Path,
    output_dir: Path,
    seam_radius: int,
) -> Dict:
    ref_uv = np.load(reference_dir / "reference_uv.npy").astype(np.float32)
    ref_chart = np.load(reference_dir / "reference_chart_id.npy").astype(np.int32)
    pred_uv = np.load(prediction_dir / "pred_uv.npy").astype(np.float32)
    pred_chart = np.load(prediction_dir / "pred_chart_id.npy").astype(np.int32)
    ref_image = np.asarray(Image.open(reference_dir / "reference.png").convert("RGB"))
    pred_image = np.asarray(Image.open(prediction_dir / "render_cpu.png").convert("RGB"))
    texture = np.asarray(Image.open(texture_path).convert("RGB"))

    h, w = ref_chart.shape
    ref_valid = load_mask(reference_dir / "reference_mask.png", (h, w)) & (ref_chart >= 0)
    pred_valid = np.load(prediction_dir / "pred_valid_mask.npy").astype(bool) if (prediction_dir / "pred_valid_mask.npy").exists() else pred_chart >= 0
    valid = ref_valid & pred_valid
    chart_correct = valid & (ref_chart == pred_chart)
    chart_wrong = valid & ~chart_correct
    seam = dilate(compute_chart_boundary(ref_chart, ref_valid), seam_radius) & valid
    non_seam_correct = chart_correct & ~seam

    delta = pred_uv - ref_uv
    chart_offsets: Dict[int, np.ndarray] = {}
    chart_stats: Dict[int, Dict] = {}
    residual_gt = delta.copy()
    residual_pred = delta.copy()

    all_charts = sorted(int(v) for v in np.unique(ref_chart[valid]))
    for chart_id in all_charts:
        stable_mask = non_seam_correct & (ref_chart == chart_id)
        source = "non_seam_chart_correct"
        if stable_mask.sum() < 16:
            stable_mask = chart_correct & (ref_chart == chart_id)
            source = "all_chart_correct"
        if stable_mask.sum() == 0:
            offset = np.zeros(2, dtype=np.float32)
            source = "fallback_zero"
        else:
            offset = np.median(delta[stable_mask], axis=0).astype(np.float32)
        chart_offsets[chart_id] = offset
        chart_mask = valid & (ref_chart == chart_id)
        corrected_delta = delta[chart_mask] - offset
        chart_stats[chart_id] = {
            "offset_u": float(offset[0]),
            "offset_v": float(offset[1]),
            "offset_l2": float(np.linalg.norm(offset)),
            "offset_angle_deg": float(math.degrees(math.atan2(float(offset[1]), float(offset[0])))),
            "offset_estimation_source": source,
            "offset_estimation_pixels": int(stable_mask.sum()),
            "chart_pixels": int(chart_mask.sum()),
            "chart_accuracy": float((chart_correct & (ref_chart == chart_id)).sum() / max(1, chart_mask.sum())),
            "uv_l2_before": summarize(np.linalg.norm(delta[chart_mask], axis=-1)),
            "uv_l2_after_gt_chart_offset": summarize(np.linalg.norm(corrected_delta, axis=-1)),
        }
        residual_gt[ref_chart == chart_id] -= offset
        residual_pred[pred_chart == chart_id] -= offset

    corrected_uv_gt = pred_uv.copy()
    corrected_uv_pred = pred_uv.copy()
    for chart_id, offset in chart_offsets.items():
        corrected_uv_gt[(ref_chart == chart_id) & valid] -= offset
        corrected_uv_pred[(pred_chart == chart_id) & valid] -= offset

    corrected_gt_img = render_from_uv(texture, corrected_uv_gt, valid)
    corrected_pred_img = render_from_uv(texture, corrected_uv_pred, valid)

    metrics = {
        "original": image_metrics(ref_image, pred_image, valid),
        "gt_chart_corrected": image_metrics(ref_image, corrected_gt_img, valid),
        "pred_chart_corrected": image_metrics(ref_image, corrected_pred_img, valid),
    }

    raw_l2 = np.linalg.norm(delta, axis=-1)
    residual_gt_l2 = np.linalg.norm(residual_gt, axis=-1)
    residual_pred_l2 = np.linalg.norm(residual_pred, axis=-1)
    summary = {
        "reference_dir": str(reference_dir),
        "prediction_dir": str(prediction_dir),
        "texture_path": str(texture_path),
        "seam_radius": int(seam_radius),
        "valid_pixels": int(valid.sum()),
        "chart_correct_pixels": int(chart_correct.sum()),
        "chart_wrong_pixels": int(chart_wrong.sum()),
        "seam_pixels": int(seam.sum()),
        "image_metrics": metrics,
        "uv_l2_before": summarize(raw_l2[valid]),
        "uv_l2_after_gt_chart_offset": summarize(residual_gt_l2[valid]),
        "uv_l2_after_pred_chart_offset": summarize(residual_pred_l2[valid]),
        "uv_l2_before_chart_correct": summarize(raw_l2[chart_correct]),
        "uv_l2_after_gt_chart_offset_chart_correct": summarize(residual_gt_l2[chart_correct]),
        "uv_l2_before_chart_wrong": summarize(raw_l2[chart_wrong]),
        "uv_l2_after_pred_chart_offset_chart_wrong": summarize(residual_pred_l2[chart_wrong]),
        "per_chart": chart_stats,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrected_gt_img).save(output_dir / "compensated_render_gt_chart.png")
    Image.fromarray(corrected_pred_img).save(output_dir / "compensated_render_pred_chart.png")
    Image.fromarray(delta_to_rgb(delta, chart_correct, np.percentile(raw_l2[chart_correct], 99))).save(output_dir / "per_pixel_uv_offset_direction.png")
    Image.fromarray(colorize_l2(residual_gt_l2, valid)).save(output_dir / "residual_after_gt_chart_offset.png")
    Image.fromarray(colorize_l2(residual_pred_l2, valid)).save(output_dir / "residual_after_pred_chart_offset.png")
    draw_chart_offset_map(ref_chart, valid, chart_offsets, chart_stats, output_dir / "chart_median_offset_map.png")
    make_compare_strip(ref_image, pred_image, corrected_gt_img, corrected_pred_img, output_dir / "offset_compensated_compare.png", metrics)

    # Compact chart offset bar plot.
    chart_ids = list(chart_offsets.keys())
    offsets = np.array([chart_offsets[c] for c in chart_ids], dtype=np.float32)
    mags = np.linalg.norm(offsets, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar([str(c) for c in chart_ids], mags, color="#2563eb", alpha=0.8)
    for i, c in enumerate(chart_ids):
        ax.text(i, mags[i], f"{offsets[i,0]:+.3f}\\n{offsets[i,1]:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("GT chart id")
    ax.set_ylabel("Median UV offset magnitude")
    ax.set_title("Chart-wise median UV translation")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_dir / "chart_offset_magnitude.png", bbox_inches="tight")
    plt.close(fig)

    with open(output_dir / "chart_offset_metrics.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    logger.info("Saved chart UV offset analysis: %s", output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze chart-wise UV translation drift.")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seam-radius", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    analyze_offsets(
        reference_dir=Path(args.reference_dir),
        prediction_dir=Path(args.prediction_dir),
        texture_path=Path(args.texture),
        output_dir=Path(args.output_dir),
        seam_radius=args.seam_radius,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
