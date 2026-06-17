#!/usr/bin/env python
"""Analyze per-pixel MA-IUVF predicted UV error against high-UV reference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

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


def load_mask(path: Path, fallback_shape: Tuple[int, int]) -> np.ndarray:
    if path.exists():
        return np.asarray(Image.open(path).convert("L")) > 0
    return np.ones(fallback_shape, dtype=bool)


def summarize_values(values: np.ndarray) -> Dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def mask_summary(name: str, mask: np.ndarray, errors: Dict[str, np.ndarray]) -> Dict:
    out = {
        "name": name,
        "pixels": int(mask.sum()),
        "raw_l2": summarize_values(errors["raw_l2"][mask]),
        "wrap_l2": summarize_values(errors["wrap_l2"][mask]),
        "clipped_l2": summarize_values(errors["clipped_l2"][mask]),
        "raw_abs_u": summarize_values(np.abs(errors["raw_delta"][..., 0][mask])),
        "raw_abs_v": summarize_values(np.abs(errors["raw_delta"][..., 1][mask])),
    }
    return out


def compute_chart_boundary(chart: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(valid, dtype=bool)
    # Avoid np.roll wrap-around by slicing explicit neighbor pairs.
    pairs = [
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ]
    for a, b in pairs:
        both_valid = valid[a] & valid[b]
        diff = chart[a] != chart[b]
        boundary[a] |= both_valid & diff
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


def colorize_error(error: np.ndarray, valid: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    finite = error[valid & np.isfinite(error)]
    scale = float(np.percentile(finite, percentile)) if finite.size else 1.0
    scale = max(scale, 1e-8)
    norm = np.clip(error / scale, 0.0, 1.0)
    cmap = plt.get_cmap("magma")
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def save_binary_overlay(mask: np.ndarray, valid: np.ndarray, path: Path, color=(255, 60, 60)):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[valid] = (38, 42, 50)
    out[mask] = color
    Image.fromarray(out).save(path)


def analyze(
    reference_dir: Path,
    prediction_dir: Path,
    output_dir: Path,
    seam_radius: int,
) -> Dict:
    ref_uv = np.load(reference_dir / "reference_uv.npy").astype(np.float32)
    ref_chart = np.load(reference_dir / "reference_chart_id.npy").astype(np.int32)
    pred_uv = np.load(prediction_dir / "pred_uv.npy").astype(np.float32)
    pred_chart = np.load(prediction_dir / "pred_chart_id.npy").astype(np.int32)

    if ref_uv.shape != pred_uv.shape:
        raise ValueError(f"UV shape mismatch: reference {ref_uv.shape}, prediction {pred_uv.shape}")
    if ref_chart.shape != pred_chart.shape:
        raise ValueError(f"Chart shape mismatch: reference {ref_chart.shape}, prediction {pred_chart.shape}")

    h, w = ref_chart.shape
    ref_valid = load_mask(reference_dir / "reference_mask.png", (h, w)) & (ref_chart >= 0)
    if (prediction_dir / "pred_valid_mask.npy").exists():
        pred_valid = np.load(prediction_dir / "pred_valid_mask.npy").astype(bool)
    else:
        pred_valid = pred_chart >= 0
    valid = ref_valid & pred_valid

    raw_delta = pred_uv - ref_uv
    wrap_delta = ((raw_delta + 0.5) % 1.0) - 0.5
    clipped_delta = np.clip(pred_uv, 0.0, 1.0) - ref_uv
    errors = {
        "raw_delta": raw_delta,
        "raw_l2": np.linalg.norm(raw_delta, axis=-1),
        "wrap_l2": np.linalg.norm(wrap_delta, axis=-1),
        "clipped_l2": np.linalg.norm(clipped_delta, axis=-1),
    }

    chart_correct = valid & (pred_chart == ref_chart)
    chart_wrong = valid & (pred_chart != ref_chart)
    chart_boundary = compute_chart_boundary(ref_chart, ref_valid)
    seam_band = dilate(chart_boundary, seam_radius) & valid
    non_seam = valid & ~seam_band

    summary = {
        "reference_dir": str(reference_dir),
        "prediction_dir": str(prediction_dir),
        "seam_radius": int(seam_radius),
        "total_pixels": int(valid.size),
        "valid_pixels": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "chart_correct_pixels": int(chart_correct.sum()),
        "chart_wrong_pixels": int(chart_wrong.sum()),
        "chart_accuracy": float(chart_correct.sum() / max(1, valid.sum())),
        "seam_band_pixels": int(seam_band.sum()),
        "non_seam_pixels": int(non_seam.sum()),
        "regions": {
            "all_valid": mask_summary("all_valid", valid, errors),
            "chart_correct": mask_summary("chart_correct", chart_correct, errors),
            "chart_wrong": mask_summary("chart_wrong", chart_wrong, errors),
            "seam_band": mask_summary("seam_band", seam_band, errors),
            "non_seam": mask_summary("non_seam", non_seam, errors),
            "seam_chart_correct": mask_summary("seam_chart_correct", seam_band & chart_correct, errors),
            "seam_chart_wrong": mask_summary("seam_chart_wrong", seam_band & chart_wrong, errors),
            "non_seam_chart_correct": mask_summary("non_seam_chart_correct", non_seam & chart_correct, errors),
            "non_seam_chart_wrong": mask_summary("non_seam_chart_wrong", non_seam & chart_wrong, errors),
        },
        "per_gt_chart": {},
    }

    for chart_id in sorted(int(v) for v in np.unique(ref_chart[valid])):
        gt_mask = valid & (ref_chart == chart_id)
        correct = gt_mask & (pred_chart == chart_id)
        summary["per_gt_chart"][chart_id] = {
            "pixels": int(gt_mask.sum()),
            "chart_accuracy": float(correct.sum() / max(1, gt_mask.sum())),
            "clipped_l2": summarize_values(errors["clipped_l2"][gt_mask]),
            "clipped_l2_correct_only": summarize_values(errors["clipped_l2"][correct]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "uv_error_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    Image.fromarray(colorize_error(errors["clipped_l2"], valid)).save(output_dir / "uv_clipped_l2_error_map.png")
    Image.fromarray(colorize_error(errors["wrap_l2"], valid)).save(output_dir / "uv_wrap_l2_error_map.png")
    save_binary_overlay(chart_wrong, valid, output_dir / "chart_mismatch_map.png", color=(255, 50, 50))
    save_binary_overlay(seam_band, valid, output_dir / "seam_band_map.png", color=(60, 180, 255))

    # Histogram and per-chart plots.
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for name, mask, color in [
        ("all", valid, "#334155"),
        ("chart correct", chart_correct, "#059669"),
        ("seam band", seam_band, "#0284c7"),
        ("chart wrong", chart_wrong, "#dc2626"),
    ]:
        vals = errors["clipped_l2"][mask]
        if vals.size:
            ax.hist(vals, bins=80, alpha=0.45, label=f"{name} (n={vals.size})", color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Clipped UV L2 error")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title("Per-pixel clipped UV error distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "uv_error_histogram.png", bbox_inches="tight")
    plt.close(fig)

    chart_ids = list(summary["per_gt_chart"].keys())
    means = [summary["per_gt_chart"][c]["clipped_l2"]["mean"] or 0 for c in chart_ids]
    accs = [summary["per_gt_chart"][c]["chart_accuracy"] for c in chart_ids]
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    ax1.bar([str(c) for c in chart_ids], means, color="#2563eb", alpha=0.75)
    ax1.set_xlabel("GT chart id")
    ax1.set_ylabel("Mean clipped UV L2", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot([str(c) for c in chart_ids], accs, color="#dc2626", marker="o")
    ax2.set_ylabel("Chart accuracy", color="#dc2626")
    ax2.set_ylim(0.0, 1.02)
    ax1.set_title("Per-chart UV error and chart accuracy")
    fig.savefig(output_dir / "per_chart_uv_error.png", bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved UV error analysis: %s", metrics_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze per-pixel predicted UV error.")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seam-radius", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    analyze(
        reference_dir=Path(args.reference_dir),
        prediction_dir=Path(args.prediction_dir),
        output_dir=Path(args.output_dir),
        seam_radius=args.seam_radius,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
