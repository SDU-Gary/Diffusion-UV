"""
Compare a MA-IUVF prediction render against a high-poly UV reference render.

The comparison is screen-space and mask-aware. It is intended for Phase 1
experiments where the reference and prediction use the same rasterization
convention and resolution.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

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


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: str, shape) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        data = np.load(p)
        if data.ndim == 3:
            data = data[..., 0]
        return data.astype(bool)

    image = np.asarray(Image.open(p))
    if image.ndim == 3:
        image = image[..., 0]
    return image > 0


def maybe_resize_prediction(pred: np.ndarray, target_shape, allow_resize: bool) -> np.ndarray:
    if pred.shape == target_shape:
        return pred
    if not allow_resize:
        raise ValueError(f"Image shapes differ: prediction {pred.shape}, reference {target_shape}")
    resized = Image.fromarray(np.clip(pred * 255.0, 0, 255).astype(np.uint8)).resize(
        (target_shape[1], target_shape[0]),
        Image.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def make_error_map(abs_error: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scalar = abs_error.mean(axis=-1)
    if mask.any():
        max_value = max(float(np.percentile(scalar[mask], 99.0)), 1e-8)
    else:
        max_value = 1.0
    normalized = np.clip(scalar / max_value, 0.0, 1.0)

    heat = np.zeros((*scalar.shape, 3), dtype=np.float32)
    heat[..., 0] = normalized
    heat[..., 1] = np.maximum(0.0, 1.0 - np.abs(normalized - 0.5) * 2.0)
    heat[..., 2] = 1.0 - normalized
    heat[~mask] = 0.0
    return np.clip(heat * 255.0, 0, 255).astype(np.uint8)


def compute_ssim(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    # SSIM is computed on the masked bounding box to avoid background dominance.
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    ref_crop = ref[y0:y1, x0:x1]
    pred_crop = pred[y0:y1, x0:x1]
    if ref_crop.shape[0] < 7 or ref_crop.shape[1] < 7:
        return None
    return float(
        structural_similarity(
            ref_crop,
            pred_crop,
            channel_axis=-1,
            data_range=1.0,
        )
    )


def compare_images(
    reference_path: str,
    prediction_path: str,
    output_dir: str,
    reference_mask_path: Optional[str],
    prediction_mask_path: Optional[str],
    mask_mode: str,
    resize_prediction: bool,
) -> Dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ref = load_image(reference_path)
    pred = load_image(prediction_path)
    pred = maybe_resize_prediction(pred, ref.shape, resize_prediction)

    ref_mask = load_mask(reference_mask_path, ref.shape) if reference_mask_path else np.ones(ref.shape[:2], dtype=bool)
    pred_mask = load_mask(prediction_mask_path, pred.shape) if prediction_mask_path else None

    if mask_mode == "reference":
        mask = ref_mask
    elif mask_mode == "prediction":
        if pred_mask is None:
            raise ValueError("--prediction-mask is required when --mask-mode prediction")
        mask = pred_mask
    elif mask_mode == "intersection":
        if pred_mask is None:
            logger.warning("No prediction mask provided; using reference mask for intersection")
            mask = ref_mask
        else:
            mask = ref_mask & pred_mask
    elif mask_mode == "union":
        if pred_mask is None:
            logger.warning("No prediction mask provided; using reference mask for union")
            mask = ref_mask
        else:
            mask = ref_mask | pred_mask
    else:
        raise ValueError(f"Unknown mask mode: {mask_mode}")

    diff = pred - ref
    abs_error = np.abs(diff)
    sq_error = diff * diff

    if mask.any():
        masked_sq = sq_error[mask]
        masked_abs = abs_error[mask]
        mse = float(masked_sq.mean())
        mae = float(masked_abs.mean())
        rmse = float(np.sqrt(mse))
        per_channel_mse = masked_sq.mean(axis=0)
    else:
        mse = mae = rmse = float("nan")
        per_channel_mse = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    ssim = compute_ssim(ref, pred, mask)

    error_map = make_error_map(abs_error, mask)
    error_path = output / "error_map.png"
    side_by_side_path = output / "side_by_side.png"
    metrics_path = output / "metrics.json"

    Image.fromarray(error_map).save(error_path)
    side = np.concatenate(
        [
            np.clip(ref * 255.0, 0, 255).astype(np.uint8),
            np.clip(pred * 255.0, 0, 255).astype(np.uint8),
            error_map,
        ],
        axis=1,
    )
    Image.fromarray(side).save(side_by_side_path)

    metrics = {
        "reference_path": str(reference_path),
        "prediction_path": str(prediction_path),
        "mask_mode": mask_mode,
        "num_pixels": int(mask.size),
        "valid_pixels": int(mask.sum()),
        "valid_ratio": float(mask.mean()),
        "reference_coverage": float(ref_mask.mean()),
        "prediction_coverage": float(pred_mask.mean()) if pred_mask is not None else None,
        "coverage_diff": float(pred_mask.mean() - ref_mask.mean()) if pred_mask is not None else None,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
        "per_channel_mse": per_channel_mse.tolist(),
        "outputs": {
            "error_map": str(error_path),
            "side_by_side": str(side_by_side_path),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)

    logger.info("MSE %.6f | PSNR %.2f dB | valid %.2f%%", mse, psnr, metrics["valid_ratio"] * 100.0)
    logger.info("Saved metrics: %s", metrics_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MA-IUVF render against reference.")
    parser.add_argument("--reference", required=True, help="Reference RGB image")
    parser.add_argument("--prediction", required=True, help="Prediction RGB image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-mask", help="Reference mask PNG or NPY")
    parser.add_argument("--prediction-mask", help="Prediction mask PNG or NPY")
    parser.add_argument(
        "--mask-mode",
        choices=["reference", "prediction", "intersection", "union"],
        default="reference",
    )
    parser.add_argument("--resize-prediction", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    compare_images(
        reference_path=args.reference,
        prediction_path=args.prediction,
        output_dir=args.output_dir,
        reference_mask_path=args.reference_mask,
        prediction_mask_path=args.prediction_mask,
        mask_mode=args.mask_mode,
        resize_prediction=args.resize_prediction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
