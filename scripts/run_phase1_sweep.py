"""
Run MA-IUVF Phase 1 experiments.

Each run performs:
  bake -> train -> high-UV reference render -> MA-IUVF render -> image compare
and optionally exports low-poly textured OBJ demos from the same checkpoint.
"""

import argparse
import csv
import itertools
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def safe_name_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    return value


def run_command(cmd: List[str], cwd: Path, log_path: Path, dry_run: bool) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running: %s", " ".join(cmd))
    start = time.time()

    if dry_run:
        with open(log_path, "w") as f:
            f.write("DRY RUN\n")
            f.write(" ".join(cmd) + "\n")
        return 0.0

    with open(log_path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    logger.info("Completed in %.1fs", elapsed)
    return elapsed


def read_train_metrics(csv_path: Path) -> Dict:
    if not csv_path.exists():
        return {}
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return {}

    def column_float(name: str) -> List[float]:
        return [float(row[name]) for row in rows if row.get(name) not in (None, "")]

    loss = column_float("loss")
    metric = column_float("metric")
    cls_acc = column_float("cls_acc") if "cls_acc" in rows[0] else []

    out = {
        "epochs": len(rows),
        "initial_loss": loss[0] if loss else None,
        "final_loss": loss[-1] if loss else None,
        "best_loss": min(loss) if loss else None,
        "initial_metric": metric[0] if metric else None,
        "final_metric": metric[-1] if metric else None,
        "best_metric": min(metric) if metric else None,
    }
    if cls_acc:
        out["final_cls_acc"] = cls_acc[-1]
        out["best_cls_acc"] = max(cls_acc)
    return out


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def read_bake_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    baked = torch.load(path, map_location="cpu", weights_only=False)
    metadata = baked.get("metadata", {})
    chart_id = baked.get("chart_id")
    face_chart_id = baked.get("face_chart_id")
    out = {
        "data_kind": metadata.get("data_kind", "baked_samples"),
        "num_samples": metadata.get("num_samples"),
        "num_faces": metadata.get("num_faces"),
        "num_charts": metadata.get("num_charts"),
        "chart_mode": metadata.get("chart_mode"),
        "extrusion_sigma_ratio": metadata.get("extrusion_sigma_ratio"),
        "extrusion_sigma": metadata.get("extrusion_sigma"),
    }
    if chart_id is not None:
        out["chart_sample_counts"] = torch.bincount(
            chart_id.long(), minlength=int(chart_id.max().item()) + 1
        ).tolist()
    if face_chart_id is not None:
        out["chart_face_counts"] = torch.bincount(
            face_chart_id.long(), minlength=int(face_chart_id.max().item()) + 1
        ).tolist()
    if "chart_stats" in metadata:
        out["chart_stats"] = metadata["chart_stats"]
    return out


def run_one(
    args,
    samples: int,
    sigma_ratio: float,
    epochs: int,
    run_index: int,
) -> Dict:
    run_name = (
        f"run_{run_index:03d}_samples{samples}_sigma{safe_name_float(sigma_ratio)}"
        f"_epochs{epochs}"
    )
    run_dir = Path(args.output_dir) / run_name
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    python = args.python
    cwd = Path(args.project_root)

    data_path = run_dir / ("mesh_constants.pt" if args.sampling_mode == "dynamic_gpu" else "baked.pt")
    train_dir = run_dir / "train"
    reference_dir = run_dir / "reference"
    prediction_dir = run_dir / "prediction"
    compare_dir = run_dir / "compare"
    low_dir = run_dir / "low_demos"

    config = {
        "input_mesh": args.input_mesh,
        "texture": args.texture,
        "samples": samples,
        "sampling_mode": args.sampling_mode,
        "sigma_ratio": sigma_ratio,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hash_lr": args.hash_lr,
        "metric_weight": args.metric_weight,
        "anchor_weight": args.anchor_weight,
        "com_weight": args.com_weight,
        "cls_weight": args.cls_weight,
        "loss_schedule": args.loss_schedule,
        "phase_a_epochs": args.phase_a_epochs,
        "target_metric_weight": args.target_metric_weight,
        "target_anchor_weight": args.target_anchor_weight,
        "target_cls_weight": args.target_cls_weight,
        "schedule_ramp": args.schedule_ramp,
        "encoder_type": args.encoder_type,
        "activation": args.activation,
        "hash_num_levels": args.hash_num_levels,
        "hash_features_per_level": args.hash_features_per_level,
        "hash_log2_size": args.hash_log2_size,
        "hash_base_res": args.hash_base_res,
        "hash_max_res": args.hash_max_res,
        "hash_cuda_backend": args.hash_cuda_backend,
        "hash_weight_decay": args.hash_weight_decay,
        "mlp_weight_decay": args.mlp_weight_decay,
        "resolution": args.resolution,
        "reference_backend": args.reference_backend,
        "device": args.device,
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(to_jsonable(config), f, indent=2)

    timings = {}
    if args.sampling_mode == "dynamic_gpu":
        bake_cmd = [
            python,
            "scripts/bake_metric_aligned_iuv_constants.py",
            "--mesh",
            args.input_mesh,
            "--output",
            str(data_path),
            "--extrusion-sigma-ratio",
            str(sigma_ratio),
            "--chart-mode",
            args.chart_mode,
            "--seed",
            str(args.seed),
            "--texture",
            args.texture,
        ]
    else:
        bake_cmd = [
            python,
            "scripts/bake_metric_aligned_iuv_data.py",
            "--mesh",
            args.input_mesh,
            "--output",
            str(data_path),
            "--num-samples",
            str(samples),
            "--extrusion-sigma-ratio",
            str(sigma_ratio),
            "--chart-mode",
            args.chart_mode,
            "--seed",
            str(args.seed),
            "--texture",
            args.texture,
        ]

    timings["bake"] = run_command(
        bake_cmd,
        cwd,
        logs_dir / "bake.log",
        args.dry_run,
    )

    train_cmd = [
        python,
        "scripts/train_metric_aligned_iuv_field.py",
        "--data",
        str(data_path),
        "--data-format",
        "mesh_constants" if args.sampling_mode == "dynamic_gpu" else "baked",
        "--output-dir",
        str(train_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hash-lr",
        str(args.hash_lr if args.hash_lr is not None else args.lr),
        "--device",
        args.device,
        "--encoder-type",
        args.encoder_type,
        "--activation",
        args.activation,
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--positional-enc-freqs",
        str(args.positional_enc_freqs),
        "--hash-num-levels",
        str(args.hash_num_levels),
        "--hash-features-per-level",
        str(args.hash_features_per_level),
        "--hash-log2-size",
        str(args.hash_log2_size),
        "--hash-base-res",
        str(args.hash_base_res),
        "--hash-max-res",
        str(args.hash_max_res),
        "--hash-cuda-backend",
        args.hash_cuda_backend,
        "--metric-weight",
        str(args.metric_weight),
        "--anchor-weight",
        str(args.anchor_weight),
        "--com-weight",
        str(args.com_weight),
        "--cls-weight",
        str(args.cls_weight),
        "--loss-schedule",
        args.loss_schedule,
        "--phase-a-epochs",
        str(args.phase_a_epochs),
        "--target-metric-weight",
        str(args.target_metric_weight),
        "--target-anchor-weight",
        str(args.target_anchor_weight),
        "--target-cls-weight",
        str(args.target_cls_weight),
        "--schedule-ramp",
        args.schedule_ramp,
        "--hash-weight-decay",
        str(args.hash_weight_decay),
        "--mlp-weight-decay",
        str(args.mlp_weight_decay),
        "--dynamic-cls-decay",
        str(args.dynamic_cls_decay),
        "--cls-decay-epoch-threshold",
        str(args.cls_decay_epoch_threshold),
        "--cls-acc-threshold",
        str(args.cls_acc_threshold),
    ]
    if args.sampling_mode == "dynamic_gpu":
        train_cmd.extend(
            [
                "--virtual-epoch-size",
                str(samples),
                "--sigma-ratio",
                str(sigma_ratio),
                "--sampling-seed",
                str(args.seed),
            ]
        )

    timings["train"] = run_command(
        train_cmd,
        cwd,
        logs_dir / "train.log",
        args.dry_run,
    )

    checkpoint = train_dir / args.checkpoint_name
    timings["reference"] = run_command(
        [
            python,
            "scripts/render_high_uv_reference.py",
            "--mesh",
            args.input_mesh,
            "--texture",
            args.texture,
            "--baked-data",
            str(data_path),
            "--output-dir",
            str(reference_dir),
            "--resolution",
            str(args.resolution),
            "--backend",
            args.reference_backend,
            "--sampling",
            args.reference_sampling,
        ],
        cwd,
        logs_dir / "reference.log",
        args.dry_run,
    )

    timings["prediction"] = run_command(
        [
            python,
            "scripts/render_metric_aligned_iuv_test.py",
            "--checkpoint",
            str(checkpoint),
            "--input-mesh",
            args.input_mesh,
            "--texture",
            args.texture,
            "--output-dir",
            str(prediction_dir),
            "--render-mode",
            "cpu",
            "--resolution",
            str(args.resolution),
            "--device",
            args.device,
            "--no-viewer",
        ],
        cwd,
        logs_dir / "prediction.log",
        args.dry_run,
    )

    timings["compare"] = run_command(
        [
            python,
            "scripts/compare_render_metrics.py",
            "--reference",
            str(reference_dir / "reference.png"),
            "--prediction",
            str(prediction_dir / "render_cpu.png"),
            "--reference-mask",
            str(reference_dir / "reference_mask.png"),
            "--output-dir",
            str(compare_dir),
            "--mask-mode",
            args.compare_mask_mode,
        ],
        cwd,
        logs_dir / "compare.log",
        args.dry_run,
    )

    low_outputs = []
    for target_faces in args.target_faces_list:
        target_dir = low_dir / f"faces_{target_faces}"
        timings[f"low_{target_faces}"] = run_command(
            [
                python,
                "scripts/infer_metric_aligned_iuv.py",
                "--checkpoint",
                str(checkpoint),
                "--input-mesh",
                args.input_mesh,
                "--texture",
                args.texture,
                "--output-dir",
                str(target_dir),
                "--target-faces",
                str(target_faces),
                "--device",
                args.device,
                "--export-npz",
            ],
            cwd,
            logs_dir / f"low_{target_faces}.log",
            args.dry_run,
        )
        low_outputs.append(str(target_dir / "low_maiuvf_textured.obj"))

    if args.dry_run:
        summary = {"run_name": run_name, "config": config, "dry_run": True}
    else:
        summary = {
            "run_name": run_name,
            "config": config,
            "timings": timings,
            "bake": read_bake_metrics(data_path),
            "train": read_train_metrics(train_dir / "train_loss.csv"),
            "reference": read_json(reference_dir / "reference_info.json"),
            "prediction": read_json(prediction_dir / "render_info.json"),
            "compare": read_json(compare_dir / "metrics.json"),
            "low_outputs": low_outputs,
        }

    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    return summary


def write_summary_csv(output_dir: Path, summaries: List[Dict]):
    path = output_dir / "phase1_summary.csv"
    fields = [
        "run_name",
        "samples",
        "sigma_ratio",
        "epochs",
        "encoder_type",
        "num_charts",
        "final_loss",
        "best_loss",
        "final_metric",
        "best_metric",
        "best_cls_acc",
        "render_coverage",
        "render_chart_accuracy",
        "reference_coverage",
        "mse",
        "psnr",
        "ssim",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            config = summary.get("config", {})
            bake = summary.get("bake", {})
            train = summary.get("train", {})
            pred = summary.get("prediction", {})
            ref = summary.get("reference", {})
            comp = summary.get("compare", {})
            writer.writerow(
                {
                    "run_name": summary.get("run_name"),
                    "samples": config.get("samples"),
                    "sigma_ratio": config.get("sigma_ratio"),
                    "epochs": config.get("epochs"),
                    "encoder_type": config.get("encoder_type"),
                    "num_charts": bake.get("num_charts"),
                    "final_loss": train.get("final_loss"),
                    "best_loss": train.get("best_loss"),
                    "final_metric": train.get("final_metric"),
                    "best_metric": train.get("best_metric"),
                    "best_cls_acc": train.get("best_cls_acc"),
                    "render_coverage": pred.get("coverage"),
                    "render_chart_accuracy": pred.get("chart_accuracy"),
                    "reference_coverage": ref.get("coverage"),
                    "mse": comp.get("mse"),
                    "psnr": comp.get("psnr"),
                    "ssim": comp.get("ssim"),
                }
            )
    logger.info("Saved sweep summary: %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MA-IUVF Phase 1 sweep.")
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).parent.parent))
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--num-samples-list", default="10000")
    parser.add_argument(
        "--sampling-mode",
        choices=["static", "dynamic_gpu"],
        default="static",
        help="static 使用离线点云；dynamic_gpu 使用 mesh constants 并在训练 step 内动态采样",
    )
    parser.add_argument("--sigma-ratios", default="0.01")
    parser.add_argument("--epochs-list", default="10")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hash-lr", type=float, default=None)
    parser.add_argument("--metric-weight", type=float, default=0.01)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--com-weight", type=float, default=0.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--loss-schedule", choices=["fixed", "two_stage"], default="fixed")
    parser.add_argument("--phase-a-epochs", type=int, default=30)
    parser.add_argument("--target-metric-weight", type=float, default=1.0)
    parser.add_argument("--target-anchor-weight", type=float, default=0.01)
    parser.add_argument("--target-cls-weight", type=float, default=0.1)
    parser.add_argument("--schedule-ramp", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--encoder-type", choices=["fourier", "bspline_hash"], default="bspline_hash")
    parser.add_argument("--activation", choices=["softplus", "silu", "relu"], default="silu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--positional-enc-freqs", type=int, default=8)
    parser.add_argument("--hash-num-levels", type=int, default=16)
    parser.add_argument("--hash-features-per-level", type=int, default=2)
    parser.add_argument("--hash-log2-size", type=int, default=19)
    parser.add_argument("--hash-base-res", type=int, default=16)
    parser.add_argument("--hash-max-res", type=int, default=2048)
    parser.add_argument("--hash-cuda-backend", choices=["auto", "torch", "cuda"], default="auto")
    parser.add_argument("--hash-weight-decay", type=float, default=1e-6)
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0)
    # NEW: Dynamic Classification Loss Decay Arguments
    parser.add_argument("--dynamic-cls-decay", type=float, default=0.01,
                        help="Exponential decay factor for classification weight when cls_acc > 0.99")
    parser.add_argument("--cls-decay-epoch-threshold", type=int, default=20,
                        help="Epoch threshold to start checking classification accuracy for decay")
    parser.add_argument("--cls-acc-threshold", type=float, default=0.99,
                        help="Classification accuracy threshold to trigger exponential decay")
    parser.add_argument("--chart-mode", default="uv_islands", choices=["uv_islands", "face_component"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--reference-backend", choices=["cpu", "opengl", "auto"], default="cpu")
    parser.add_argument("--reference-sampling", choices=["bilinear", "nearest"], default="bilinear")
    parser.add_argument(
        "--compare-mask-mode",
        choices=["reference", "prediction", "intersection", "union"],
        default="reference",
    )
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--target-faces-list", default="", help="Optional comma list for low-poly OBJ demos")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.num_samples_list = parse_int_list(args.num_samples_list)
    args.sigma_ratio_list = parse_float_list(args.sigma_ratios)
    args.epochs_list = parse_int_list(args.epochs_list)
    args.target_faces_list = parse_int_list(args.target_faces_list)

    if not args.num_samples_list or not args.sigma_ratio_list or not args.epochs_list:
        raise ValueError("Sweep lists must not be empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.num_samples_list, args.sigma_ratio_list, args.epochs_list))
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    summaries = []
    failures = []
    for index, (samples, sigma_ratio, epochs) in enumerate(combos):
        logger.info(
            "Starting run %d/%d: samples=%d sigma=%s epochs=%d",
            index + 1,
            len(combos),
            samples,
            sigma_ratio,
            epochs,
        )
        try:
            summaries.append(run_one(args, samples, sigma_ratio, epochs, index))
        except Exception as exc:
            logger.exception("Run failed: %s", exc)
            failures.append(
                {
                    "index": index,
                    "samples": samples,
                    "sigma_ratio": sigma_ratio,
                    "epochs": epochs,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                break

    write_summary_csv(output_dir, summaries)
    with open(output_dir / "phase1_failures.json", "w") as f:
        json.dump(to_jsonable(failures), f, indent=2)

    if failures and not args.continue_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
