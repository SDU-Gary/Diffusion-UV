#!/usr/bin/env python
"""Bake MA-IUVF per-face mesh constants for dynamic GPU sampling."""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.gpu_constant_baker import MetricAlignedIUVGPUConstantBaker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bake MA-IUVF mesh_constants.pt for GPU dynamic sampling."
    )
    parser.add_argument("--mesh", required=True, help="Input OBJ with face-corner UVs")
    parser.add_argument("--output", required=True, help="Output mesh_constants.pt")
    parser.add_argument("--extrusion-sigma-ratio", type=float, default=0.01)
    parser.add_argument("--chart-mode", choices=["uv_islands", "face_component"], default="uv_islands")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture", help="Optional texture path stored in metadata")
    parser.add_argument("--no-obj-parser", action="store_true", help="Use trimesh instead of direct OBJ parser")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    baker = MetricAlignedIUVGPUConstantBaker(
        mesh_path=args.mesh,
        seed=args.seed,
        use_obj_parser=not args.no_obj_parser,
    )
    baker.save(
        output_path=args.output,
        extrusion_sigma_ratio=args.extrusion_sigma_ratio,
        chart_mode=args.chart_mode,
        texture_path=args.texture,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
