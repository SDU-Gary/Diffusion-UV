"""JIT loader for the CUDA B-Spline hash-grid op.

The extension is loaded lazily so normal CPU tests and model imports do not
require a CUDA toolchain.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXTENSION = None


def load_extension(verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError("B-Spline hash CUDA extension requires CUDA")

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "bspline_hash.cpp"),
        str(root / "bspline_hash_kernel.cu"),
    ]
    _EXTENSION = load(
        name="bspline_hash_ext",
        sources=sources,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=verbose,
    )
    return _EXTENSION


def bspline_hash_forward(
    positions: torch.Tensor,
    hash_table: torch.Tensor,
    base_res: int,
    max_res: int,
    verbose: bool = False,
) -> torch.Tensor:
    ext = load_extension(verbose=verbose)
    return ext.forward(positions, hash_table, int(base_res), int(max_res))


def bspline_hash_backward_hash(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    hash_table_shape: torch.Size,
    base_res: int,
    max_res: int,
    verbose: bool = False,
) -> torch.Tensor:
    ext = load_extension(verbose=verbose)
    return ext.backward_hash(
        grad_output.contiguous(),
        positions.contiguous(),
        list(hash_table_shape),
        int(base_res),
        int(max_res),
    )
