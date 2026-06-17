"""Cubic B-Spline multi-resolution hash-grid encoder.

The module provides two execution paths:
- a CUDA custom op for fast forward and hash-table gradient accumulation;
- a pure PyTorch path used whenever gradients with respect to input positions
  are required. MA-IUVF's metric loss needs dUV/dpos with create_graph=True,
  so the training path must remain fully differentiable through positions.
"""

from __future__ import annotations

import math
import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BSplineHashGridFunction(torch.autograd.Function):
    """Autograd wrapper around the CUDA hash-grid op.

    This backward only returns gradients for the hash table. It is intended for
    inference or anchor/classification-only training paths. Full metric-loss
    training should use :class:`BSplineHashGrid`'s PyTorch path.
    """

    @staticmethod
    def forward(ctx, positions, hash_table, base_res: int, max_res: int):
        from src.ops.bspline_hash import bspline_hash_forward

        positions = positions.contiguous()
        hash_table = hash_table.contiguous()
        ctx.save_for_backward(positions)
        ctx.hash_table_shape = hash_table.shape
        ctx.base_res = int(base_res)
        ctx.max_res = int(max_res)
        return bspline_hash_forward(positions, hash_table, base_res, max_res)

    @staticmethod
    def backward(ctx, grad_output):
        from src.ops.bspline_hash import bspline_hash_backward_hash

        (positions,) = ctx.saved_tensors
        grad_hash = bspline_hash_backward_hash(
            grad_output,
            positions,
            ctx.hash_table_shape,
            ctx.base_res,
            ctx.max_res,
        )
        return None, grad_hash, None, None


class BSplineHashGrid(nn.Module):
    """Multi-resolution cubic B-Spline hash-grid encoder.

    Args:
        num_levels: Number of resolution levels L.
        features_per_level: Feature dimension F per level.
        log2_hashmap_size: Hash table size T is ``2 ** log2_hashmap_size``.
        base_res: Coarsest grid resolution.
        max_res: Finest grid resolution.
        init_scale: Uniform initialization range for hash features.
        cuda_backend: ``"auto"``, ``"torch"``, or ``"cuda"``.
        normalize_positions: If true, input positions are mapped from bbox
            bounds to [0, 1]^3 before encoding.
    """

    PRIME_X = 73_856_093
    PRIME_Y = 19_349_663
    PRIME_Z = 83_492_791

    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_res: int = 16,
        max_res: int = 2048,
        init_scale: float = 1e-4,
        cuda_backend: str = "auto",
        normalize_positions: bool = True,
        bbox_min: Optional[Iterable[float]] = None,
        bbox_max: Optional[Iterable[float]] = None,
    ):
        super().__init__()
        if cuda_backend not in {"auto", "torch", "cuda"}:
            raise ValueError(f"unknown cuda_backend: {cuda_backend}")
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        if features_per_level < 1:
            raise ValueError("features_per_level must be >= 1")

        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.hashmap_size = int(2 ** log2_hashmap_size)
        self.base_res = int(base_res)
        self.max_res = int(max_res)
        self.output_dim = self.num_levels * self.features_per_level
        self.cuda_backend = cuda_backend
        self.normalize_positions = bool(normalize_positions)

        table = torch.empty(
            self.num_levels,
            self.hashmap_size,
            self.features_per_level,
            dtype=torch.float32,
        )
        nn.init.uniform_(table, -init_scale, init_scale)
        self.hash_table = nn.Parameter(table)

        if bbox_min is None:
            bbox_min = (0.0, 0.0, 0.0)
        if bbox_max is None:
            bbox_max = (1.0, 1.0, 1.0)
        self.register_buffer("bbox_min", torch.tensor(list(bbox_min), dtype=torch.float32))
        self.register_buffer("bbox_max", torch.tensor(list(bbox_max), dtype=torch.float32))

    def set_normalization_bounds(self, bbox_min, bbox_max) -> None:
        self.bbox_min.copy_(torch.as_tensor(bbox_min, dtype=self.bbox_min.dtype, device=self.bbox_min.device))
        self.bbox_max.copy_(torch.as_tensor(bbox_max, dtype=self.bbox_max.dtype, device=self.bbox_max.device))

    def _normalize(self, positions: torch.Tensor) -> torch.Tensor:
        if not self.normalize_positions:
            return positions.clamp(0.0, 1.0)
        denom = (self.bbox_max - self.bbox_min).clamp_min(1e-8)
        return ((positions - self.bbox_min) / denom).clamp(0.0, 1.0)

    def _level_resolution(self, level: int) -> float:
        if self.num_levels == 1:
            return float(self.base_res)
        ratio = float(self.max_res) / float(self.base_res)
        return float(self.base_res) * (ratio ** (float(level) / float(self.num_levels - 1)))

    @staticmethod
    def _bspline_weights(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t2 = t * t
        t3 = t2 * t
        one_minus_t = 1.0 - t
        w0 = (one_minus_t * one_minus_t * one_minus_t) / 6.0
        w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        w3 = t3 / 6.0
        return w0, w1, w2, w3

    def _hash(self, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        x = ix.to(torch.int64) * self.PRIME_X
        y = iy.to(torch.int64) * self.PRIME_Y
        z = iz.to(torch.int64) * self.PRIME_Z
        return torch.remainder(torch.bitwise_xor(torch.bitwise_xor(x, y), z), self.hashmap_size).long()

    def _forward_torch(self, positions01: torch.Tensor) -> torch.Tensor:
        features = []
        offsets = (-1, 0, 1, 2)
        for level in range(self.num_levels):
            res = self._level_resolution(level)
            p = positions01 * res
            base = torch.floor(p).to(torch.int64)
            t = p - base.to(p.dtype)
            wx = self._bspline_weights(t[:, 0])
            wy = self._bspline_weights(t[:, 1])
            wz = self._bspline_weights(t[:, 2])

            level_feat = positions01.new_zeros((positions01.shape[0], self.features_per_level))
            table = self.hash_table[level]
            for ox_i, ox in enumerate(offsets):
                ix = base[:, 0] + ox
                for oy_i, oy in enumerate(offsets):
                    iy = base[:, 1] + oy
                    wxy = wx[ox_i] * wy[oy_i]
                    for oz_i, oz in enumerate(offsets):
                        iz = base[:, 2] + oz
                        h = self._hash(ix, iy, iz)
                        w = (wxy * wz[oz_i]).unsqueeze(-1)
                        level_feat = level_feat + w * table.index_select(0, h)
            features.append(level_feat)
        return torch.cat(features, dim=-1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions01 = self._normalize(positions)
        use_cuda = (
            self.cuda_backend in {"auto", "cuda"}
            and positions01.is_cuda
            and self.hash_table.is_cuda
            and positions01.dtype == torch.float32
            and self.hash_table.dtype == torch.float32
            and not positions01.requires_grad
        )
        if self.cuda_backend == "cuda" and not use_cuda:
            raise RuntimeError(
                "CUDA B-Spline path requires CUDA float32 positions without position gradients"
            )
        if use_cuda:
            try:
                return BSplineHashGridFunction.apply(
                    positions01,
                    self.hash_table,
                    self.base_res,
                    self.max_res,
                )
            except Exception as exc:
                if self.cuda_backend == "cuda":
                    raise
                logger.warning("B-Spline CUDA path failed, falling back to PyTorch: %s", exc)
        return self._forward_torch(positions01)
