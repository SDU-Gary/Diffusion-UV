"""
Inference Module

Provides tools for mesh simplification, model inference, and visualization.
"""

from .mesh_simplification import (
    MeshSimplifier,
    create_low_poly_mesh,
)

__all__ = [
    "MeshSimplifier",
    "create_low_poly_mesh",
]
