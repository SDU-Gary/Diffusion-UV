"""
MA-IUVF Analysis Module

Provides experimental framework for analyzing MA-IUVF model behavior:
- Continuity contradictions at chart boundaries
- Ambient grid misalignment in thin regions
- Non-manifold extrapolation robustness
- Normal gradient noise validation
"""

from .maiuvf_analyzer import MAIUVFAnalyzer

__all__ = ['MAIUVFAnalyzer']
