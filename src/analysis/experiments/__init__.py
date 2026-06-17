"""
MA-IUVF Analysis Experiments

Individual experiment implementations:
- exp1_seam_continuity: Seam jump & network hesitation test
- exp2_thin_shell: Thin-shell feature penetration test
- exp3_extrapolation: Non-manifold extrapolation robustness test
- exp4_normal_noise: Normal gradient noise validation
"""

from .exp4_normal_noise import run_experiment4
from .exp1_seam_continuity import run_experiment1
from .exp2_thin_shell import run_experiment2
from .exp3_extrapolation import run_experiment3

__all__ = [
    'run_experiment1',
    'run_experiment2',
    'run_experiment3',
    'run_experiment4'
]
