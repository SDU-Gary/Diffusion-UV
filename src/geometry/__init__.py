"""
几何处理模块 - 独立于主干网络的UV映射辅助工具

该模块实现了基于热方法的连续法线场生成，用于替代SDF网络提供切空间投影。
"""

from .heat_method import HeatMethodNormalField
from .projection import ClosestPointProjection

__all__ = ['HeatMethodNormalField', 'ClosestPointProjection']