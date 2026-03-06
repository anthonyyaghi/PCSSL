"""
Encoder Module

Hierarchical encoding components for DBBD.
"""

from .projection import ProjectionMLP, CombineProjection
from .point_encoder import PointCloudEncoder, PointNetBackbone
from .traversal import G2LTraversal, L2GTraversal
from .collector import FeatureCollector
from .hierarchical_encoder import HierarchicalEncoder

__all__ = [
    "ProjectionMLP",
    "CombineProjection",
    "PointCloudEncoder",
    "PointNetBackbone",
    "G2LTraversal",
    "L2GTraversal",
    "FeatureCollector",
    "HierarchicalEncoder",
]

try:
    from .spunet import SpUNet, SpUNetSceneEncoder
    __all__.extend(["SpUNet", "SpUNetSceneEncoder"])
except ImportError:
    pass

