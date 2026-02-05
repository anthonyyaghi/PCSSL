"""
Feature Processing Networks

Bidirectional feature transformation modules.
"""

from .propagator import FeaturePropagator
from .aggregator import FeatureAggregator

__all__ = ["FeaturePropagator", "FeatureAggregator"]
