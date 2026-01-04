"""
Geometric Mechanism Analyzers for RLHF in Tangent Space

Three core mechanisms:
1. Riemannian Confinement: KL trust region bounds parameter displacement
2. Fisher Steering: FIM penalizes principal component updates
3. Sequence-Level Stability: Linearization bounds for long-horizon reasoning
"""

from .riemannian_confinement import RiemannianConfinementAnalyzer
from .fisher_steering import FisherSteeringAnalyzer
from .sequence_stability import SequenceLevelStabilityAnalyzer

__all__ = [
    'RiemannianConfinementAnalyzer',
    'FisherSteeringAnalyzer',
    'SequenceLevelStabilityAnalyzer',
]
