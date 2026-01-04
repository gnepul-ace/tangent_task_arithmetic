"""
RLHF in the Tangent Space Regime

This module implements RLHF with geometric analysis to validate
the hypothesis that RLHF operates in the tangent space regime.
"""

from .linearize_lm import LinearizedLLM, ImprovedLinearizedLLM
from .fisher_information import FisherInformationComputer
from .ntk_tracker import NTKTracker

__all__ = [
    'LinearizedLLM',
    'ImprovedLinearizedLLM',
    'FisherInformationComputer',
    'NTKTracker',
]
