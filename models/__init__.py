"""
TASA: Transition-Aware Sparse Attention for Sleep Stage Classification

Model components:
- SparseAttention: Local windowed attention mechanism
- SparseTransformerBackbone: Efficient transformer encoder
- SleepStagingHead: 5-class sleep stage classifier
- TransitionDetectionHead: Binary transition detector
- TASAModel: Complete multi-task architecture
- UncertaintyLossWrapper: Homoscedastic uncertainty weighting
"""

from .tasa import (
    SparseAttention,
    SparseTransformerBackbone,
    SleepStagingHead,
    TransitionDetectionHead,
    TASAModel,
    UncertaintyLossWrapper,
)

__all__ = [
    "SparseAttention",
    "SparseTransformerBackbone",
    "SleepStagingHead",
    "TransitionDetectionHead",
    "TASAModel",
    "UncertaintyLossWrapper",
]
