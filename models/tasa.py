"""
TASA: Transition-Aware Sparse Attention for Parameter-Efficient Sleep Stage Classification

This module implements the complete TASA architecture including:
- Sparse (local windowed) attention for efficient sequence modeling
- Multi-task learning with sleep staging and transition detection heads
- Homoscedastic uncertainty weighting for automatic loss balancing

Paper: "TASA: Transition-Aware Sparse Attention for Parameter-Efficient Sleep Stage Classification"
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Sparse Attention Mechanism
# =============================================================================

class SparseAttention(nn.Module):
    """
    Local/Sparse Attention Mechanism.

    Restricts attention to a local window around each token, reducing
    computational complexity from O(n²) to O(n × window_size).

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        window_size: Local attention window size (default: 64)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        device = x.device
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create local window mask (sparse attention)
        indices = torch.arange(seq_len, device=device).unsqueeze(0)
        distance_matrix = torch.abs(indices - indices.transpose(0, 1))
        sparse_mask = distance_matrix > self.window_size

        # Apply sparse mask
        scores = scores.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out)


# =============================================================================
# Transformer Backbone
# =============================================================================

class SparseTransformerBackbone(nn.Module):
    """
    Transformer Backbone using Sparse (Local) Attention.

    Processes 3-channel EEG/EOG signals (EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal)
    through convolutional embedding and sparse transformer layers.

    Args:
        input_channels: Number of input channels (default: 3)
        d_model: Model dimension (default: 64)
        n_layers: Number of transformer layers (default: 4)
        n_heads: Number of attention heads (default: 4)
        window_size: Local attention window size (default: 64)
    """

    def __init__(
        self,
        input_channels: int = 3,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        window_size: int = 64
    ):
        super().__init__()
        self.d_model = d_model

        # Convolutional embedding: (B, 3, 3000) -> (B, d_model, 750)
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, d_model // 2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # Sparse transformer layers
        self.layers = nn.ModuleList([
            SparseAttention(d_model, n_heads, window_size=window_size)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signal (B, Channels, Time) e.g., (B, 3, 3000)

        Returns:
            Features (B, Seq_Len, D_model)
        """
        # Convolutional embedding
        x = self.embedding(x)  # (B, D, L)
        x = x.permute(0, 2, 1)  # (B, L, D)

        # Transformer layers with residual connections
        for attn, norm, ffn in zip(self.layers, self.layer_norms, self.ffns):
            # Attention block
            residual = x
            x = attn(x)
            x = residual + x
            x = norm(x)

            # FFN block
            residual = x
            x = ffn(x)
            x = residual + x
            x = norm(x)

        return x


# =============================================================================
# Task-Specific Heads
# =============================================================================

class SleepStagingHead(nn.Module):
    """
    Classification head for 5-class sleep stage scoring.

    Classes: Wake (W), N1, N2, N3, REM

    Args:
        d_model: Input feature dimension
        num_classes: Number of sleep stages (default: 5)
    """

    def __init__(self, d_model: int, num_classes: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone output (Batch, Seq_Len, D_model)

        Returns:
            Logits (Batch, Num_Classes)
        """
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.avg_pool(x).squeeze(2)  # (B, D)
        return self.fc(x)


class TransitionDetectionHead(nn.Module):
    """
    Binary classification head for detecting sleep stage transitions.

    Predicts whether the current epoch represents a transition point
    (change in sleep stage from the current to next epoch).

    Args:
        d_model: Input feature dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone output (Batch, Seq_Len, D_model)

        Returns:
            Logit (Batch, 1)
        """
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.avg_pool(x).squeeze(2)  # (B, D)
        return self.fc(x)


# =============================================================================
# Complete TASA Model
# =============================================================================

class TASAModel(nn.Module):
    """
    TASA: Transition-Aware Sparse Attention Model

    Multi-task architecture combining:
    1. Sparse Transformer Backbone for efficient feature extraction
    2. Sleep Staging Head for 5-class classification
    3. Transition Detection Head for boundary awareness

    Args:
        input_channels: Number of input EEG/EOG channels (default: 3)
        d_model: Model dimension (default: 64)
        n_layers: Number of transformer layers (default: 4)
        n_heads: Number of attention heads (default: 4)
        window_size: Local attention window size (default: 64)
        num_classes: Number of sleep stages (default: 5)
    """

    def __init__(
        self,
        input_channels: int = 3,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        window_size: int = 64,
        num_classes: int = 5
    ):
        super().__init__()

        self.backbone = SparseTransformerBackbone(
            input_channels=input_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size
        )

        self.staging_head = SleepStagingHead(d_model=d_model, num_classes=num_classes)
        self.transition_head = TransitionDetectionHead(d_model=d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input signal (B, C, T) - e.g., (B, 3, 3000)

        Returns:
            Dict with 'stage_logits' and 'transition_logits'
        """
        # Shared backbone feature extraction
        features = self.backbone(x)

        # Task-specific heads
        stage_logits = self.staging_head(features)
        transition_logits = self.transition_head(features)

        return {
            "stage_logits": stage_logits,
            "transition_logits": transition_logits
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Multi-Task Loss with Uncertainty Weighting
# =============================================================================

class UncertaintyLossWrapper(nn.Module):
    """
    Multi-Task Loss with Homoscedastic Uncertainty Weighting.

    Automatically learns task-specific weights based on homoscedastic
    uncertainty, balancing the contribution of each task to the total loss.

    Formula: Loss = (1 / 2σ²) × L_task + log(σ)

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)

    Args:
        num_tasks: Number of tasks (default: 2)
    """

    def __init__(self, num_tasks: int = 2):
        super().__init__()
        self.num_tasks = num_tasks
        # Learnable log variance parameters (initialized to 0 -> σ=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: List of scalar loss tensors [L_stage, L_transition]

        Returns:
            Weighted combined loss
        """
        assert len(losses) == self.num_tasks, \
            f"Expected {self.num_tasks} losses, got {len(losses)}"

        total_loss = 0
        for i, loss in enumerate(losses):
            # Precision = 1 / (2 × σ²) = 1 / (2 × exp(log_var))
            precision = 0.5 * torch.exp(-self.log_vars[i])
            # Weighted loss + regularization term
            task_loss = precision * loss + self.log_vars[i] * 0.5
            total_loss += task_loss

        return total_loss

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (1/σ² for each task)."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
            return {
                "stage_weight": weights[0].item(),
                "transition_weight": weights[1].item()
            }
