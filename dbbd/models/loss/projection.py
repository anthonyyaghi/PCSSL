"""
Contrastive Projection Head

Projects encoder features to a space optimized for contrastive learning.
Standard practice in SimCLR, MoCo, PointContrast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Projects encoder features to a lower-dimensional space optimized for
    contrastive learning. The output is L2-normalized for cosine similarity.
    
    Architecture:
        For num_layers=2: Linear -> BN -> ReLU -> Linear -> L2-normalize
        For num_layers=1: Linear -> L2-normalize
    
    Args:
        input_dim: Dimension of input features (encoder output)
        hidden_dim: Dimension of hidden layer (only used if num_layers > 1)
        output_dim: Dimension of projected features
        num_layers: Number of layers (1 or 2)
    
    Note:
        Output is always L2-normalized. This is critical for InfoNCE loss
        which uses cosine similarity.
    """
    
    def __init__(
        self,
        input_dim: int = 96,
        hidden_dim: int = 96,
        output_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        
        if num_layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            # First layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Use LayerNorm instead of BatchNorm to handle batch_size=1
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            # Middle layers (if any)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            
            # Final layer (no norm/ReLU before L2 normalization)
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.layers = nn.Sequential(*layers)
        
        logger.debug(
            f"ContrastiveProjectionHead: {input_dim} -> {output_dim} "
            f"(hidden={hidden_dim}, layers={num_layers})"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project and normalize features.
        
        Args:
            x: Input features (N, input_dim)
            
        Returns:
            L2-normalized projected features (N, output_dim)
        """
        projected = self.layers(x)
        # L2 normalize - critical for cosine similarity
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized
    
    def __repr__(self):
        return (
            f"ContrastiveProjectionHead("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_layers={self.num_layers})"
        )
