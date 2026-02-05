"""
Projection MLP

Projects raw point features to encoder input dimension.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ProjectionMLP(nn.Module):
    """
    MLP that projects raw point features to encoder input dimension.
    
    Used to standardize input dimension before encoding, and to combine
    raw features with propagated context in G2L branch.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension (default 96)
        hidden_dim: Hidden layer dimension (default: output_dim)
        num_layers: Number of MLP layers (default 2)
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 96,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim
        self.num_layers = num_layers
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else self.hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Add normalization and activation except for last layer
            if i < num_layers - 1:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input features.
        
        Args:
            x: (M, input_dim) input features
            
        Returns:
            (M, output_dim) projected features
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {x.shape[-1]}"
            )
        
        return self.mlp(x)
    
    def __repr__(self):
        return (
            f"ProjectionMLP(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers})"
        )


class CombineProjection(nn.Module):
    """
    Projects concatenated features (raw + propagated) to encoder input.
    
    Used in G2L branch to combine raw features with parent context.
    
    Args:
        raw_dim: Dimension of raw features
        context_dim: Dimension of propagated context
        output_dim: Output dimension
    """
    
    def __init__(
        self,
        raw_dim: int,
        context_dim: int,
        output_dim: int = 96,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.raw_dim = raw_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        
        # Combined input dimension
        combined_dim = raw_dim + context_dim
        
        self.projection = ProjectionMLP(
            input_dim=combined_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim or output_dim,
            num_layers=num_layers
        )
    
    def forward(
        self,
        raw_feats: torch.Tensor,
        context_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine raw and context features, then project.
        
        Args:
            raw_feats: (M, raw_dim) raw point features
            context_feats: (M, context_dim) propagated context features
            
        Returns:
            (M, output_dim) projected combined features
        """
        if raw_feats.shape[0] != context_feats.shape[0]:
            raise ValueError(
                f"Feature count mismatch: raw {raw_feats.shape[0]}, "
                f"context {context_feats.shape[0]}"
            )
        
        combined = torch.cat([raw_feats, context_feats], dim=-1)
        return self.projection(combined)
    
    def __repr__(self):
        return (
            f"CombineProjection(raw_dim={self.raw_dim}, "
            f"context_dim={self.context_dim}, "
            f"output_dim={self.output_dim})"
        )
