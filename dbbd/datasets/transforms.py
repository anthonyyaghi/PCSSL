"""
Data Augmentation Transforms for DBBD

Geometric and appearance transformations for point clouds.
All transforms are hierarchy-invariant (indices remain valid).
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List):
        """
        Initialize composed transforms.
        
        Args:
            transforms: List of transform objects
        """
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {}'.format(t)
        format_string += '\n)'
        return format_string


class RandomRotate:
    """
    Random rotation around specified axis.
    
    For indoor scenes, typically rotate around Z-axis (vertical).
    """
    
    def __init__(
        self,
        angle: Optional[Tuple[float, float]] = None,
        axis: str = 'z',
        p: float = 1.0
    ):
        """
        Initialize random rotation.
        
        Args:
            angle: Rotation angle range in degrees [min, max]. 
                   If None, uses [-180, 180]
            axis: Rotation axis ('x', 'y', or 'z')
            p: Probability of applying transform
        """
        self.angle = angle or [-180, 180]
        self.axis = axis.lower()
        self.p = p
        
        if self.axis not in ['x', 'y', 'z']:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{self.axis}'")
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random rotation to coordinates."""
        if np.random.rand() > self.p:
            return data
        
        # Sample rotation angle
        angle_deg = np.random.uniform(self.angle[0], self.angle[1])
        angle_rad = np.deg2rad(angle_deg)
        
        # Create rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        if self.axis == 'z':
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        elif self.axis == 'y':
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float32)
        else:  # x
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ], dtype=np.float32)
        
        # Apply rotation to coordinates
        coord = data['coord']
        if isinstance(coord, torch.Tensor):
            coord = coord.numpy()
        
        data['coord'] = coord @ rotation_matrix.T
        
        # Also rotate normals if present
        if 'normal' in data:
            normal = data['normal']
            if isinstance(normal, torch.Tensor):
                normal = normal.numpy()
            data['normal'] = normal @ rotation_matrix.T
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angle={self.angle}, axis='{self.axis}', p={self.p})"


class RandomScale:
    """Random uniform scaling."""
    
    def __init__(
        self,
        scale: Tuple[float, float] = (0.8, 1.2),
        anisotropic: bool = False,
        p: float = 1.0
    ):
        """
        Initialize random scaling.
        
        Args:
            scale: Scale range [min, max]
            anisotropic: If True, scale each axis independently
            p: Probability of applying transform
        """
        self.scale = scale
        self.anisotropic = anisotropic
        self.p = p
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random scaling to coordinates."""
        if np.random.rand() > self.p:
            return data
        
        coord = data['coord']
        if isinstance(coord, torch.Tensor):
            coord = coord.numpy()
        
        if self.anisotropic:
            # Different scale per axis
            scale_factor = np.random.uniform(self.scale[0], self.scale[1], size=3)
        else:
            # Uniform scaling
            scale_factor = np.random.uniform(self.scale[0], self.scale[1])
        
        data['coord'] = coord * scale_factor
        
        return data
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(scale={self.scale}, "
                f"anisotropic={self.anisotropic}, p={self.p})")


class RandomTranslation:
    """Random translation (jitter)."""
    
    def __init__(
        self,
        shift: float = 0.2,
        p: float = 1.0
    ):
        """
        Initialize random translation.
        
        Args:
            shift: Maximum shift in each direction
            p: Probability of applying transform
        """
        self.shift = shift
        self.p = p
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random translation to coordinates."""
        if np.random.rand() > self.p:
            return data
        
        coord = data['coord']
        if isinstance(coord, torch.Tensor):
            coord = coord.numpy()
        
        # Sample translation vector
        translation = np.random.uniform(-self.shift, self.shift, size=3)
        
        data['coord'] = coord + translation
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shift={self.shift}, p={self.p})"


class RandomDropout:
    """
    Randomly drop points to simulate sensor noise.
    
    Note: This modifies the number of points, so indices in hierarchy
    will need to be remapped. Use with caution in hierarchical settings.
    """
    
    def __init__(
        self,
        dropout_ratio: Tuple[float, float] = (0.0, 0.2),
        p: float = 0.5
    ):
        """
        Initialize random dropout.
        
        Args:
            dropout_ratio: Range of dropout ratio [min, max]
            p: Probability of applying transform
        """
        self.dropout_ratio = dropout_ratio
        self.p = p
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random point dropout."""
        if np.random.rand() > self.p:
            return data
        
        coord = data['coord']
        if isinstance(coord, torch.Tensor):
            coord = coord.numpy()
        
        num_points = len(coord)
        dropout_ratio = np.random.uniform(self.dropout_ratio[0], self.dropout_ratio[1])
        num_keep = int(num_points * (1 - dropout_ratio))
        
        # Randomly sample points to keep
        keep_indices = np.random.choice(num_points, num_keep, replace=False)
        keep_indices = np.sort(keep_indices)
        
        # Apply to coord and feat
        data['coord'] = coord[keep_indices]
        if 'feat' in data:
            feat = data['feat']
            if isinstance(feat, torch.Tensor):
                feat = feat.numpy()
            data['feat'] = feat[keep_indices]
        
        # WARNING: This breaks hierarchy indices!
        # Only use if you're not using hierarchies or can remap indices
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dropout_ratio={self.dropout_ratio}, p={self.p})"


class ColorJitter:
    """Add random color jitter to RGB features."""
    
    def __init__(
        self,
        std: float = 0.05,
        clip: bool = True,
        p: float = 1.0
    ):
        """
        Initialize color jitter.
        
        Args:
            std: Standard deviation of Gaussian noise
            clip: Whether to clip values to [0, 1]
            p: Probability of applying transform
        """
        self.std = std
        self.clip = clip
        self.p = p
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply color jitter to features."""
        if np.random.rand() > self.p:
            return data
        
        if 'feat' not in data:
            return data
        
        feat = data['feat']
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        
        # Assume first 3 channels are RGB
        if feat.shape[1] >= 3:
            noise = np.random.randn(feat.shape[0], 3) * self.std
            feat[:, :3] = feat[:, :3] + noise
            
            if self.clip:
                feat[:, :3] = np.clip(feat[:, :3], 0, 1)
            
            data['feat'] = feat
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std}, clip={self.clip}, p={self.p})"


class ChromaticAutoContrast:
    """
    Chromatic auto-contrast normalization.
    
    Normalizes color channels to use full [0, 1] range.
    """
    
    def __init__(
        self,
        randomize_blend_factor: bool = True,
        blend_factor: float = 0.5,
        p: float = 1.0
    ):
        """
        Initialize chromatic auto-contrast.
        
        Args:
            randomize_blend_factor: If True, randomize blend with original
            blend_factor: Blend factor if not randomized
            p: Probability of applying transform
        """
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor
        self.p = p
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply chromatic auto-contrast."""
        if np.random.rand() > self.p:
            return data
        
        if 'feat' not in data:
            return data
        
        feat = data['feat']
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        
        # Assume first 3 channels are RGB
        if feat.shape[1] >= 3:
            rgb = feat[:, :3]
            
            # Normalize each channel to [0, 1]
            rgb_normalized = np.zeros_like(rgb)
            for i in range(3):
                channel = rgb[:, i]
                min_val, max_val = channel.min(), channel.max()
                if max_val > min_val:
                    rgb_normalized[:, i] = (channel - min_val) / (max_val - min_val)
                else:
                    rgb_normalized[:, i] = channel
            
            # Blend with original
            if self.randomize_blend_factor:
                alpha = np.random.rand()
            else:
                alpha = self.blend_factor
            
            feat[:, :3] = alpha * rgb_normalized + (1 - alpha) * rgb
            data['feat'] = feat
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(blend={self.blend_factor}, p={self.p})"


class NormalNoise:
    """Add Gaussian noise to normal vectors."""
    
    def __init__(
        self,
        std: float = 0.01,
        normalize: bool = True,
        p: float = 1.0
    ):
        """
        Initialize normal noise.
        
        Args:
            std: Standard deviation of Gaussian noise
            normalize: Whether to re-normalize normals after adding noise
            p: Probability of applying transform
        """
        self.std = std
        self.normalize = normalize
        self.p = p
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply noise to normal vectors."""
        if np.random.rand() > self.p:
            return data
        
        if 'normal' not in data:
            return data
        
        normal = data['normal']
        if isinstance(normal, torch.Tensor):
            normal = normal.numpy()
        
        # Add noise
        noise = np.random.randn(*normal.shape) * self.std
        normal = normal + noise
        
        # Re-normalize
        if self.normalize:
            norm = np.linalg.norm(normal, axis=1, keepdims=True)
            norm = np.where(norm > 0, norm, 1)  # Avoid division by zero
            normal = normal / norm
        
        data['normal'] = normal
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std}, normalize={self.normalize}, p={self.p})"


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert coord and feat to tensors."""
        if 'coord' in data and isinstance(data['coord'], np.ndarray):
            data['coord'] = torch.from_numpy(data['coord']).float()
        
        if 'feat' in data and isinstance(data['feat'], np.ndarray):
            data['feat'] = torch.from_numpy(data['feat']).float()
        
        if 'normal' in data and isinstance(data['normal'], np.ndarray):
            data['normal'] = torch.from_numpy(data['normal']).float()
        
        for key in ['segment', 'semantic_gt', 'instance_gt']:
            if key in data and isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key]).long()
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DualViewTransform:
    """
    Apply transforms twice with different random seeds for contrastive learning.
    
    Returns two differently augmented views of the same input.
    """
    
    def __init__(self, transform):
        """
        Initialize dual-view transform.
        
        Args:
            transform: Transform or Compose to apply twice
        """
        self.transform = transform
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transform twice with different random states.
        
        Returns:
            Dictionary with 'view1', 'view2', and 'scene_id'
        """
        import copy
        
        # Create deep copies for two views
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)
        
        # Apply transforms with different random states
        view1 = self.transform(data1)
        view2 = self.transform(data2)
        
        # Keep hierarchy only in view1 (they're identical, index-based)
        output = {
            'view1': view1,
            'view2': view2,
            'scene_id': data.get('scene_id', 'unknown')
        }
        
        return output
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  {self.transform}\n)"
