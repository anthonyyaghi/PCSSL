"""
SpUNet (Sparse UNet) backbone for scene-level point cloud encoding.

Adapted from Pointcept's SpUNet-v1m1 implementation by Xiaoyang Wu.
Original: https://github.com/Pointcept/Pointcept

Key adaptations for DBBD:
- Removed classification head (outputs raw decoder features)
- Added SpUNetSceneEncoder wrapper with voxelization/desparsification
- Removed Pointcept registry and torch_geometric dependencies
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from spconv.pytorch import ConvAlgo
from timm.layers import trunc_normal_


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SpUNet(nn.Module):
    """
    Sparse UNet for point cloud feature extraction.

    Encoder-decoder architecture with skip connections using sparse 3D convolutions.
    Outputs per-voxel features (no classification head).

    Args:
        in_channels: Input feature dimension (e.g., 3 for normals)
        output_dim: Output feature dimension per point
        base_channels: Initial convolution channels
        channels: Channel config per stage (must be even length: encoder + decoder)
        layers: Number of BasicBlocks per stage
    """

    def __init__(
        self,
        in_channels,
        output_dim=96,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                        algo=ConvAlgo.Native,
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )

            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        channels[len(channels) - s - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                        algo=ConvAlgo.Native,
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
                )
            )
            self.dec.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                (
                                    f"block{i}",
                                    block(
                                        dec_channels + enc_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                                if i == 0
                                else (
                                    f"block{i}",
                                    block(
                                        dec_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                            )
                            for i in range(layers[len(channels) - s - 1])
                        ]
                    )
                )
            )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        # Project final decoder output to desired output_dim if needed
        final_decoder_channels = channels[-1]
        if final_decoder_channels != output_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(final_decoder_channels, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.output_proj = nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, grid_coord, feat, offset):
        """
        Args:
            grid_coord: (N, 3) integer voxel coordinates
            feat: (N, in_channels) point features
            offset: (B,) cumulative point counts per batch

        Returns:
            (N, output_dim) per-voxel features
        """
        batch = offset2batch(offset)
        sparse_shape = torch.add(
            torch.max(grid_coord, dim=0).values, 96
        ).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input(x)
        skips = [x]

        for s in range(self.num_stages):
            if x.features.shape[0] == 0:
                break
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)

        x = skips.pop(-1)

        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(
                torch.cat((x.features, skip.features), dim=1)
            )
            x = self.dec[s](x)

        return self.output_proj(x.features)


class SpUNetSceneEncoder(nn.Module):
    """
    Scene-level encoder wrapping SpUNet with voxelization and desparsification.

    Handles the full pipeline: continuous coords → voxel grid → SpUNet → per-point features.

    Args:
        in_channels: Input feature dimension (e.g., 3 for normals)
        output_dim: Output feature dimension
        voxel_size: Voxel size in meters for quantization
        **spunet_kwargs: Additional arguments for SpUNet (base_channels, channels, layers)
    """

    def __init__(
        self,
        in_channels=3,
        output_dim=96,
        voxel_size=0.02,
        **spunet_kwargs,
    ):
        super().__init__()
        self.spunet = SpUNet(
            in_channels=in_channels,
            output_dim=output_dim,
            **spunet_kwargs,
        )
        self.voxel_size = voxel_size
        self.output_dim = output_dim

    def voxelize(self, coords, feats):
        """
        Convert continuous coordinates to voxel grid coordinates.

        Points mapping to the same voxel have their features averaged.

        Args:
            coords: (N, 3) continuous coordinates
            feats: (N, C) point features

        Returns:
            grid_coord: (V, 3) unique voxel integer coordinates
            voxel_feats: (V, C) averaged features per voxel
            inverse_map: (N,) index mapping each point to its voxel
        """
        grid_coord = torch.floor(coords / self.voxel_size).int()
        grid_coord -= grid_coord.min(0)[0]

        # Ravel multi-index to find unique voxels
        dims = grid_coord.max(0)[0] + 1
        raveled = (
            grid_coord[:, 0].long() * dims[1].long() * dims[2].long()
            + grid_coord[:, 1].long() * dims[2].long()
            + grid_coord[:, 2].long()
        )

        unique_raveled, inverse_map = torch.unique(raveled, return_inverse=True)

        # Average features per voxel using scatter
        num_voxels = unique_raveled.shape[0]
        voxel_feats = torch.zeros(
            num_voxels, feats.shape[1], device=feats.device, dtype=feats.dtype
        )
        counts = torch.zeros(num_voxels, device=feats.device, dtype=feats.dtype)
        voxel_feats.scatter_add_(0, inverse_map.unsqueeze(1).expand_as(feats), feats)
        counts.scatter_add_(0, inverse_map, torch.ones(coords.shape[0], device=coords.device, dtype=feats.dtype))
        voxel_feats = voxel_feats / counts.unsqueeze(1).clamp(min=1)

        # Recover grid coords for unique voxels (pick first occurrence)
        voxel_grid_coord = torch.zeros(
            num_voxels, 3, device=coords.device, dtype=torch.int32
        )
        # Use scatter with index to pick coordinates for each unique voxel
        perm = torch.arange(inverse_map.shape[0], device=coords.device)
        # For each voxel, take the first point's grid coord
        first_occurrence = torch.zeros(
            num_voxels, device=coords.device, dtype=torch.long
        )
        first_occurrence.scatter_(0, inverse_map, perm)
        voxel_grid_coord = grid_coord[first_occurrence]

        return voxel_grid_coord, voxel_feats, inverse_map

    def desparsify(self, voxel_feats, inverse_map):
        """
        Map per-voxel features back to per-point features.

        Args:
            voxel_feats: (V, D) features per voxel
            inverse_map: (N,) maps each point to its voxel index

        Returns:
            (N, D) per-point features
        """
        return voxel_feats[inverse_map]

    def forward(self, coords, feats, offset):
        """
        Encode a batch of scenes through SpUNet.

        Each scene is voxelized independently (different grid origins),
        then all voxels are processed together through SpUNet.

        Args:
            coords: (N_total, 3) concatenated scene coordinates
            feats: (N_total, C) concatenated raw features
            offset: (B+1,) batch boundary indices

        Returns:
            (N_total, output_dim) per-point features
        """
        batch_size = offset.shape[0] - 1
        device = coords.device

        all_grid_coords = []
        all_voxel_feats = []
        all_inverse_maps = []
        voxel_counts = []

        for i in range(batch_size):
            start = offset[i].item()
            end = offset[i + 1].item()

            scene_coords = coords[start:end]
            scene_feats = feats[start:end]

            grid_coord, voxel_feats, inverse_map = self.voxelize(
                scene_coords, scene_feats
            )

            all_grid_coords.append(grid_coord)
            all_voxel_feats.append(voxel_feats)
            all_inverse_maps.append(inverse_map)
            voxel_counts.append(grid_coord.shape[0])

        cat_grid_coord = torch.cat(all_grid_coords, dim=0)
        cat_voxel_feats = torch.cat(all_voxel_feats, dim=0)

        # Build voxel-level offset for SpUNet batching
        voxel_offset = torch.cumsum(
            torch.tensor(voxel_counts, device=device, dtype=torch.long), dim=0
        )

        # Run SpUNet on all voxels
        spunet_out = self.spunet(cat_grid_coord, cat_voxel_feats, voxel_offset)

        # Desparsify: map voxel features back to original points
        result_parts = []
        voxel_start = 0
        for i in range(batch_size):
            voxel_end = voxel_start + voxel_counts[i]
            scene_voxel_feats = spunet_out[voxel_start:voxel_end]
            point_feats = self.desparsify(scene_voxel_feats, all_inverse_maps[i])
            result_parts.append(point_feats)
            voxel_start = voxel_end

        return torch.cat(result_parts, dim=0)
