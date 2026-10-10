"""CDA-UNet: channel- and gate-conditioned dynamic residual adapters.

The encoder uses a separate batch-normalization bank for each unique
``(parent, x channel, y channel)`` input group. Decoder normalization is
gate-specific. All convolutions, adapters, embeddings and the binary output
head are shared within a panel. Outputs are logits, not probabilities.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class BatchNormBank(nn.Module):
    """Independent affine parameters and running statistics for each domain."""

    def __init__(self, channels: int, domains: int):
        super().__init__()
        self.banks = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(domains))

    def forward(self, features: torch.Tensor, domain: int) -> torch.Tensor:
        return self.banks[domain](features)


class DoubleConv(nn.Module):
    """Two bias-free 3 x 3 convolution, BN and ReLU operations."""

    def __init__(self, in_channels: int, out_channels: int, domains: int):
        super().__init__()
        operations = []
        for incoming_width in (in_channels, out_channels):
            operations.extend((
                nn.Conv2d(incoming_width, out_channels, 3, padding=1, bias=False),
                BatchNormBank(out_channels, domains),
                nn.ReLU(inplace=True),
            ))
        self.conv = nn.Sequential(*operations)

    def forward(self, features: torch.Tensor, domain: int) -> torch.Tensor:
        for layer in self.conv:
            features = layer(features, domain) if isinstance(layer, BatchNormBank) else layer(features)
        return features


class DynamicAdapter(nn.Module):
    """Decoder DoubleConv and a condition-generated narrow residual branch.

    ``rank`` denotes the adapter width. The condition generator supplies a
    rank-by-rank 3 x 3 kernel, a bias and GroupNorm scale/shift corrections.
    The final 1 x 1 projection starts at zero, so the residual initially
    vanishes while its output projection receives gradients on the first step.
    """

    def __init__(self, in_channels: int, out_channels: int, rank: int, domains: int):
        super().__init__()
        self.rank = rank
        self.kernel_size = 9 * rank * rank
        self.generated_parameter_count = self.kernel_size + 3 * rank
        self.base = DoubleConv(in_channels, out_channels, domains)
        self.down = nn.Conv2d(out_channels, rank, 1, bias=False)
        self.norm = nn.GroupNorm(8, rank, affine=False)
        self.up = nn.Conv2d(rank, out_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def adapter(self, features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Apply one condition-generated kernel shared by a homogeneous batch."""
        if parameters.shape != (1, self.generated_parameter_count):
            raise ValueError("Each adapter call requires one generated condition")
        kernel = parameters[0, :self.kernel_size].reshape(self.rank, self.rank, 3, 3)
        bias, delta_gamma, beta = parameters[0, self.kernel_size:].split(self.rank)
        features = F.conv2d(self.down(features), kernel, bias, padding=1)
        features = (1 + delta_gamma[None, :, None, None]) * self.norm(features) + beta[None, :, None, None]
        return self.up(F.relu(features))


class CDAUNet(nn.Module):
    """One conditional binary segmentation model for all gates in a panel.

    Parameters
    ----------
    panel:
        Panel schema with ``channels`` and ``gates``. Each gate defines
        ``name``, ``parent``, ``x`` and ``y``; see :mod:`cda_unet.schema`.
    rank:
        Width of the dynamic adapters (32 in the released configuration).
    features:
        Four encoder widths, defaulting to 64, 128, 256 and 512.

    A training batch must contain one gate. ``gate_ids`` accepts an integer,
    a one-element integer tensor or a one-dimensional tensor with one index
    per image. Mixed-gate batches are supported in evaluation mode; each
    sample retains its original position in the result. Input images have
    shape ``(batch, 1, height, width)``. Output has the same shape and uses
    logits; the research decision rule is strictly ``logits > 0``.
    """

    def __init__(self, panel, rank: int = 32, features: Sequence[int] = (64, 128, 256, 512)):
        super().__init__()
        features = tuple(features)
        if (len(features) != 4 or any(type(c) is not int or c < 1 for c in features)
                or any(b != 2 * a for a, b in zip(features, features[1:]))):
            raise ValueError("features must be four positive, successively doubled integers")
        if type(rank) is not int or rank <= 0 or rank % 8:
            raise ValueError("rank must be a positive integer divisible by eight")
        channels = tuple(panel.channels)
        gates = tuple(panel.gates)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("panel channels must be nonempty and unique")
        if not gates or len({g.name for g in gates}) != len(gates):
            raise ValueError("panel gates must be nonempty and uniquely named")
        channel_indices = {name: index for index, name in enumerate(channels)}
        if any(g.x not in channel_indices or g.y not in channel_indices for g in gates):
            raise ValueError("Every gate channel must occur in panel.channels")
        self.input_groups = tuple(dict.fromkeys((g.parent, g.x, g.y) for g in gates))
        group_lookup = {key: index for index, key in enumerate(self.input_groups)}
        group_ids = [group_lookup[(g.parent, g.x, g.y)] for g in gates]
        self.panel = panel
        self.features = features
        self.rank = rank
        self.embedding_dim = 32
        self.hidden_dim = 128
        self.num_gates = len(gates)
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        in_channels = 1
        for channels_count in features:
            self.downs.append(DoubleConv(in_channels, channels_count, len(self.input_groups)))
            in_channels = channels_count
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, len(self.input_groups))
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.stage_heads = nn.ModuleList()
        for channels_count in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(channels_count * 2, channels_count, 2, 2))
            block = DynamicAdapter(channels_count * 2, channels_count, rank, len(gates))
            head = nn.Linear(self.hidden_dim, block.generated_parameter_count)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight[:block.kernel_size], std=math.sqrt(2 / (128 * 9 * rank)))
            self.decoder_blocks.append(block)
            self.stage_heads.append(head)
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)
        self.marker_embedding = nn.Embedding(len(channels), self.embedding_dim)
        self.gate_embedding = nn.Embedding(len(gates), self.embedding_dim)
        self.condition_stem = nn.Sequential(nn.Linear(3 * self.embedding_dim, self.hidden_dim), nn.ReLU())
        nn.init.kaiming_normal_(self.condition_stem[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.condition_stem[0].bias)
        self.register_buffer("gate_marker_indices", torch.tensor(
            [[channel_indices[g.x], channel_indices[g.y]] for g in gates], dtype=torch.long))
        self.register_buffer("gate_input_groups", torch.tensor(group_ids, dtype=torch.long))

    def _gate_ids(self, gate_ids, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(gate_ids, Integral) and not isinstance(gate_ids, bool):
            gate_ids = torch.tensor([int(gate_ids)], dtype=torch.long, device=device)
        if not isinstance(gate_ids, torch.Tensor):
            raise TypeError("gate_ids must be an integer or an integer tensor")
        if gate_ids.ndim != 1 or gate_ids.numel() not in (1, batch):
            raise ValueError("gate_ids must be one-dimensional, with one or batch indices")
        if gate_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("gate_ids must use int32 or int64")
        gate_ids = gate_ids.to(device=device, dtype=torch.long)
        if int(gate_ids.min()) < 0 or int(gate_ids.max()) >= self.num_gates:
            raise ValueError(f"gate_ids must be between 0 and {self.num_gates - 1}")
        if self.training and gate_ids.unique().numel() != 1:
            raise ValueError("Training batches must contain a single gate")
        return gate_ids.expand(batch) if gate_ids.numel() == 1 else gate_ids

    def _forward_gate(self, images: torch.Tensor, gate: int) -> torch.Tensor:
        group = int(self.gate_input_groups[gate])
        features, skips = images, []
        for block in self.downs:
            features = block(features, group)
            skips.append(features)
            features = self.pool(features)
        features = self.bottleneck(features, group)
        ids = torch.tensor([gate], device=images.device, dtype=torch.long)
        # Keep hypernetwork calculations in float32 under mixed precision,
        # matching the reference training implementation.
        with torch.autocast(device_type=images.device.type, enabled=False):
            markers = self.marker_embedding(self.gate_marker_indices[ids])
            condition = self.condition_stem(torch.cat(
                (markers[:, 0], markers[:, 1], self.gate_embedding(ids)), dim=1))
            parameters = [head(condition) for head in self.stage_heads]
        for stage, (upconv, block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            features = upconv(features)
            skip = skips[-1 - stage]
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear",
                                         align_corners=False, antialias=False)
            features = block.base(torch.cat((skip, features), dim=1), gate)
            features = features + block.adapter(features, parameters[stage])
        return self.final_conv(features)

    def forward(self, images: torch.Tensor, gate_ids) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape batch x 1 x height x width")
        if images.shape[0] == 0 or images.shape[1] != 1 or not images.is_floating_point():
            raise ValueError("images must be nonempty floating-point single-channel tensors")
        if min(images.shape[-2:]) < 16:
            raise ValueError("Each image dimension must be at least 16 for four pooling stages")
        ids = self._gate_ids(gate_ids, images.shape[0], images.device)
        gates = ids.unique(sorted=True).tolist()
        if len(gates) == 1:
            return self._forward_gate(images, gates[0])
        parts, positions = [], []
        for gate in gates:
            indices = torch.nonzero(ids == gate, as_tuple=True)[0]
            parts.append(self._forward_gate(images.index_select(0, indices), gate))
            positions.append(indices)
        return torch.cat(parts).index_select(0, torch.cat(positions).argsort())
