import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_freq = cfg.MAX_FREQ
        self.dimensionality = cfg.DIMENSIONALITY
        self.num_bands = math.floor(cfg.FEAT_SIZE / cfg.DIMENSIONALITY / 2)
        self.base = cfg.BASE
        pad = cfg.FEAT_SIZE - self.num_bands * 2 * cfg.DIMENSIONALITY
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, _x):
        """
        _x [B,N,3]: batched point coordinates
        returns: [B,N,C]: positional encoding of dimension C
        """
        x = _x.clone()
        x[:, :, 0] = x[:, :, 0] / 48
        x[:, :, 1] = x[:, :, 1] / 48
        x[:, :, 2] = x[:, :, 2] / 4
        x = x.unsqueeze(-1)
        scales = torch.logspace(
            0.0,
            math.log(self.max_freq / 2) / math.log(self.base),
            self.num_bands,
            base=self.base,
            device=x.device,
            dtype=x.dtype,
        )
        # reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(2)
        enc = self.zero_pad(x)
        return enc


class DepthEncoder(nn.Module):
    """Depth component of 4D-Former's positional encoding (Sec. 3.2).

    Encodes the Euclidean distance of each feature from the LiDAR sensor with
    sine/cosine activations at various frequencies. The paper concatenates this
    with the Fourier xyz encoding; here it is produced at full feature size so
    it can be added to the xyz encoding (additive rather than concatenated,
    which keeps the decoder dims unchanged).
    """

    def __init__(self, cfg):
        super().__init__()
        feat_size = cfg.FEAT_SIZE
        self.num_bands = feat_size // 2
        self.max_freq = cfg.MAX_FREQ
        self.base = cfg.BASE
        pad = feat_size - self.num_bands * 2
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))

    def forward(self, coords):
        """coords [B,N,3] -> depth encoding [B,N,FEAT_SIZE]."""
        depth = torch.linalg.norm(coords, dim=-1, keepdim=True) / 48.0  # [B,N,1]
        scales = torch.logspace(
            0.0,
            math.log(self.max_freq / 2) / math.log(self.base),
            self.num_bands,
            base=self.base,
            device=coords.device,
            dtype=coords.dtype,
        )
        x = depth * scales * math.pi  # [B,N,num_bands]
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.zero_pad(x)
