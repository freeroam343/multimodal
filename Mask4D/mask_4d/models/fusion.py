"""Point-level LiDAR<->camera fusion (4D-Former Sec 3.1, Eq. 1 & Eq. 4).

Given per-point LiDAR features and the highest-resolution image feature map
I4, we project each point into the image, bilinearly sample I4, and fuse:

    Z+_lidar  <- MLP_fusion([Z+_lidar, Z_img])     # projectable points
    Z-_lidar  <- MLP_pseudo(Z-_lidar)              # non-projectable points

and supervise the pseudo path so it learns to imitate fused features:

    L_pf = || MLP_fusion([Z+_lidar, Z_img]) - MLP_pseudo(Z+_lidar) ||_2   (Eq. 4)

Both MLPs are 3-layer. The fusion is applied to whatever LiDAR point-feature
tensor it is handed, so it can be hooked at intermediate backbone stages and/or
at the final point features Z.
"""

import torch
import torch.nn as nn

from mask_4d.models.blocks import MLP
from mask_4d.utils import projection as proj


class PointImageFusion(nn.Module):
    def __init__(self, dim, img_stride=4):
        super().__init__()
        self.dim = dim
        self.img_stride = img_stride  # I4 is the highest-res FPN level (stride 4)
        # 3-layer MLPs (paper: "both MLPs contain 3 layers")
        self.mlp_fusion = MLP(2 * dim, dim, dim, 3)
        self.mlp_pseudo = MLP(dim, dim, dim, 3)

    def forward(self, point_feats, xyz, image_feat_s4, T, img_hw):
        """Fuse a single point cloud's features with its image features.

        Args:
            point_feats: (N, D) LiDAR point features.
            xyz: (N, 3) LiDAR coordinates (same order as point_feats).
            image_feat_s4: (1, D, H/4, W/4) image feature map I4, or None.
            T: (4,4) projection matrix for this frame, or None.
            img_hw: (H, W) full-resolution image size, or None.

        Returns:
            fused: (N, D) fused point features.
            pf_loss: scalar pseudo-fusion L2 loss (0 if no image / not training).
        """
        if image_feat_s4 is None or T is None:
            # No image available -> pseudo path for everything, no pf loss.
            return self.mlp_pseudo(point_feats), point_feats.new_zeros(())

        h, w = img_hw
        uv, depth = proj.project_points(xyz, T)
        valid = proj.valid_projection_mask(uv, depth, w, h)

        fused = torch.empty_like(point_feats)
        pf_loss = point_feats.new_zeros(())

        if valid.any():
            z_plus = point_feats[valid]  # (M, D)
            z_img = proj.sample_feature_map(image_feat_s4, uv[valid], self.img_stride)
            fused_plus = self.mlp_fusion(torch.cat([z_plus, z_img], dim=-1))
            fused[valid] = fused_plus
            # Eq. 4: pseudo path imitates fusion on the projectable subset.
            if self.training:
                pseudo_plus = self.mlp_pseudo(z_plus)
                pf_loss = (fused_plus.detach() - pseudo_plus).pow(2).sum(-1).sqrt().mean()

        if (~valid).any():
            fused[~valid] = self.mlp_pseudo(point_feats[~valid])

        return fused, pf_loss
