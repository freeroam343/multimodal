"""LiDAR <-> camera projection utilities for 4D-Former multimodal fusion.

The projection convention matches the dataset's `calib.txt`
(see current_polar/scripts/project_labels_to_image.py):

    pixel = P2 @ R_rect_00 @ Tr_velo_to_cam @ R_axes @ point_h

where LiDAR is (X-forward, Y-left, Z-up) and R_axes rotates LiDAR axes into
camera axes (X-right, Y-down, Z-forward). All matrices are 4x4 homogeneous.

Two responsibilities live here:
  1. Parsing calib.txt into the composed 4x4 projection matrix `T`.
  2. Projecting 3D points to (u, v, depth) and bilinearly sampling
     image / FPN feature maps at those locations to build per-point image
     features (used by both point-level fusion and the decoder's image
     cross-attention).
"""

import numpy as np
import torch
import torch.nn.functional as F

# LiDAR (X-fwd, Y-left, Z-up) -> camera (X-right, Y-down, Z-fwd)
#   cam_x = -lidar_y, cam_y = -lidar_z, cam_z = lidar_x
R_AXES = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def parse_calibration(filename):
    """Parse a KITTI-style calib.txt into a dict of 4x4 matrices."""
    calib = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, content = line.split(":", 1)
            values = [float(v) for v in content.strip().split()]
            if len(values) == 12:
                mat = np.eye(4)
                mat[0, 0:4] = values[0:4]
                mat[1, 0:4] = values[4:8]
                mat[2, 0:4] = values[8:12]
            elif len(values) == 9:
                mat = np.eye(4)
                mat[0, 0:3] = values[0:3]
                mat[1, 0:3] = values[3:6]
                mat[2, 0:3] = values[6:9]
            elif len(values) == 16:
                mat = np.array(values).reshape(4, 4)
            else:
                continue
            calib[key.strip()] = mat
    return calib


def projection_matrix(calib):
    """Compose the full 4x4 LiDAR->image projection matrix from a calib dict."""
    P = calib.get("P2", calib.get("P0", np.eye(4)))
    R_rect = calib.get("R_rect_00", np.eye(4))
    Tr = calib.get("Tr_velo_to_cam", calib.get("Tr", np.eye(4)))
    return (P @ R_rect @ Tr @ R_AXES).astype(np.float32)


def adjust_projection_for_resize(T, sx, sy):
    """Scale the pixel-space rows of T for an image resized by (sx, sy)."""
    S = np.diag([sx, sy, 1.0, 1.0]).astype(T.dtype)
    return S @ T


def adjust_projection_for_crop(T, x0, y0):
    """Offset the pixel-space rows of T for a crop whose top-left is (x0, y0)."""
    C = np.eye(4, dtype=T.dtype)
    C[0, 2] = -x0  # note: applied in homogeneous pixel coords (post divide handled below)
    C[1, 2] = -y0
    # Crop acts on *pixel* coordinates after the perspective divide. We fold it
    # into T by subtracting (x0, y0) * w from the first two rows, where w is the
    # third (depth) row, so that after dividing by depth the offset is exact.
    T = T.copy()
    T[0, :] -= x0 * T[2, :]
    T[1, :] -= y0 * T[2, :]
    return T


def project_points(xyz, T):
    """Project (N,3) LiDAR points with a 4x4 matrix T.

    Returns (uv, depth): uv is (N,2) float pixel coords, depth is (N,) camera-Z.
    Works for both numpy arrays and torch tensors (kept in their framework).
    """
    if isinstance(xyz, np.ndarray):
        n = xyz.shape[0]
        pts_h = np.concatenate([xyz, np.ones((n, 1), dtype=xyz.dtype)], axis=1)
        proj = (T @ pts_h.T)  # (4, N)
        depth = proj[2]
        uv = (proj[:2] / (depth + 1e-8)).T
        return uv, depth
    else:
        Tt = torch.as_tensor(T, dtype=xyz.dtype, device=xyz.device)
        n = xyz.shape[0]
        ones = torch.ones((n, 1), dtype=xyz.dtype, device=xyz.device)
        pts_h = torch.cat([xyz, ones], dim=1)
        proj = (Tt @ pts_h.T)  # (4, N)
        depth = proj[2]
        uv = (proj[:2] / (depth + 1e-8)).t()
        return uv, depth


def valid_projection_mask(uv, depth, width, height, min_depth=0.1):
    """Boolean mask of points that land in front of and inside the image."""
    if isinstance(uv, np.ndarray):
        return (
            (depth > min_depth)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
    return (
        (depth > min_depth)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )


def sample_feature_map(feat_map, uv, stride):
    """Bilinearly sample a feature map at LiDAR-projected pixel locations.

    Args:
        feat_map: (1, C, Hs, Ws) feature map at the given `stride`.
        uv: (N, 2) pixel coordinates in the *full-resolution* image frame.
        stride: int downsampling factor of `feat_map` w.r.t. the full image.

    Returns:
        (N, C) sampled features (zeros where uv falls outside the map).
    """
    _, c, hs, ws = feat_map.shape
    # full-res pixel -> feature-map pixel
    fx = uv[:, 0] / stride
    fy = uv[:, 1] / stride
    # normalize to [-1, 1] for grid_sample (align_corners=False convention)
    gx = (fx + 0.5) / ws * 2 - 1
    gy = (fy + 0.5) / hs * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2).to(feat_map.dtype)
    sampled = F.grid_sample(
        feat_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (1, C, 1, N)
    return sampled.view(c, -1).t()  # (N, C)
