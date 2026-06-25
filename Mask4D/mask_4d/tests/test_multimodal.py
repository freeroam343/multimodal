"""CPU unit tests for 4D-Former multimodal modules.

These cover the pure-PyTorch additions (projection, image backbone, point-level
fusion, depth positional encoding, decoder image cross-attention, and the TAM).
The spconv LiDAR backbone needs CUDA and is not exercised here.

Run:  pytest mask_4d/tests/test_multimodal.py
"""

import types

import numpy as np
import torch
from easydict import EasyDict

from mask_4d.models.fusion import PointImageFusion
from mask_4d.models.image_backbone import ResNetFPN
from mask_4d.models.positional_encoder import DepthEncoder, PositionalEncoder
from mask_4d.models.tam import (
    TrackletAssociationModule,
    TrackMemoryBank,
    sincos_expand,
)
from mask_4d.utils import projection as proj

D = 256
N = 1500

CALIB = {
    "P2": np.array([[1334.05, 0, 986.73, 0], [0, 1331.99, 514.32, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    "R_rect_00": np.eye(4),
    "Tr_velo_to_cam": np.array([[1, 0, 0, -0.068], [0, 1, 0, -0.073], [0, 0, 1, 0.004], [0, 0, 0, 1]]),
}


def _xyz():
    rng = np.random.RandomState(0)
    xyz = (rng.randn(N, 3) * 10).astype(np.float32)
    xyz[:, 0] = np.abs(xyz[:, 0]) + 3.0
    return xyz


def test_projection_numpy_torch_agree():
    T = proj.projection_matrix(CALIB)
    xyz = _xyz()
    uv_np, _ = proj.project_points(xyz, T)
    uv_t, _ = proj.project_points(torch.from_numpy(xyz), T)
    assert np.allclose(uv_np, uv_t.numpy(), atol=1e-3)


def test_resize_and_crop_adjust():
    T = proj.projection_matrix(CALIB)
    xyz = _xyz()
    uv, _ = proj.project_points(xyz, T)
    uv_r, _ = proj.project_points(xyz, proj.adjust_projection_for_resize(T, 0.5, 0.5))
    assert np.allclose(uv_r, uv * 0.5, atol=1e-3)
    uv_c, _ = proj.project_points(xyz, proj.adjust_projection_for_crop(T, 10, 20))
    assert np.allclose(uv_c, uv - np.array([10, 20]), atol=1e-2)


def test_sample_feature_map_shape():
    T = proj.projection_matrix(CALIB)
    uv, _ = proj.project_points(torch.from_numpy(_xyz()), T)
    fmap = torch.randn(1, D, 120, 200)
    assert proj.sample_feature_map(fmap, uv, 4).shape == (N, D)


def test_point_fusion_and_pf_loss():
    T = proj.projection_matrix(CALIB)
    fusion = PointImageFusion(D, img_stride=4).train()
    pts = torch.randn(N, D)
    fmap = torch.randn(1, D, 120, 200)
    fused, pf = fusion(pts, torch.from_numpy(_xyz()), fmap, T, (1028, 1973))
    assert fused.shape == (N, D) and pf.item() >= 0
    fused0, pf0 = fusion(pts, torch.from_numpy(_xyz()), None, None, None)
    assert fused0.shape == (N, D) and pf0.item() == 0


def test_encoders():
    pe = PositionalEncoder(EasyDict(MAX_FREQ=10000, FEAT_SIZE=D, DIMENSIONALITY=3, BASE=2))
    de = DepthEncoder(EasyDict(MAX_FREQ=10000, FEAT_SIZE=D, BASE=2))
    coors = torch.from_numpy(_xyz()).unsqueeze(0)
    assert pe(coors).shape == (1, N, D)
    assert de(coors).shape == (1, N, D)


def test_image_backbone_strides():
    ib = ResNetFPN(out_dim=D, pretrained=False, freeze_bn=True).eval()
    with torch.no_grad():
        f = ib(torch.randn(1, 3, 480, 800))
    assert f["s4"].shape[2:] == (120, 200)
    assert f["s8"].shape[2:] == (60, 100)


def test_tam_and_memory_bank():
    tam = TrackletAssociationModule(D)
    t = 5
    s = tam(torch.randn(t, D), torch.randn(t, 3), torch.randn(t, D), torch.randn(t, 3),
            torch.randint(1, 5, (t,)), torch.rand(t))
    assert s.shape == (t,)
    assert sincos_expand(torch.randn(3, 2), 64).shape == (3, 128)
    bank = TrackMemoryBank(t_hist=4)
    ids0 = bank.associate(tam, torch.randn(3, D), torch.randn(3, 3), torch.zeros(3, 0), 0)
    assert len(ids0) == 3 and len(set(ids0)) == 3
    ids1 = bank.associate(tam, torch.randn(2, D), torch.randn(2, 3),
                          torch.rand(2, len(bank.tracks)), 1)
    assert len(ids1) == 2


def test_decoder_image_cross_attention():
    from mask_4d.models.decoder import MaskedTransformerDecoder

    cfg = EasyDict(
        NHEADS=8, DIM_FEEDFORWARD=512, FEATURE_LEVELS=3, DEC_BLOCKS=2,
        NUM_QUERIES=10, HIDDEN_DIM=D, IMAGE_CROSS_ATTN=True, DEPTH_PE=True,
        POS_ENC=EasyDict(MAX_FREQ=10000, FEAT_SIZE=D, DIMENSIONALITY=3, BASE=2),
    )
    dec = MaskedTransformerDecoder(
        cfg, EasyDict(CHANNELS=[32, 64, 128, 256, 256]), EasyDict(NUM_CLASSES=12)
    ).eval()
    n, q = 400, 10
    feats = torch.randn(n, 32)
    coors = torch.randn(1, n, 3)
    mock = types.SimpleNamespace(
        query=torch.randn(q, D), query_pe=torch.randn(q, D),
        center=torch.zeros(q, 3), size_xy=torch.ones(q, 2), angle=torch.zeros(q),
    )
    with torch.no_grad():
        out = dec(feats, coors, mock, torch.randn(1, n, D),
                  torch.rand(1, n) > 0.5, torch.randn(n, D))
    assert out["pred_logits"].shape == (1, q, 13)
    assert out["pred_masks"].shape == (1, n, q)
    with torch.no_grad():
        out2 = dec(feats, coors, mock, None, None, None)
    assert out2["pred_masks"].shape == (1, n, q)
