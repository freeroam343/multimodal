# 4D-Former multimodal replication (on top of Mask4D)

This branch (`4dformer-multimodal`) adapts Mask4D into a faithful re-implementation
of **4D-Former** (Athar et al., CoRL 2023, arXiv:2311.01520) for the
**single forward-facing camera** (SemanticKITTI-style) configuration.

Mask4D already provided the query-based mask transformer decoder, Hungarian
matcher, mask/dice/CE losses with deep supervision, the sequence dataloader, and
the 4D panoptic evaluator. This branch adds the three multimodal pillars that
Mask4D lacked.

## What was added

| 4D-Former component | File | Notes |
|---|---|---|
| LiDAR→image projection | `mask_4d/utils/projection.py` | Matches your `calib.txt` convention `P2 @ R_rect_00 @ Tr_velo_to_cam @ R_axes`; resize/crop-aware intrinsics |
| Image backbone (ResNet-50 + FPN) | `mask_4d/models/image_backbone.py` | Outputs I4 (stride 4) and I8 (stride 8) at hidden dim |
| Point-level fusion + pseudo-fusion L2 (Eq. 1, 4) | `mask_4d/models/fusion.py` | `MLP_fusion`/`MLP_pseudo`; `L_pf` added to total loss |
| Image cross-attention in decoder | `mask_4d/models/decoder.py` | Per-block query→image-feature cross-attention; coverage-masked keys |
| Depth positional encoding (Sec. 3.2) | `mask_4d/models/positional_encoder.py` | `DepthEncoder`, added to Fourier(xyz) encoding |
| Tracklet Association Module (Sec. 3.3) | `mask_4d/models/tam.py` | 4-layer MLP + memory bank for inference |
| Multimodal wiring | `mask_4d/models/mask_model.py` | Image branch, fusion, `pf_loss`, TAM stage-2 helpers |
| Dataset images/calib | `mask_4d/datasets/kitti_dataset.py` | Loads `image_2/<stem>.*`, parses calib, resize(480)/crop(0.7)/color-aug |
| Config | `config/model.yaml`, `config/decoder.yaml` | `IMAGE`, `TAM`, `LOSS.W_PF`, `IMAGE_CROSS_ATTN`, `DEPTH_PE` |
| Stage-2 TAM trainer | `mask_4d/scripts/train_tam.py` | Freezes stage-1, trains TAM with pairwise BCE |
| Projection validation | `mask_4d/scripts/validate_projection.py` | Overlay tool — run FIRST to confirm calibration |
| CPU unit tests | `mask_4d/tests/test_multimodal.py` | 8 tests, all passing |

## Expected data layout (per sequence)
```
sequences/SS/
  velodyne/NNNNNN.bin     # x,y,z,intensity float32
  labels/NNNNNN.label     # sem (low 16b) + instance/track id (high 16b)
  image_2/NNNNNN.png      # forward camera, stem matches the scan
  calib.txt               # P2, R_rect_00, Tr_velo_to_cam
  poses.txt
```
Points without a valid image projection fall back to the pseudo-fusion path and
are masked out of image cross-attention, so partial camera coverage is fine.

## How to run
```bash
# 0. Validate calibration on a few scans (DO THIS FIRST)
python -m mask_4d.scripts.validate_projection --sequence_dir data/kitti/sequences/00 --max 5

# 1. Stage 1 — full multimodal network (image branch trains end-to-end)
python -m mask_4d.scripts.train_model

# 2. Stage 2 — train the TAM with the stage-1 net frozen
python -m mask_4d.scripts.train_tam --w experiments/mask_4d/<stage1>.ckpt

# tests
python -m pytest mask_4d/tests/test_multimodal.py
```

## Deliberate divergences from the paper
These were chosen to respect Mask4D's proven LiDAR stack rather than rewrite the
backbone; each is a localized, documented choice:

1. **LiDAR backbone.** Kept Mask4D's SphereFormer (sparse-conv U-Net + sphere
   transformer) instead of 4D-Former's point-voxel backbone. Point-level fusion
   is therefore applied at the **final** point features Z (projected to hidden
   dim) rather than injected at multiple intermediate backbone stages. This is
   the principal fusion site; multi-stage injection can be added later.
2. **Image cross-attention granularity.** Queries cross-attend to **per-point**
   projected image features (I4+I8 summed) aligned with the LiDAR points, rather
   than to per-stride voxel-projected feature sets. This preserves the
   query→image mechanism while reusing the existing point coordinate set.
3. **Feature dim D = 256** (Mask4D's hidden dim) rather than the paper's 128.
4. **Depth PE is additive** to the Fourier(xyz) encoding (paper concatenates);
   both encode the same information and this keeps decoder dims unchanged.
5. **TAM at inference is not yet wired into the association loop.** The module,
   memory bank, and stage-2 trainer are complete and unit-tested; Mask4D's
   existing query-propagation tracking still drives inference. Swapping in the
   TAM-driven association (via `TrackMemoryBank.associate` in
   `panoptic4d_inference`) is the final integration step — left as a toggle
   because it can't be validated without the dataset + GPU.

## Validation status
- 8/8 CPU unit tests pass (projection round-trip, resize/crop intrinsics,
  feature sampling, fusion + pf loss, encoders, image backbone strides, TAM +
  memory bank, decoder image cross-attention + LiDAR-only fallback).
- The spconv LiDAR backbone and full forward pass require CUDA + the dataset and
  have **not** been run in this environment — first real-data step is the
  projection validation script, then a short stage-1 overfit on one sequence.
