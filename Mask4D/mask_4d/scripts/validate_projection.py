"""Validate LiDAR->image projection on real data BEFORE training.

This uses the exact same projection path the multimodal model uses
(mask_4d.utils.projection), so a correct overlay here means fusion will sample
the right image pixels. Run this first on a few sequences; if points don't land
on the right objects, the calibration / convention is wrong and fusion will
hurt rather than help.

Usage:
    python -m mask_4d.scripts.validate_projection \
        --sequence_dir data/kitti/sequences/00 --max 5
"""

import os
from os.path import join

import click
import cv2
import numpy as np

from mask_4d.utils import projection as proj


@click.command()
@click.option("--sequence_dir", required=True, help="contains velodyne/, image_2/, calib.txt")
@click.option("--out", default=None, help="output dir for overlays")
@click.option("--max", "max_files", type=int, default=5)
def main(sequence_dir, out, max_files):
    velo_dir = join(sequence_dir, "velodyne")
    img_dir = join(sequence_dir, "image_2")
    calib_path = join(sequence_dir, "calib.txt")
    for p in (velo_dir, img_dir, calib_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    out = out or join(sequence_dir, "proj_check")
    os.makedirs(out, exist_ok=True)

    T = proj.projection_matrix(proj.parse_calibration(calib_path))
    print("Projection matrix T =\n", T)

    bins = sorted(f for f in os.listdir(velo_dir) if f.endswith(".bin"))[:max_files]
    for f in bins:
        stem = os.path.splitext(f)[0]
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = join(img_dir, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            print(f"  {stem}: no image, skip")
            continue

        pts = np.fromfile(join(velo_dir, f), dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        uv, depth = proj.project_points(xyz, T)
        valid = proj.valid_projection_mask(uv, depth, w, h)
        uvv = uv[valid].astype(np.int32)
        dv = depth[valid]

        # color by inverse depth (near=red, far=blue)
        if len(dv):
            dn = np.clip(dv / 40.0, 0, 1)
            order = np.argsort(-dv)
            for k in order:
                c = (int(255 * dn[k]), 0, int(255 * (1 - dn[k])))
                cv2.circle(img, (uvv[k, 0], uvv[k, 1]), 2, c, -1)
        cov = 100.0 * valid.sum() / len(xyz)
        cv2.imwrite(join(out, stem + ".png"), img)
        print(f"  {stem}: {valid.sum():,}/{len(xyz):,} pts ({cov:.1f}%) -> {out}")

    print("Done. Inspect the overlays: points should align with scene structure.")


if __name__ == "__main__":
    main()
