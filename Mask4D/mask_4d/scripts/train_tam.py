"""Stage-2 training of the Tracklet Association Module (4D-Former Sec. 3.4).

The stage-1 network (backbone + multimodal decoder) is frozen; only the TAM is
optimized. For each training clip we extract per-frame tracklets (their queries
and mask centroids), form all cross-frame tracklet pairs, and supervise the
TAM's association score with a binary cross-entropy loss (positive pair =
same ground-truth track id).

Usage:
    python -m mask_4d.scripts.train_tam --w experiments/<stage1>.ckpt \
        --epochs 2 --lr 1e-4
"""

import os
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from mask_4d.datasets.kitti_dataset import SemanticDatasetModule
from mask_4d.models.mask_model import Mask4D


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--w", type=str, required=True, help="stage-1 checkpoint")
@click.option("--epochs", type=int, default=2)
@click.option("--lr", type=float, default=1e-4)
@click.option("--out", type=str, default="experiments/tam.ckpt")
def main(w, epochs, lr, out):
    cfg = edict(
        {
            **yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))),
            **yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))),
            **yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))),
        }
    )
    assert cfg.get("TAM", {}).get("ENABLED", False), "TAM disabled in config"

    data = SemanticDatasetModule(cfg)
    data.setup()
    loader = data.train_dataloader()

    model = Mask4D(cfg).cuda()
    ckpt = torch.load(w, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Freeze everything except the TAM.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.tam.parameters():
        p.requires_grad_(True)
    model.eval()
    model.tam.train()

    opt = torch.optim.AdamW(model.tam.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        for x in loader:
            tracklets = model.extract_clip_tracklets(x)
            loss = model.tam_loss_from_tracklets(tracklets)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} tam_bce {loss.item():.4f}")

    # Persist: merge updated TAM weights back into the full checkpoint.
    ckpt["state_dict"] = model.state_dict()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved TAM-trained checkpoint to {out}")


if __name__ == "__main__":
    main()
