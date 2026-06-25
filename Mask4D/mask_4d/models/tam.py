"""Tracklet Association Module (4D-Former Sec. 3.3).

A learned MLP that predicts an association score for a pair of tracklets,
reasoning over appearance (the tracklet queries) and spatial cues (mask
centroid, frame gap, mask IoU). At inference a memory bank of recent tracks is
maintained for `t_hist` frames and current-clip tracklets are matched against
it (or start new tracks).

Input feature per pair (concatenated along the feature dim):
  - centroid coords of both tracklets, each sin/cos-expanded to 64-D
  - both tracklet queries (D each)
  - frame gap, sin/cos-expanded to 64-D
  - mask IoU (scalar; 0 for pairs with no overlapping frames)
"""

import math

import numpy as np
import torch
import torch.nn as nn

from mask_4d.models.blocks import MLP


def sincos_expand(values, dim=64, max_freq=10000.0):
    """Expand a (...,K) tensor to (...,K*dim) via sin/cos at log frequencies."""
    half = dim // 2
    device = values.device
    freqs = torch.logspace(
        0.0, math.log10(max_freq / 2), half, device=device, dtype=values.dtype
    )
    x = values.unsqueeze(-1) * freqs * math.pi  # (...,K,half)
    enc = torch.cat([x.sin(), x.cos()], dim=-1)  # (...,K,dim)
    return enc.flatten(-2)


class TrackletAssociationModule(nn.Module):
    def __init__(self, hidden_dim, pos_dim=64, mlp_hidden=256):
        super().__init__()
        self.pos_dim = pos_dim
        # centroids: 3 coords * pos_dim * 2 tracklets
        # queries:   hidden_dim * 2
        # frame gap: pos_dim
        # iou:       1
        in_dim = (3 * pos_dim) * 2 + hidden_dim * 2 + pos_dim + 1
        self.mlp = MLP(in_dim, mlp_hidden, 1, 4)  # 4 fully connected layers

    def pair_features(self, q_a, c_a, q_b, c_b, frame_gap, iou):
        """Build the per-pair input feature vector.

        q_a, q_b:   (P, D) tracklet queries
        c_a, c_b:   (P, 3) mask centroids
        frame_gap:  (P,) integer frame gaps
        iou:        (P,) mask IoU (0 where no overlap)
        """
        ca = sincos_expand(c_a, self.pos_dim)  # (P, 3*pos_dim)
        cb = sincos_expand(c_b, self.pos_dim)
        fg = sincos_expand(frame_gap.unsqueeze(-1).float(), self.pos_dim)  # (P,pos_dim)
        feat = torch.cat([ca, cb, q_a, q_b, fg, iou.unsqueeze(-1)], dim=-1)
        return feat

    def forward(self, q_a, c_a, q_b, c_b, frame_gap, iou):
        feat = self.pair_features(q_a, c_a, q_b, c_b, frame_gap, iou)
        return self.mlp(feat).squeeze(-1)  # (P,) association logits


class TrackMemoryBank:
    """Inference-time memory of recent tracks for the TAM.

    Each track stores its most recent (query, centroid, frame_idx, track_id).
    Tracks older than `t_hist` frames are evicted.
    """

    def __init__(self, t_hist=4):
        self.t_hist = t_hist
        self.tracks = []  # list of dicts: query, center, frame, id
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def evict(self, current_frame):
        self.tracks = [
            t for t in self.tracks if current_frame - t["frame"] <= self.t_hist
        ]

    def new_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    @torch.no_grad()
    def associate(self, tam, cur_queries, cur_centers, cur_ious_to_prev, frame_idx):
        """Associate current-clip tracklets to memory-bank tracks.

        Args:
            tam: a TrackletAssociationModule.
            cur_queries: (T, D) queries of current tracklets.
            cur_centers: (T, 3) centroids of current tracklets.
            cur_ious_to_prev: (T, M) mask IoU between each current tracklet and
                each memory track over overlapping frames (0 if none). M = len(tracks).
            frame_idx: int current frame index.

        Returns:
            assigned_ids: list[int] length T, track id for each current tracklet.
        """
        self.evict(frame_idx)
        T = cur_queries.shape[0]
        assigned = [None] * T

        if len(self.tracks) and T:
            mem_q = torch.stack([t["query"] for t in self.tracks])  # (M, D)
            mem_c = torch.stack([t["center"] for t in self.tracks])  # (M, 3)
            mem_f = torch.tensor(
                [t["frame"] for t in self.tracks], device=cur_queries.device
            )
            M = len(self.tracks)
            # Build all T*M pairs.
            qa = cur_queries.repeat_interleave(M, 0)
            ca = cur_centers.repeat_interleave(M, 0)
            qb = mem_q.repeat(T, 1)
            cb = mem_c.repeat(T, 1)
            gap = (frame_idx - mem_f).repeat(T)
            iou = cur_ious_to_prev.reshape(-1)
            scores = tam(qa, ca, qb, cb, gap, iou).view(T, M)

            # Greedy matching by descending score.
            order = torch.argsort(scores.reshape(-1), descending=True)
            used_mem, used_cur = set(), set()
            for flat in order.tolist():
                if scores.reshape(-1)[flat] <= 0:  # logit <= 0 -> not associated
                    break
                ti, mi = divmod(flat, M)
                if ti in used_cur or mi in used_mem:
                    continue
                assigned[ti] = self.tracks[mi]["id"]
                used_cur.add(ti)
                used_mem.add(mi)

        # Unmatched current tracklets start new tracks.
        for ti in range(T):
            if assigned[ti] is None:
                assigned[ti] = self.new_id()

        # Update / insert memory entries.
        id_to_track = {t["id"]: t for t in self.tracks}
        for ti in range(T):
            tid = assigned[ti]
            entry = {
                "query": cur_queries[ti].detach(),
                "center": cur_centers[ti].detach(),
                "frame": frame_idx,
                "id": tid,
            }
            if tid in id_to_track:
                id_to_track[tid].update(entry)
            else:
                self.tracks.append(entry)
        return assigned
