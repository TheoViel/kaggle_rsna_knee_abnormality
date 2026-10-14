"""Multitask 3D model: ROI segmentation + the 8 ROI findings, from one joint crop.

Backbone: the ROI U-Net (`roi.train.build_model`, residual encoder, 6 stages),
initialised from the round-3 ROI checkpoint, which already segments these 8 ROI
boxes on whole volumes (the segmentation pretraining that mattered most in the
1st place RSNA aneurysm solution). Only the first `n_dec` decoder stages run:
the full-resolution stages are expensive and add nothing to classification
(also as in that solution).

Heads:
  - segmentation: per decoder stage, its 1x1 conv gives 8 box logits (one
    sigmoid channel per ROI: boxes overlap), supervised against the boxes drawn
    at that resolution (deep supervision);
  - classification, `head`:
      masked  per ROI, the decoder features of each stage averaged under that
              ROI's own predicted mask (sigmoid of the stage's seg logits:
              "region-masked pooling"), concatenated over stages; a global token
              (bottleneck average); a small transformer over the 8 ROI tokens +
              global token, so findings see each other (medial vs lateral); one
              logit per ROI token;
      xattn   cross-attention pooling of the bottleneck, `queries` learned
              queries per finding (2nd place), one logit per finding;
  - anatomy (aux, `n_anat`): per decoder stage, a 1x1 conv to the 9 nnU-Net knee
    structures (bones, cartilage, menisci; sigmoid channels, partial labels);
  - meta (aux): fluid-sensitive and fat-suppressed flags from the global token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from roi.extract import ROI_CLASSES

UNET_CKPT = "../logs/roi_run/2026-10-09/1/ckpt_final.pt"
PLANES = ["Sagittal", "Coronal", "Axial"]


def box_masks(boxes, shape):
    """boxes (B, K, 6) fractions (z0, z1, y0, y1, x0, x1) -> (B, K, D, H, W) float masks of the
    voxels whose centre is inside; at least the voxel holding the box centre on each axis."""
    out = None
    for a, n in enumerate(shape):
        c = (torch.arange(n, device=boxes.device, dtype=torch.float32) + 0.5) / n          # voxel centres
        lo, hi = boxes[..., 2 * a, None], boxes[..., 2 * a + 1, None]
        m = (c >= lo) & (c < hi)
        mid = ((lo + hi) / 2 * n).long().clamp(0, n - 1)
        m = m | (torch.arange(n, device=boxes.device) == mid)
        m = m.float()
        view = [m.shape[0], m.shape[1], 1, 1, 1]
        view[2 + a] = n
        m = m.view(view)
        out = m if out is None else out * m
    return out


class CrossAttnPool(nn.Module):
    def __init__(self, dim, n_out, queries=4, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_out * queries, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.n_out, self.queries = n_out, queries
        self.fc = nn.Parameter(torch.zeros(n_out, queries * dim))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.xavier_uniform_(self.fc)

    def forward(self, x):                                            # x B C d h w
        t = x.flatten(2).transpose(1, 2)
        o, _ = self.attn(self.q[None].expand(len(t), -1, -1), t, t)
        o = self.norm(o).reshape(len(t), self.n_out, -1)             # B K (Q C)
        return (o * self.fc).sum(-1) + self.bias


class MT3D(nn.Module):
    def __init__(self, n_dec=3, head="masked", dim=256, tf_layers=2, dropout=0.2, queries=4, pretrained=True,
                 n_cls=len(ROI_CLASSES), detach_masks=False, n_anat=0):
        super().__init__()
        from roi.train import build_model as build_unet

        unet = build_unet(num_classes=len(ROI_CLASSES))
        if pretrained:
            unet.load_state_dict(torch.load(UNET_CKPT, map_location="cpu"))
        self.encoder = unet.encoder
        dec = unet.decoder
        self.n_dec = n_dec
        self.transpconvs = dec.transpconvs[:n_dec]
        self.stages = dec.stages[:n_dec]
        self.seg_layers = dec.seg_layers[:n_dec]
        # Second segmentation head (anatomy, `mt3d.anatomy`): one 1x1 conv per decoder stage.
        self.anat_layers = nn.ModuleList(
            [nn.Conv3d(self.encoder.output_channels[-(s + 2)], n_anat, 1) for s in range(n_dec)]) if n_anat else None
        ch = self.encoder.output_channels                            # 32 64 128 256 320 320
        dec_ch = [ch[-(s + 2)] for s in range(n_dec)]               # 320 256 128 ...
        bott = ch[-1]
        self.head, self.n_cls, self.detach_masks = head, n_cls, detach_masks
        self.plane_embed = nn.Embedding(len(PLANES), dim)
        self.glob = nn.Sequential(nn.Linear(2 * bott, dim), nn.GELU(), nn.LayerNorm(dim))
        if head == "masked":
            self.tok = nn.Sequential(nn.Linear(sum(dec_ch), dim), nn.GELU(), nn.LayerNorm(dim))
            self.roi_embed = nn.Parameter(torch.zeros(n_cls, dim))
            nn.init.normal_(self.roi_embed, std=0.02)
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 8, 2 * dim, dropout=0.1, batch_first=True,
                                                                       norm_first=True), tf_layers) if tf_layers else None
            self.drop = nn.Dropout(dropout)
            self.cls_w = nn.Parameter(torch.zeros(n_cls, dim))
            self.cls_b = nn.Parameter(torch.zeros(n_cls))
            nn.init.xavier_uniform_(self.cls_w)
        else:
            self.xattn = CrossAttnPool(bott, n_cls, queries)
        self.meta = nn.Linear(dim, 2)

    def forward(self, x, plane, boxes=None):
        """x (B, 1, D, S, S) in [0, 1] -> dict(cls (B, K), seg [per stage (B, 8, d, h, w)], meta (B, 2)).
        `boxes` is unused (pooling under given boxes: `mt3d.hybrid`)."""
        skips = self.encoder(x)
        bott = skips[-1]
        g = self.glob(torch.cat([bott.mean((2, 3, 4)), bott.amax((2, 3, 4))], 1)) + self.plane_embed(plane)
        lres, segs, pooled, anat = bott, [], [], []
        for s in range(self.n_dec):
            h = self.transpconvs[s](lres)
            h = self.stages[s](torch.cat((h, skips[-(s + 2)]), 1))
            seg = self.seg_layers[s](h)
            segs.append(seg)
            if self.anat_layers is not None:
                anat.append(self.anat_layers[s](h))
            if self.head == "masked":
                w = torch.sigmoid(seg.float())
                if self.detach_masks:
                    w = w.detach()
                # (B, K, C): features averaged under each ROI's mask.
                num = torch.einsum("bkn,bcn->bkc", w.flatten(2), h.float().flatten(2))
                pooled.append(num / (w.flatten(2).sum(-1, keepdim=True) + 1e-3))
            lres = h
        out = {"seg": segs, "anat": anat, "meta": self.meta(g)}
        if self.head == "masked":
            t = self.tok(torch.cat(pooled, -1)) + self.roi_embed[None] + self.plane_embed(plane)[:, None]
            t = torch.cat([g[:, None], t], 1)
            if self.tf is not None:
                t = self.tf(t)
            out["cls"] = (self.drop(t[:, 1:]) * self.cls_w).sum(-1) + self.cls_b
        else:
            out["cls"] = self.xattn(bott)
        return out
