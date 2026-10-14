"""Hybrid multitask model: a pretrained 2D backbone per frame, a 3D decoder over the stacked maps.

Same task and heads as `mt3d.model.MT3D` (ROI box segmentation, optional anatomy
segmentation, ROI-masked pooling + transformer over the ROI tokens, meta), but
the 3D U-Net encoder, which was pretrained on box locations only, is replaced by
an ImageNet timm backbone (ConvNeXt-tiny by default), the encoder that works best
for the 2.5D crop models:

  - `frames` positions taken every D / frames slices of the (D, S, S) crop (random
    offset in training), each with its two neighbouring slices (`nb` apart) as RGB;
  - the backbone's feature maps at `levels` (timm out_indices; 1, 2, 3 = strides 8,
    16, 32), stacked along the frames into (C, frames, h, w) volumes;
  - a 3D FPN: 1x1x1 lateral convs to `dec` channels (+ a learned frame-position
    embedding with `zpos`), top-down upsampling, one residual 3x3x3 conv block per
    level, so slices exchange information;
  - per level, 1x1x1 conv heads to the 8 ROI box logits (and the 9 anatomy logits),
    and ROI-masked average pooling of the level's features under its own
    predicted box mask (`pool="pred"`) or under the ROI U-Net's boxes, which
    come with the crop at inference too (`pool="box"`).

Tokens for the transformer: a global one, one per ROI finding (pooled under its ROI),
one per extra finding without an ROI (`n_cls` > 8: effusion, synovitis, ... pooled
under the union of the ROI masks), and with `anat_tokens` one per anatomy structure
(pooled under its predicted mask; context only, no logit).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from mt3d.model import PLANES
from roi.extract import ROI_CLASSES


def block3d(c):
    return nn.Sequential(nn.Conv3d(c, c, 3, padding=1, bias=False), nn.GroupNorm(16, c), nn.GELU(),
                         nn.Conv3d(c, c, 3, padding=1, bias=False), nn.GroupNorm(16, c))


def masked_mean(w, h):
    """w (B, K, n...) weights, h (B, C, n...) features -> (B, K, C), in float32: autocast would run the
    einsum in float16, whose sums over thousands of voxels can overflow (inference on T4s)."""
    with torch.autocast("cuda", enabled=False):
        w = w.float().flatten(2)
        return torch.einsum("bkn,bcn->bkc", w, h.float().flatten(2)) / (w.sum(-1, keepdim=True) + 1e-3)


class OrthoEncoder(nn.Module):
    """OrthoFoundation (DINOv3 ViT-L/16, `mt3d.ortho.load_ortho`) as a features_only-style encoder: the
    patch tokens of blocks `levels` as (B, 1024, 14, 14) maps at 224 px, with the attributes Hybrid reads
    (feature_info.channels() / .reduction(), pretrained_cfg). `freeze`: blocks below it are frozen (and the
    patch embedding); `ckpt`: gradient checkpointing of the trainable blocks."""

    def __init__(self, levels=(11, 17, 23), freeze=0, ckpt=False, pretrained=True):
        super().__init__()
        from types import SimpleNamespace

        import timm

        from mt3d.ortho import load_ortho
        self.model = load_ortho() if pretrained else timm.create_model(
            "vit_large_patch16_dinov3", pretrained=False, num_classes=0, img_size=224)
        self.levels = list(levels)
        n = len(self.levels)
        self.feature_info = SimpleNamespace(channels=lambda: [1024] * n, reduction=lambda: [16] * n)
        self.pretrained_cfg = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
        if freeze:
            for p_ in self.model.patch_embed.parameters():
                p_.requires_grad = False
            for name in ("cls_token", "reg_token"):
                if getattr(self.model, name, None) is not None:
                    getattr(self.model, name).requires_grad = False
            for b in self.model.blocks[:freeze]:
                for p_ in b.parameters():
                    p_.requires_grad = False
        if ckpt:
            self.model.set_grad_checkpointing(True)

    def forward(self, x):
        return self.model.forward_intermediates(x, indices=self.levels, norm=True, output_fmt="NCHW", intermediates_only=True)


class Hybrid(nn.Module):
    def __init__(self, backbone="convnext_tiny.fb_in22k", frames=16, dec=192, dim=256, tf_layers=2, dropout=0.2,
                 drop_path=0.1, n_cls=len(ROI_CLASSES), n_anat=0, pretrained=True, pool="pred",
                 levels=(1, 2, 3), anat_tokens=False, zpos=False, nb=1, zoom=(), zoom_size=128, zoom_frames=8,
                 zoom_margin=0.25, aux_roi=(), rep_dim=0, freeze=0, ckpt=False, ortho="", extra_head="union",
                 seq_embed=False):
        super().__init__()
        if backbone == "ortho":
            self.encoder = OrthoEncoder(levels, freeze=freeze, ckpt=ckpt, pretrained=pretrained)
        else:
            kw = {"img_size": 224} if "vit" in backbone else {}
            self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                             out_indices=tuple(levels), drop_path_rate=drop_path, **kw)
        ch = self.encoder.feature_info.channels()
        # Plain ViTs (e.g. DINOv2) give every level at one stride: their maps are resampled to strides
        # 8, 16, 32 (ViTDet-style simple pyramid), so the decoder and the pooling are the CNNs'.
        self.pyramid = len(set(self.encoder.feature_info.reduction())) == 1
        cfg = self.encoder.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"])[None, :, None, None])
        self.register_buffer("std", torch.tensor(cfg["std"])[None, :, None, None])
        n_roi = len(ROI_CLASSES)
        self.frames, self.n_cls, self.n_roi, self.pool, self.nb = frames, n_cls, n_roi, pool, nb
        self.anat_tokens = anat_tokens and n_anat > 0
        self.n_anat = n_anat
        self.lateral = nn.ModuleList([nn.Conv3d(c, dec, 1) for c in ch])
        self.zpos = nn.Parameter(torch.zeros(len(ch), dec, frames)) if zpos else None
        self.blocks = nn.ModuleList([block3d(dec) for _ in ch])
        self.seg_layers = nn.ModuleList([nn.Conv3d(dec, n_roi, 1) for _ in ch])
        self.anat_layers = nn.ModuleList([nn.Conv3d(dec, n_anat, 1) for _ in ch]) if n_anat else None
        # seq_embed: the "plane" index also encodes the sequence, plane + 3 x (2 x fluid-sensitive + fat-suppressed)
        # (mt3d.train.plane_index), so the model is told what bright marrow means on this series.
        self.plane_embed = nn.Embedding(len(PLANES) * (4 if seq_embed else 1), dim)
        self.glob = nn.Sequential(nn.Linear(2 * ch[-1], dim), nn.GELU(), nn.LayerNorm(dim))
        self.tok = nn.Sequential(nn.Linear(len(ch) * dec, dim), nn.GELU(), nn.LayerNorm(dim))
        # Findings without an ROI (n_cls > 8): extra_head "union" gives each a token pooled under the union of
        # the ROI masks; "attn" gives each a learned query attending over the 8 ROI tokens + the global token.
        self.extra_head = extra_head if n_cls > n_roi else "union"
        n_tok = (n_cls if self.extra_head == "union" else n_roi) + (n_anat if self.anat_tokens else 0)
        if self.extra_head == "attn":
            self.x_query = nn.Parameter(torch.randn(n_cls - n_roi, dim) * dim ** -0.5)
            self.x_key = nn.Linear(dim, dim)
        self.roi_embed = nn.Parameter(torch.randn(n_tok, dim) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 8, 2 * dim, dropout=0.1, batch_first=True,
                                                                   norm_first=True), tf_layers) if tf_layers else None
        self.drop = nn.Dropout(dropout)
        self.cls_w = nn.Parameter(torch.zeros(n_cls, dim))
        self.cls_b = nn.Parameter(torch.zeros(n_cls))
        nn.init.xavier_uniform_(self.cls_w)
        self.meta = nn.Linear(dim, 2)
        # Auxiliary targets tied to an ROI (MOAKS gradings for OAI pretraining): one logit each, from that
        # ROI's token after the transformer.
        self.aux_roi = list(aux_roi)
        if self.aux_roi:
            self.aux_w = nn.Parameter(torch.zeros(len(self.aux_roi), dim))
            self.aux_b = nn.Parameter(torch.zeros(len(self.aux_roi)))
            nn.init.xavier_uniform_(self.aux_w)
        # OrthoFoundation frozen features (`mt3d.ortho`, joint crops: per frame CLS, mean, 8 ROI-pooled
        # tokens), `ortho`: "fuse" adds them (frame-averaged, projected) to the global / ROI tokens (input);
        # "dist" predicts them from the tokens before the transformer (auxiliary target, distillation).
        self.ortho = ortho
        if ortho == "fuse":
            self.o_roi = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, dim))
            self.o_glob = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, dim))
        elif ortho == "dist":
            self.d_roi = nn.Linear(dim, 1024)
            self.d_glob = nn.Linear(dim, 1024)
        # Report head (`PairHybrid` only): the study token -> the report's text embedding (`mt3d.reports`).
        self.rep_head = nn.Linear(dim, rep_dim) if rep_dim else None
        # Zoom branch: per ROI in `zoom`, its own 2.5D crop through the same encoder (see `zoom_tokens`).
        self.zoom, self.zoom_size, self.zoom_frames, self.zoom_margin = list(zoom), zoom_size, zoom_frames, zoom_margin
        if self.zoom:
            self.zoom_attn = nn.Sequential(nn.Linear(ch[-1], 128), nn.Tanh(), nn.Linear(128, 1))
            self.zoom_proj = nn.Sequential(nn.Linear(ch[-1], dim), nn.GELU(), nn.LayerNorm(dim))
            self.zoom_gate = nn.Parameter(torch.zeros(len(self.zoom), dim))

    def zoom_tokens(self, x, boxes):
        """Per zoomed ROI, a 2.5D crop like the crop models': a square window around the ROI's box
        (side = its larger side x (1 + 2 x margin)), upsampled to zoom_size, `zoom_frames` adjacent
        slices centred on the box (neighbours as RGB), through the shared encoder; attention over the
        frames -> (B, n_zoom, dim). The boxes are the (augmented) U-Net box masks, so the window
        follows the image's augmentation."""
        b, _, d, s, _ = x.shape
        k, f, z = len(self.zoom), self.zoom_frames, self.zoom_size
        m = boxes[:, self.zoom] > 0.5                                            # B k D S S
        ext = []
        for ax, n in ((2, d), (3, s), (4, s)):
            other = tuple(a for a in (2, 3, 4) if a != ax)
            on = m.any(other[1]).any(other[0])                                   # B k n
            ar = torch.arange(n, device=x.device, dtype=torch.float32)
            big = torch.where(on, ar, torch.full_like(ar, float(n)))
            small = torch.where(on, ar, torch.full_like(ar, -1.0))
            lo, hi = big.amin(-1), small.amax(-1) + 1
            empty = ~on.any(-1)
            ext.append((torch.where(empty, torch.zeros_like(lo), lo), torch.where(empty, torch.full_like(hi, n), hi)))
        (z0, z1), (y0, y1), (x0, x1) = ext
        cz = ((z0 + z1) / 2).floor()
        idx = (cz[..., None] - f // 2 + torch.arange(f, device=x.device)).clamp(0, d - 1).long()      # B k f
        trip = torch.stack([(idx - self.nb).clamp(0, d - 1), idx, (idx + self.nb).clamp(0, d - 1)], -1)   # B k f 3
        bi = torch.arange(b, device=x.device)[:, None, None, None]
        frames = x[:, 0][bi, trip].reshape(b * k * f, 3, s, s)
        side = torch.maximum(y1 - y0, x1 - x0) * (1 + 2 * self.zoom_margin)
        side = side.clamp(min=0.15 * s) / s                                       # fraction of the crop
        cy, cx = (y0 + y1) / s - 1, (x0 + x1) / s - 1                             # centre in [-1, 1]
        theta = torch.zeros(b, k, 2, 3, device=x.device)
        theta[..., 0, 0] = theta[..., 1, 1] = side
        theta[..., 0, 2], theta[..., 1, 2] = cx, cy
        theta = theta[:, :, None].expand(b, k, f, 2, 3).reshape(-1, 2, 3)
        grid = F.affine_grid(theta, (b * k * f, 3, z, z), align_corners=False)
        crops = F.grid_sample(frames, grid.to(frames.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
        v = self.encoder((crops - self.mean) / self.std)[-1].float().mean((2, 3)).reshape(b, k, f, -1)
        w = self.zoom_attn(v).softmax(2)
        return self.zoom_proj((w * v).sum(2))

    def forward(self, x, plane, boxes=None, offset=None, tokens_only=False, ortho=None):
        """x (B, 1, D, S, S) in [0, 1] -> dict(cls (B, n_cls), seg / anat [per level (B, k, frames, h, w)], meta).

        `boxes` (B, 8, D, S, S): the ROI U-Net's boxes, drawn (and augmented) like the image; with
        pool="box" the features are pooled under them instead of under the predicted masks.
        `offset`: the first frame's slice (default: random in training, centred in eval)."""
        b, _, d, s, _ = x.shape
        step = max(d // self.frames, 1)
        off = offset if offset is not None else int(torch.randint(step, ())) if self.training else step // 2
        idx = (torch.arange(self.frames, device=x.device) * step + off).clamp(max=d - 1)
        trip = torch.stack([(idx - self.nb).clamp(0, d - 1), idx, (idx + self.nb).clamp(0, d - 1)], 1)   # F 3
        frames = x[:, 0][:, trip].reshape(b * self.frames, 3, s, s)
        maps = self.encoder((frames - self.mean) / self.std)
        if self.pyramid:
            maps = [F.interpolate(m, size=(s // r, s // r), mode="bilinear", align_corners=False)
                    for m, r in zip(maps, (8, 16, 32)[-len(maps):])]
        vols = [m.reshape(b, self.frames, *m.shape[1:]).transpose(1, 2) for m in maps]          # B C F h w
        top = vols[-1]
        g = self.glob(torch.cat([top.float().mean((2, 3, 4)), top.float().amax((2, 3, 4))], 1)) + self.plane_embed(plane)
        # Top-down 3D FPN, coarsest level first.
        feats, prev = [None] * len(vols), None
        for i in reversed(range(len(vols))):
            h = self.lateral[i](vols[i])
            if self.zpos is not None:
                h = h + self.zpos[i][None, :, :, None, None].to(h.dtype)
            if prev is not None:
                h = h + F.interpolate(prev, size=h.shape[2:], mode="trilinear", align_corners=False)
            h = F.gelu(h + self.blocks[i](h))
            feats[i] = prev = h
        segs, anat, pooled = [], [], []
        for i in reversed(range(len(feats))):                          # coarse -> fine, as in MT3D
            h = feats[i]
            seg = self.seg_layers[i](h)
            segs.append(seg)
            w = F.adaptive_avg_pool3d(boxes, seg.shape[2:]) if self.pool == "box" else torch.sigmoid(seg.float())
            ws = [w]
            if self.n_cls > self.n_roi and self.extra_head == "union":  # findings without an ROI: the union
                ws.append(w.amax(1, keepdim=True).expand(-1, self.n_cls - self.n_roi, *w.shape[2:]))
            if self.anat_layers is not None:
                a = self.anat_layers[i](h)
                anat.append(a)
                if self.anat_tokens:
                    ws.append(torch.sigmoid(a.float()))
            pooled.append(masked_mean(torch.cat(ws, 1), h))
        pooled = torch.cat(pooled, -1)
        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
            # Masked means under a near-empty mask can exceed the float16 range (inference on T4s):
            # this projection runs in float32 there. bfloat16 / float32 (training, validation) are unchanged.
            with torch.autocast("cuda", enabled=False):
                t = self.tok(pooled.float())
        else:
            t = self.tok(pooled)
        t = t + self.roi_embed[None] + self.plane_embed(plane)[:, None]
        extra = {}
        if self.ortho == "fuse":
            o = ortho.float()                                              # B 16 10 1024
            t = torch.cat([t[:, :8] + self.o_roi(o[:, :, 2:].mean(1)).to(t.dtype), t[:, 8:]], 1)
            g = g + self.o_glob(o[:, :, 0].mean(1))
        elif self.ortho == "dist":
            extra = {"dist_roi": self.d_roi(t[:, :8].float()), "dist_glob": self.d_glob(g.float())}
        if self.zoom:
            # Gated (starts at 0) so the zoomed view is added to its ROI token progressively.
            zt = self.zoom_tokens(x, boxes) * self.zoom_gate[None]
            t = t.index_add(1, torch.tensor(self.zoom, device=t.device), zt.to(t.dtype))
        t = torch.cat([g[:, None], t], 1)
        if tokens_only:                                                # for `PairHybrid`: tokens before the transformer
            return {"tokens": t, "seg": segs, "anat": anat, "meta": self.meta(g)} | extra
        if self.tf is not None:
            t = self.tf(t)
        cls = self.logits(t)
        out = {"cls": cls, "seg": segs, "anat": anat, "meta": self.meta(g)} | extra
        if self.aux_roi:
            out["aux"] = self.aux_logits(t)
        return out

    def logits(self, t):
        """t (..., tokens, dim) after the transformer -> (..., n_cls): ROI findings from their own tokens; the
        findings without an ROI from their token ("union") or by attention over the ROI + global tokens ("attn")."""
        if self.extra_head != "attn":
            return (self.drop(t[..., 1:1 + self.n_cls, :]) * self.cls_w).sum(-1) + self.cls_b
        roi = (self.drop(t[..., 1:1 + self.n_roi, :]) * self.cls_w[:self.n_roi]).sum(-1) + self.cls_b[:self.n_roi]
        # global + 8 ROI tokens (+ the anatomy tokens, pooled under the predicted bone / cartilage / meniscus masks)
        ctx = t[..., :1 + self.n_roi + (self.n_anat if self.anat_tokens else 0), :]
        att = torch.einsum("ed,...kd->...ek", self.x_query.to(ctx.dtype), self.x_key(ctx)) / ctx.shape[-1] ** 0.5
        self.last_attn = att.softmax(-1).detach()                            # (..., n_extra, 9), for inspection
        pooled = torch.einsum("...ek,...kd->...ed", att.softmax(-1), ctx)
        extra = (self.drop(pooled) * self.cls_w[self.n_roi:]).sum(-1) + self.cls_b[self.n_roi:]
        return torch.cat([roi, extra], -1)

    def aux_logits(self, t):
        """t (..., tokens, dim) after the transformer -> (..., n_aux): each target from its ROI's token."""
        i = torch.tensor(self.aux_roi, device=t.device) + 1
        return (self.drop(t[..., i, :]) * self.aux_w).sum(-1) + self.aux_b


class PairHybrid(nn.Module):
    """Cross-view study model: a sagittal and a coronal series of one study through the same Hybrid,
    their tokens (global + ROI + ...) through one transformer, so each ROI token can attend to the
    other view's; one logit per finding and view slot, averaged over the valid slots (a study with
    one plane only fills the second slot with a copy, masked out)."""

    def __init__(self, base, slots=2):
        super().__init__()
        self.base, self.slots = base, slots
        dim = base.cls_w.shape[1]
        # Pairs: one learned embedding per slot (sagittal, coronal). Sets: none, the slots are unordered
        # (each token already carries its series' plane embedding).
        self.slot_embed = nn.Parameter(torch.zeros(slots, dim)) if slots == 2 else None

    def forward(self, x, plane, boxes, valid, ortho=None):
        """x (B * slots, 1, D, S, S) slot-major per study, valid (B, slots) bool."""
        out = self.base(x, plane, boxes, tokens_only=True, ortho=ortho)
        t = out["tokens"]
        n, k, dim = t.shape
        m = self.slots
        t = t.reshape(n // m, m, k, dim)
        if self.slot_embed is not None:
            t = t + self.slot_embed[None, :, None].to(t.dtype)
        t = t.reshape(n // m, m * k, dim)
        pad = ~valid[:, :, None].expand(-1, -1, k).reshape(n // m, m * k)
        b = self.base
        if b.tf is not None:
            t = b.tf(t, src_key_padding_mask=pad)
        t = t.reshape(n // m, m, k, dim)
        logit = b.logits(t)                                                      # B slots n_cls
        w = valid.float()[..., None]
        out["cls"] = (logit * w).sum(1) / w.sum(1)
        if b.aux_roi:
            out["aux"] = (b.aux_logits(t) * w).sum(1) / w.sum(1)
        if b.rep_head is not None:
            out["rep"] = b.rep_head((t[:, :, 0] * w).sum(1) / w.sum(1))           # study token: mean of global tokens
        return out
