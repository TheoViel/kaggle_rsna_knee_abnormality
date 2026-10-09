"""2.5D attention model: 2D CNN per slice, class attention over xy, then over z,
then over the series of a study.

Shapes, with K classes and D channels:

    frames    (N, 3, H, W)       every frame (3 adjacent slices as channels) of
                                 every series in the batch, flattened, so the
                                 CNN never runs on padding
    fmap      (N, D, h, w)       stride-16 (or 8) map from an FPN on the encoder
    xy attn   (N, K, h, w)       softmax over h*w, one map per class
    tokens    (N, K, D)          class-specific attention-pooled slice features
    -> scattered to (S, Z, K, D) per series, z-padded and masked
    z context                    small transformer over z, per class (shared weights)
    z attn    (S, Z, K)          softmax over z, one distribution per class
    series    (S, K, D)          + embedding of (plane, fluid, fat-sat)
    -> scattered to (B, M, K, D) per study
    series attn (B, M, K)        softmax over the study's series
    logits    (B, K)             one linear head per class on its own feature

The 2D CNN sees one frame at a time (3 adjacent slices): z only enters after pooling, which is what
makes the xy maps per-slice segmentations and the z weights a slice ranking.

Why the z attention starts from the xy logsumexp. A softmax over xy says *where*
in the slice class k is looked for, and throws away *how much* of it there is.
The logsumexp of the same logits keeps that amount, and adding it to the z logits
makes softmax_z(lse) * softmax_xy exactly the joint softmax over (z, y, x). The
learnt z term is on top of that, so `attention_volume` is a proper 3D attention
distribution per class and per series: the localisation output.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNNeck(nn.Module):
    """Top-down FPN over the last encoder levels, output at the finest one.

    With 2 levels the map is at stride 16, with 3 at stride 8: the resolution of
    the attention maps, so of the localisation.
    """

    def __init__(self, in_channels, dim):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in in_channels])
        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )

    def forward(self, feats):
        x = self.lateral[-1](feats[-1])
        for lat, f in zip(self.lateral[-2::-1], feats[-2::-1]):
            x = lat(f) + F.interpolate(x, size=f.shape[-2:], mode="bilinear",
                                       align_corners=False)
        return self.out(x)


class ClassAttentionPool2d(nn.Module):
    """`heads` softmax attention maps per class over the (h, w) grid.

    Head j of class k pools channel group j of the values, as in multi-head
    attention, and the groups are concatenated back to D. With one head, class k
    has a single map and it settles on the most discriminative evidence -- for
    ACL that can be the pivot-shift contusion rather than the ligament. Several
    heads let distinct evidence take distinct maps.
    """

    def __init__(self, dim, num_classes, heads=1):
        super().__init__()
        assert dim % heads == 0
        self.k, self.h = num_classes, heads
        self.attn = nn.Conv2d(dim, num_classes * heads, 1)
        self.value = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """-> tokens (N, K, D), attn (N, K, H, hw), lse (N, K)."""
        n = x.shape[0]
        logits = self.attn(x).flatten(2).float().reshape(n, self.k, self.h, -1)
        attn = logits.softmax(-1)
        value = self.value(x).flatten(2).float().reshape(n, self.h, -1, logits.shape[-1])
        tokens = torch.einsum("nkhp,nhcp->nkhc", attn, value).reshape(n, self.k, -1)
        lse = torch.logsumexp(logits.flatten(2), -1) - math.log(logits[0, 0].numel())
        return tokens, attn, lse


class GatedAttention(nn.Module):
    """Gated attention-MIL scores (Ilse et al. 2018), one logit per token."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)


def _masked_softmax(logits, mask, dim):
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits.softmax(dim).nan_to_num(0.0)


class ClassHead(nn.Module):
    """K independent linear classifiers, class k reading only feature k."""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, dim) * dim ** -0.5)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):                                       # ... K D
        return (x * self.weight).sum(-1) + self.bias


class KneeAttnNet(nn.Module):
    def __init__(
        self,
        backbone="convnext_tiny.fb_in22k",
        pretrained=True,
        num_classes=12,
        dim=256,
        z_layers=1,
        z_heads=4,
        n_meta=3 + 2 + 2,
        dropout=0.1,
        drop_path=0.1,
        attn_stride=16,
        in_chans=3,
        xy_heads=1,
        grad_checkpointing=False,
        verbose=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        kwargs = {"drop_path_rate": drop_path} if drop_path else {}
        n_levels = {16: 2, 8: 3}[int(attn_stride)]
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, in_chans=in_chans,
            out_indices=tuple(range(-n_levels, 0)), **kwargs)
        if grad_checkpointing:
            self.encoder.set_grad_checkpointing(True)
        chans = self.encoder.feature_info.channels()
        if verbose:
            print(f"-> encoder {backbone}  channels {chans}  "
                  f"strides {self.encoder.feature_info.reduction()}")

        self.neck = FPNNeck(chans, dim)
        self.xy_pool = ClassAttentionPool2d(dim, num_classes, heads=xy_heads)

        self.class_embed = nn.Parameter(torch.zeros(num_classes, dim))
        # Position along z as a function of the *relative* position of the frame
        # in its series, not its index: the model then runs on any number of
        # frames, e.g. one per slice when exporting attention.
        self.z_rel = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.token_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            dim, z_heads, dim_feedforward=2 * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.z_ctx = nn.TransformerEncoder(layer, z_layers, enable_nested_tensor=False)
        self.z_attn = GatedAttention(dim, dim // 2)

        self.meta = nn.Linear(n_meta, dim)
        self.series_norm = nn.LayerNorm(dim)
        self.series_attn = GatedAttention(dim, dim // 2)

        self.dropout = nn.Dropout(dropout)
        self.head = ClassHead(dim, num_classes)
        self.series_head = ClassHead(dim, num_classes)

        nn.init.normal_(self.class_embed, std=0.02)

    def encode_slices(self, x, pad_to=32):
        """(N, 1, H, W) -> fmap (N, D, h, w), tokens (N, K, D), xy attn, lse.

        N changes every batch, and cuDNN pays ~0.7 s to set up kernels for each
        new shape -- more than the step itself. Padding N to a multiple of
        `pad_to` keeps the set of shapes small. Everything here is per-slice
        (GroupNorm, not BatchNorm, in the neck), so the zero slices are inert
        and are cut off at the end.
        """
        n = x.shape[0]
        if pad_to and n % pad_to:
            x = torch.cat([x, x.new_zeros(pad_to - n % pad_to, *x.shape[1:])])
        fmap = self.neck(self.encoder(x))
        tokens, xy_attn, lse = self.xy_pool(fmap)
        return fmap[:n], tokens[:n], xy_attn[:n], lse[:n]

    def forward(self, batch, return_attn=False):
        """`batch` is the collated dict from `knee_cls.data.collate_studies`.

        Required keys: images (N,1,H,W) float, slice_series (N,), slice_z (N,),
        slice_zrel (N,), series_study (S,), series_pos (S,), series_meta (S,M),
        n_studies (int), max_z (int), max_series (int).
        """
        images = batch["images"]
        K, D = self.num_classes, self.dim
        S = batch["series_study"].shape[0]
        B = int(batch["n_studies"])
        Z = int(batch["max_z"])
        M = int(batch["max_series"])

        fmap, tokens, xy_attn, lse = self.encode_slices(images)  # N K D, N K hw, N K

        # Slice tokens -> (S, Z, K, D).
        si, zi = batch["slice_series"], batch["slice_z"]
        z_mask = torch.zeros(S, Z, dtype=torch.bool, device=images.device)
        z_mask[si, zi] = True
        pos = self.z_rel(batch["slice_zrel"][:, None].to(tokens.dtype))
        tok = self.token_norm(tokens) + pos[:, None] + self.class_embed
        grid = images.new_zeros(S, Z, K, D, dtype=torch.float32)
        grid[si, zi] = tok.float()
        lse_grid = images.new_zeros(S, Z, K, dtype=torch.float32)
        lse_grid[si, zi] = lse.float()

        # z context, per class: sequences are (series, class) pairs.
        seq = grid.permute(0, 2, 1, 3).reshape(S * K, Z, D)
        pad = ~z_mask[:, None].expand(S, K, Z).reshape(S * K, Z)
        seq = self.z_ctx(seq, src_key_padding_mask=pad)
        seq = seq.reshape(S, K, Z, D).permute(0, 2, 1, 3)       # S Z K D

        z_logits = self.z_attn(seq).float() + lse_grid          # S Z K
        z_attn = _masked_softmax(z_logits, z_mask[..., None], dim=1)
        series = torch.einsum("szk,szkd->skd", z_attn, seq.float())

        series = self.series_norm(series + self.meta(batch["series_meta"])[:, None])
        series_logits = self.series_head(self.dropout(series))  # S K

        # Series -> (B, M, K, D).
        bi, mi = batch["series_study"], batch["series_pos"]
        s_mask = torch.zeros(B, M, dtype=torch.bool, device=images.device)
        s_mask[bi, mi] = True
        sgrid = images.new_zeros(B, M, K, D, dtype=torch.float32)
        sgrid[bi, mi] = series.float()
        s_logits = self.series_attn(sgrid).float()
        s_attn = _masked_softmax(s_logits, s_mask[..., None], dim=1)   # B M K
        study = torch.einsum("bmk,bmkd->bkd", s_attn, sgrid)
        logits = self.head(self.dropout(study))

        out = {"logits": logits, "series_logits": series_logits}
        if return_attn:
            h, w = fmap.shape[-2:]
            out.update({
                "xy_attn": xy_attn.mean(2).reshape(-1, K, h, w),     # N K h w
                "xy_attn_heads": xy_attn.reshape(-1, K, xy_attn.shape[2], h, w),
                "z_attn": z_attn,                                # S Z K
                "z_logits": z_logits,                            # S Z K
                "series_attn": s_attn,                           # B M K
            })
        return out


def attention_volume(xy_attn, z_attn, slice_series, slice_z, series_idx, size=None):
    """3D attention of one series: (K, Z, h, w), summing to 1 over (Z, h, w) per class.

    xy_attn (N, K, h, w) and z_attn (S, Z, K) are from `forward(return_attn=True)`.
    Pass `size=(H, W)` to upsample in-plane to the input resolution.
    """
    sel = slice_series == series_idx
    zs = slice_z[sel]
    xy = xy_attn[sel]                                           # n K h w
    w = z_attn[series_idx][zs]                                  # n K
    vol = (xy * w[..., None, None]).permute(1, 0, 2, 3)          # K n h w
    if size is not None:
        vol = F.interpolate(vol, size=size, mode="bilinear", align_corners=False)
        vol = vol / vol.flatten(1).sum(-1).clamp_min(1e-12)[:, None, None, None]
    return vol
