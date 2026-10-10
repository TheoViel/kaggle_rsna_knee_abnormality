"""Box coordinates between a series' original voxel grid and the U-Net volume grid.

Original: slices 0..n_slices-1 (output/series_npy order), rows, cols.
Cache (`prepare_cache.py`): the same slices, or n_cache = 64 of them taken at
round(linspace(0, n - 1, 64)) when there were more; in-plane, the slice padded to
a centred square of side s = max(rows, cols).
Volume (`roi.prepare`): the cache resampled to D slices and S x S pixels with
half-pixel-centred (align_corners=False) interpolation.

Boxes are inclusive voxel index ranges; conversions go through continuous edge
coordinates (a voxel i spans [i, i + 1)), so a box survives the round trip up to
the grid's resolution.
"""

import numpy as np


def _orig_to_cache_z(o, n_slices, n_cache):
    """Continuous slice coordinate, original -> cache (linear when subsampled)."""
    if n_cache == n_slices:
        return o
    return o * (n_cache - 1) / max(n_slices - 1, 1)


def _cache_to_orig_z(c, n_slices, n_cache):
    if n_cache == n_slices:
        return c
    return c * max(n_slices - 1, 1) / (n_cache - 1)


def to_volume(box, n_slices, n_cache, rows, cols, depth, size):
    """Original voxel box -> edge coordinates (z0, z1, y0, y1, x0, x1) in the volume grid."""
    s = max(rows, cols)
    oy, ox = (s - rows) // 2, (s - cols) // 2
    # Slice edges in cache coordinates: a slice o covers [o - 0.5, o + 0.5] around its centre.
    cz0 = _orig_to_cache_z(box["z0"] - 0.5, n_slices, n_cache) + 0.5
    cz1 = _orig_to_cache_z(box["z1"] + 0.5, n_slices, n_cache) + 0.5
    fz, fs = depth / n_cache, size / s
    return (cz0 * fz, cz1 * fz, (box["y0"] + oy) * fs, (box["y1"] + 1 + oy) * fs,
            (box["x0"] + ox) * fs, (box["x1"] + 1 + ox) * fs)


def box_mask(edges, depth, size):
    """(D, S, S) bool mask of the voxels whose centre lies inside the edge box.

    At least one voxel per axis: a box thinner than a voxel still marks the
    voxel it falls in.
    """
    m = np.zeros((depth, size, size), bool)
    sl = []
    for lo, hi, n in zip(edges[0::2], edges[1::2], (depth, size, size)):
        a = int(np.ceil(lo - 0.5))
        b = int(np.floor(hi - 0.5))
        if b < a:
            a = b = int(np.clip(np.floor((lo + hi) / 2), 0, n - 1))
        sl.append(slice(max(a, 0), min(b, n - 1) + 1))
    m[tuple(sl)] = True
    return m


def from_volume(vbox, n_slices, n_cache, rows, cols, depth, size):
    """Volume voxel box (inclusive indices) -> original voxel box (inclusive), clipped."""
    s = max(rows, cols)
    oy, ox = (s - rows) // 2, (s - cols) // 2
    fz, fs = n_cache / depth, s / size
    cz0, cz1 = vbox["z0"] * fz - 0.5, (vbox["z1"] + 1) * fz - 0.5
    z0 = int(np.round(_cache_to_orig_z(cz0, n_slices, n_cache) + 0.5))
    z1 = int(np.round(_cache_to_orig_z(cz1, n_slices, n_cache) - 0.5))
    out = dict(z0=z0, z1=max(z0, z1),
               y0=int(np.floor(vbox["y0"] * fs)) - oy, y1=int(np.ceil((vbox["y1"] + 1) * fs)) - 1 - oy,
               x0=int(np.floor(vbox["x0"] * fs)) - ox, x1=int(np.ceil((vbox["x1"] + 1) * fs)) - 1 - ox)
    for k, n in (("z", n_slices), ("y", rows), ("x", cols)):
        out[f"{k}0"] = int(np.clip(out[f"{k}0"], 0, n - 1))
        out[f"{k}1"] = int(np.clip(out[f"{k}1"], out[f"{k}0"], n - 1))
    return out
