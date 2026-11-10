"""
Helpers for using NCS seismic-foundation-model embeddings alongside the RGT workflow.

NCS output (from NCS_models' run_inference): an xarray/Zarr group (e.g. "dir0") with
  features (inline, xline, time_depth, feature)  plus per-channel mean/std.
Coordinates are 0-based SAMPLE INDICES of patch centres along the SEG-Y's sorted inline / crossline / sample axes
(NCS.inference.tiling.compute_densified_axis: coord = phase_offset + patch_size//2 + k*patch_size; with patch 16 and
densify d the step is 16/d and the first centre is 8). So feature coord c on the inline axis is ilines[c] (segyio's
sorted inline numbers), and on the time axis it is samples[c].

Everything here works on the feature grid in chunks (the cube can be many GB) and maps results onto the seismic grid
(inline, crossline, time) used by rgt_lfm.read_segy_cube: linear interpolation for continuous values, nearest
neighbour for categorical labels. Beyond the first/last patch centre values are held constant.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Registration: feature grid -> seismic grid
# ---------------------------------------------------------------------------
def axis_maps(coords, n):
    """For seismic indices 0..n-1: fractional feature index (linear) and nearest feature index."""
    coords = np.asarray(coords, float)
    if np.any(np.diff(coords) <= 0):
        raise ValueError("feature coordinates must be strictly increasing")
    frac = np.interp(np.arange(n), coords, np.arange(len(coords)))
    nearest = np.clip(np.rint(frac).astype(int), 0, len(coords) - 1)
    return frac, nearest


def describe_registration(ds, ilines, xlines, samples):
    """Human-readable check of how the feature grid sits on the seismic grid."""
    out = []
    for name, ax, labels, unit in (("inline", "inline", ilines, ""), ("xline", "xline", xlines, ""),
                                   ("time_depth", "time_depth", samples, " ms")):
        c = np.asarray(ds[ax].values)
        if c.max() >= len(labels):
            raise ValueError(f"feature {name} coordinates reach {c.max()}, beyond the seismic axis length {len(labels)}: "
                             "are the feature cube and SEG-Y from the same volume?")
        step = np.unique(np.diff(c))
        out.append(f"{name:10s}: {len(c)} positions, sample index {c[0]}–{c[-1]} (step {step.tolist()}) "
                   f"= {labels[c[0]]:g}–{labels[c[-1]]:g}{unit} of {labels[0]:g}–{labels[-1]:g}{unit}")
    return "\n".join(out)


def to_seismic_grid(vol, maps, order=1):
    """
    vol: (n_il_f, n_xl_f, n_t_f) on the feature grid. maps: ((frac_il, near_il), (frac_xl, near_xl), (frac_t, near_t)).
    order=1: separable linear interpolation (continuous values); order=0: nearest (categorical labels).
    Returns (NI, NX, NT).
    """
    vol = np.asarray(vol)
    if order == 0:
        return vol[np.ix_(maps[0][1], maps[1][1], maps[2][1])]
    out = vol.astype(np.float32)
    for axis, (frac, _) in enumerate(maps):
        n = out.shape[axis]
        i0 = np.clip(np.floor(frac).astype(int), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        w = (frac - i0).astype(np.float32)
        shape = [1, 1, 1]; shape[axis] = -1
        out = np.take(out, i0, axis=axis) * (1 - w.reshape(shape)) + np.take(out, i1, axis=axis) * w.reshape(shape)
    return out


# ---------------------------------------------------------------------------
# Chunked access to the feature cube
# ---------------------------------------------------------------------------
def iter_inline_chunks(features, chunk=8):
    """Yield (start, stop, numpy block (c, n_xl, n_t, F)) along the inline axis of an xarray DataArray."""
    n = features.sizes["inline"]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        yield a, b, np.asarray(features.isel(inline=slice(a, b)).values, dtype=np.float32)


def l2n(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def sample_vectors(features, n_samples=100_000, seed=0, normalise=True, chunk=8):
    """Random voxel embeddings, drawn chunk by chunk (never loads the whole cube)."""
    rng = np.random.default_rng(seed)
    sizes = [features.sizes[d] for d in ("inline", "xline", "time_depth")]
    total = int(np.prod(sizes))
    per_inline = n_samples / sizes[0]
    out = []
    for a, b, blk in iter_inline_chunks(features, chunk):
        flat = blk.reshape(-1, blk.shape[-1])
        k = min(len(flat), int(round(per_inline * (b - a))))
        out.append(flat[rng.choice(len(flat), k, replace=False)])
    X = np.concatenate(out)
    return l2n(X) if normalise else X


def pca_transform(features, pca, normalise=True, chunk=8, dtype=np.float32):
    """Project every voxel onto the fitted PCA: returns (n_il_f, n_xl_f, n_t_f, n_components)."""
    s = features.sizes
    out = np.empty((s["inline"], s["xline"], s["time_depth"], pca.n_components_), dtype)
    for a, b, blk in iter_inline_chunks(features, chunk):
        flat = blk.reshape(-1, blk.shape[-1])
        flat = l2n(flat) if normalise else flat
        out[a:b] = pca.transform(flat).reshape(b - a, s["xline"], s["time_depth"], -1).astype(dtype)
    return out


def cosine_to(features, ref_vec, chunk=8):
    """Cosine similarity of every voxel embedding to one reference embedding: (n_il_f, n_xl_f, n_t_f)."""
    r = l2n(np.asarray(ref_vec, np.float32))
    s = features.sizes
    out = np.empty((s["inline"], s["xline"], s["time_depth"]), np.float32)
    for a, b, blk in iter_inline_chunks(features, chunk):
        out[a:b] = (l2n(blk) * r).sum(-1)
    return out


def lateral_similarity(features, chunk=8):
    """
    Cosine similarity between laterally neighbouring embeddings (same time index):
    returns (sim_il (n_il_f-1, n_xl_f, n_t_f), sim_xl (n_il_f, n_xl_f-1, n_t_f)).
    Low values = the seismic character changes abruptly sideways (faults, facies boundaries, chaos);
    an embedding-based counterpart of amplitude coherence.
    """
    s = features.sizes
    sim_il = np.empty((s["inline"] - 1, s["xline"], s["time_depth"]), np.float32)
    sim_xl = np.empty((s["inline"], s["xline"] - 1, s["time_depth"]), np.float32)
    prev = None
    for a, b, blk in iter_inline_chunks(features, chunk):
        n = l2n(blk)
        sim_xl[a:b] = (n[:, 1:] * n[:, :-1]).sum(-1)
        if prev is not None:
            sim_il[a - 1] = (prev * n[0]).sum(-1)
        sim_il[a:b - 1] = (n[1:] * n[:-1]).sum(-1)
        prev = n[-1]
    return sim_il, sim_xl


def node_from_links(sim_il, sim_xl):
    """Per-position minimum of the neighbouring-link similarities (a discontinuity map on the feature grid)."""
    NI, NX, NT = sim_xl.shape[0], sim_il.shape[1], sim_il.shape[2]
    w = np.ones((NI, NX, NT), np.float32)
    w[:-1] = np.minimum(w[:-1], sim_il); w[1:] = np.minimum(w[1:], sim_il)
    w[:, :-1] = np.minimum(w[:, :-1], sim_xl); w[:, 1:] = np.minimum(w[:, 1:], sim_xl)
    return w
