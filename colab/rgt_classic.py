"""
Classical (non-deep-learning) RGT by dip-based global flattening, and a hybrid with a DNN RGT.

Idea (Lomask et al. 2006 "flattening without picking"; Wu & Fomel 2018 least-squares horizons; the principle
behind global-interpretation tools such as PaleoScan or HorizonCube):
  1. local_dips: estimate the reflection dip everywhere (time samples per inline / crossline step) and a 0–1
     quality (how well a single plane explains the local seismic).
  2. flatten: a stack of K surfaces z_k(inline, crossline) (time-sample index). Each is solved by weighted least
     squares so its slopes match the dips sampled ON the surface, re-sampling the dips and re-solving a few times
     (Gauss-Newton). Surfaces are pinned at a reference trace, optionally pinned to picked horizons, optionally
     pulled towards a reference set (e.g. DNN surfaces), and never allowed to cross.
  3. rgt_from_surfaces: RGT = which surface you're on (0 at the top surface, 1 at the bottom), interpolated.

Limitation of the pure dip method: dips carry no information across faults (where reflections are cut) or in
chaotic zones, so surfaces integrate smoothly through those with no fault throw. The hybrid uses the DNN RGT as the
starting point and as a reference that takes over where the dip quality is low.

Arrays: volumes are (inline, crossline, time); surfaces are (K, inline, crossline) in time-sample index.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Local dips
# ---------------------------------------------------------------------------
def local_dips(cube, sigma_grad=1.0, sigma_win=(3.0, 3.0, 2.0), max_dip=4.0, weight=None):
    """
    Least-squares plane-wave slopes: p minimises <(g_il + p g_t)^2>, so p = -<g_il g_t> / <g_t g_t> (same for q),
    with <> a Gaussian window (sigma_win in inline, crossline, time samples).
    Returns p, q (time samples per inline / crossline step, positive = deepening) and quality 0..1
    (1 = a single plane explains the gradients perfectly; low = chaotic, noisy or faulted).
    weight: optional (NI, NX, NT) 0..1 weights for the averaging window (normalised convolution), e.g. from
    voxel_weight_from_links, so dips next to a fault are estimated only from reflections on their own side.
    """
    from scipy.ndimage import gaussian_filter
    c = np.asarray(cube, np.float32)
    c = c / (np.std(c) + 1e-12)
    gi = gaussian_filter(c, sigma_grad, order=(1, 0, 0))
    gx = gaussian_filter(c, sigma_grad, order=(0, 1, 0))
    gt = gaussian_filter(c, sigma_grad, order=(0, 0, 1))
    if weight is None:
        S = lambda a: gaussian_filter(a, sigma_win)
    else:
        # exclude low-weight samples (faults) from the window, including the fault-contaminated gradients there
        w = np.asarray(weight, np.float32)
        norm = gaussian_filter(w, sigma_win) + 1e-6
        S = lambda a: gaussian_filter(a * w, sigma_win) / norm
    Stt = S(gt * gt) + 1e-6
    Sit, Sxt = S(gi * gt), S(gx * gt)
    p = np.clip(-Sit / Stt, -max_dip, max_dip).astype(np.float32)
    q = np.clip(-Sxt / Stt, -max_dip, max_dip).astype(np.float32)
    # residual energy of the plane fit, relative to total gradient energy
    Sii, Sxx = S(gi * gi), S(gx * gx)
    resid = (Sii - Sit ** 2 / Stt) + (Sxx - Sxt ** 2 / Stt)                 # what the plane fails to explain
    q_raw = 1.0 - resid / (Sii + Sxx + Stt)                                  # 1 = plane, ~1/3 = isotropic noise
    quality = np.clip((q_raw - 1 / 3) / (2 / 3), 0, 1).astype(np.float32)
    return p, q, quality


def link_coherence(cube, p, q, sigma_t=4.0, sigma_bg=4.0, widen=2):
    """
    How well each trace matches its neighbour once shifted by the local dip (normalised cross-correlation in a
    Gaussian time window). ~1 = continuous reflections; low = a fault (different events side by side) or chaos.
    The shift uses the BACKGROUND dip (dips smoothed laterally over sigma_bg traces): at a fault the local dip
    estimate partly "explains" the step as a steep dip, the background dip doesn't, so coherence drops properly.
    Low-coherence links are widened by `widen` traces each side (minimum filter): the dips right next to a fault are
    contaminated by it (the dip window straddles the step), so those links are distrusted too.
    Returns (ci, cx): links between inline neighbours (NI-1, NX, NT) and crossline neighbours (NI, NX-1, NT), 0..1.
    """
    from scipy.ndimage import gaussian_filter1d, gaussian_filter
    if sigma_bg:
        p = gaussian_filter(p, (sigma_bg, sigma_bg, 0))
        q = gaussian_filter(q, (sigma_bg, sigma_bg, 0))
    c = np.asarray(cube, np.float32) / (np.std(cube) + 1e-12)
    NT = c.shape[2]
    t = np.arange(NT, dtype=np.float32)

    def coh(a, b, d):
        tq = np.clip(t + d, 0, NT - 1.000001)                          # event at t on a sits at t + d on b
        t0 = np.floor(tq).astype(np.int32); f = tq - t0
        bs = np.take_along_axis(b, t0, 2) * (1 - f) + np.take_along_axis(b, np.minimum(t0 + 1, NT - 1), 2) * f
        S = lambda x: gaussian_filter1d(x, sigma_t, axis=2)
        return np.clip(S(a * bs) / np.sqrt(S(a * a) * S(bs * bs) + 1e-12), 0, 1).astype(np.float32)

    from scipy.ndimage import minimum_filter
    ci = coh(c[:-1], c[1:], 0.5 * (p[:-1] + p[1:]))
    cx = coh(c[:, :-1], c[:, 1:], 0.5 * (q[:, :-1] + q[:, 1:]))
    if widen:
        ci = minimum_filter(ci, size=(2 * widen + 1, 2 * widen + 1, 3))
        cx = minimum_filter(cx, size=(2 * widen + 1, 2 * widen + 1, 3))
    return ci, cx


def voxel_weight_from_links(links, power=4):
    """Per-sample weight 0..1: the lowest coherence of the links touching that sample, to the power `power`."""
    ci, cx = links
    NI, NX, NT = ci.shape[0] + 1, cx.shape[1] + 1, ci.shape[2]
    w = np.ones((NI, NX, NT), np.float32)
    w[:-1] = np.minimum(w[:-1], ci); w[1:] = np.minimum(w[1:], ci)
    w[:, :-1] = np.minimum(w[:, :-1], cx); w[:, 1:] = np.minimum(w[:, 1:], cx)
    return w ** power


def fault_aware_dips(cube, **kw):
    """Dips, quality and link coherence, with the dips re-estimated away from faults (two passes).
    Returns p, q, quality, links."""
    p, q, quality = local_dips(cube, **kw)
    links = link_coherence(cube, p, q)
    p2, q2, quality2 = local_dips(cube, weight=voxel_weight_from_links(links), **kw)
    return p2, q2, np.minimum(quality, quality2), links


# ---------------------------------------------------------------------------
# 2. Flattening
# ---------------------------------------------------------------------------
def _sample(vol, z):
    """Linear interpolation of vol (NI, NX, NT) along time at surfaces z (K, NI, NX)."""
    NI, NX, NT = vol.shape
    zc = np.clip(z, 0, NT - 1.000001)
    t0 = np.floor(zc).astype(np.int32)
    f = (zc - t0).astype(np.float32)
    I = np.arange(NI)[None, :, None]
    X = np.arange(NX)[None, None, :]
    return vol[I, X, t0] * (1 - f) + vol[I, X, np.minimum(t0 + 1, NT - 1)] * f


def _dct_precond(shape, wbar, mu):
    from scipy.fft import dctn, idctn
    K, NI, NX = shape
    li = 2 - 2 * np.cos(np.pi * np.arange(NI) / NI)
    lx = 2 - 2 * np.cos(np.pi * np.arange(NX) / NX)
    denom = (wbar * (li[:, None] + lx[None, :]) + mu).astype(np.float32)[None]
    return lambda r: idctn(dctn(r, type=2, axes=(1, 2), norm="ortho", workers=-1) / denom,
                           type=2, axes=(1, 2), norm="ortho", workers=-1)


def _solve(wi, wx, bi, bx, diag, rhs_diag, z0, n_iter):
    """Batched preconditioned CG for  D_i^T wi D_i z + D_x^T wx D_x z + diag*z = D_i^T wi bi + D_x^T wx bx + rhs_diag."""
    def DiT(v):
        out = np.zeros(z0.shape, np.float32); out[:, :-1] -= v; out[:, 1:] += v; return out

    def DxT(v):
        out = np.zeros(z0.shape, np.float32); out[:, :, :-1] -= v; out[:, :, 1:] += v; return out

    def A(z):
        return DiT(wi * (z[:, 1:] - z[:, :-1])) + DxT(wx * (z[:, :, 1:] - z[:, :, :-1])) + diag * z

    b = DiT(wi * bi) + DxT(wx * bx) + rhs_diag
    M = _dct_precond(z0.shape, float(0.5 * (wi.mean() + wx.mean())), float(max(diag.mean(), 1e-3)))
    z = z0.astype(np.float32).copy()
    r = b - A(z)
    y = M(r)
    d = y.copy()
    ry = (r * y).sum(axis=(1, 2))
    for _ in range(n_iter):
        Ad = A(d)
        alpha = ry / np.maximum((d * Ad).sum(axis=(1, 2)), 1e-20)
        z += alpha[:, None, None] * d
        r -= alpha[:, None, None] * Ad
        y = M(r)
        ry_new = (r * y).sum(axis=(1, 2))
        d = y + (ry_new / np.maximum(ry, 1e-20))[:, None, None] * d
        ry = ry_new
    return z


def assign_guides(guides, ref_ij, NT, n_surfaces):
    """Which surface each picked horizon pins: the one whose t0 is closest to the pick nearest the reference trace."""
    ri, rj = ref_ij
    t0 = np.linspace(0, NT - 1, n_surfaces)
    out = []
    for g in guides or []:
        pts = np.argwhere(np.isfinite(g))
        if len(pts):
            ni, nj = pts[np.argmin((pts[:, 0] - ri) ** 2 + (pts[:, 1] - rj) ** 2)]
            out.append((int(np.argmin(np.abs(t0 - g[ni, nj]))), g))
    return out


def flatten(p, q, quality, ref_ij, n_surfaces=200, init=None, guides=None, reference=None,
            links=None, link_threshold=0.7, ref_scale=300.0, ref_weight_lowq=2.0, low_quality=0.5, cut_tol=None, guide_weight=10.0, anchor_weight=10.0,
            outer_iters=8, cg_iters=25, min_gap=0.05, verbose=True):
    """
    Solve the surface stack. Returns z (K, NI, NX) in time-sample index, ordered top to bottom.

    ref_ij    : (i, j) reference trace; surface k passes through t0_k there (t0 evenly spaced over the trace)
    init      : optional starting surfaces (K, NI, NX), e.g. from a DNN RGT (otherwise flat)
    guides    : optional list of picked horizons as time-sample index maps (NaN = not picked); each pins one surface
    reference : optional surfaces (K, NI, NX) to combine with the dips (a DNN RGT, or picks spread by
                reference_from_guides). How the two are combined:
                  * ref_scale (traces): the reference is trusted for lateral features broader than this; the dips
                    decide everything finer
                  * ref_weight_lowq: extra pull where the dip quality is below `low_quality` (chaotic zones),
                    rising linearly to ref_weight_lowq at quality 0; none at all where quality is above it
                  * cut_tol (samples, optional): also cut links where the reference jumps by more than this beyond
                    what the dips allow
    links     : optional (ci, cx) from link_coherence. Each dip link between neighbouring traces is weighted by
                ((coherence - link_threshold) / (1 - link_threshold))^2, zero below the threshold, so links across
                faults and through chaos are cut (a soft power isn't enough: unrelated reflections are sometimes
                fairly coherent by chance): the dips then
                integrate within continuous blocks, and the reference (or anchors/picks) sets the blocks' offsets.
    """
    import time
    NI, NX, NT = p.shape
    ri, rj = ref_ij
    K = n_surfaces
    t0 = np.linspace(0, NT - 1, K).astype(np.float32)
    z = np.broadcast_to(t0[:, None, None], (K, NI, NX)).copy() if init is None else init.astype(np.float32).copy()

    diag = np.zeros((K, NI, NX), np.float32)
    rhs = np.zeros_like(diag)
    diag[:, ri, rj] += anchor_weight
    rhs[:, ri, rj] += anchor_weight * t0
    for k, g in assign_guides(guides, ref_ij, NT, K):
        m = np.isfinite(g)
        diag[k][m] += guide_weight
        rhs[k][m] += guide_weight * g[m]
        if init is None and reference is None:                 # start the pinned surface at the picks
            z[k] = np.where(m, g, z[k])
    t_start = time.time()
    for it in range(outer_iters):
        P, Q, W = _sample(p, z), _sample(q, z), _sample(quality, z) ** 2 + 1e-3
        bi, bx = 0.5 * (P[:, 1:] + P[:, :-1]), 0.5 * (Q[:, :, 1:] + Q[:, :, :-1])
        wi, wx = 0.5 * (W[:, 1:] + W[:, :-1]), 0.5 * (W[:, :, 1:] + W[:, :, :-1])
        if links is not None:
            cut = lambda c: np.clip((c - link_threshold) / (1 - link_threshold), 0, 1) ** 2
            wi = wi * cut(_sample(links[0], 0.5 * (z[:, 1:] + z[:, :-1])))
            wx = wx * cut(_sample(links[1], 0.5 * (z[:, :, 1:] + z[:, :, :-1])))
        d, r = diag.copy(), rhs.copy()
        d += 1e-4; r += 1e-4 * z                                       # keeps disconnected blocks well-posed
        if reference is not None:
            if cut_tol:
                mi = np.abs((reference[:, 1:] - reference[:, :-1]) - bi)
                mx = np.abs((reference[:, :, 1:] - reference[:, :, :-1]) - bx)
                wi = wi * np.exp(-(np.maximum(mi - cut_tol, 0) / cut_tol) ** 2)
                wx = wx * np.exp(-(np.maximum(mx - cut_tol, 0) / cut_tol) ** 2)
            lam = float(np.mean(W)) * (2 * np.pi / ref_scale) ** 2 if ref_scale else 0.0
            poor = np.clip((low_quality - _sample(quality, reference)) / low_quality, 0, 1)
            wr = lam + ref_weight_lowq * poor
            d += wr; r += wr * reference
        z_new = _solve(wi, wx, bi, bx, d, r, z, cg_iters)
        change = float(np.abs(z_new - z).mean())
        z = z_new
        z = np.maximum.accumulate(z + np.arange(K, dtype=np.float32)[:, None, None] * min_gap, axis=0) \
            - np.arange(K, dtype=np.float32)[:, None, None] * min_gap   # never cross
        if verbose:
            print(f"   iteration {it + 1}/{outer_iters}: mean surface change {change:.3f} samples  ({time.time() - t_start:.0f} s)")
    return z


def reference_from_guides(z, guides, ref_ij, NT):
    """
    Turn picked horizons into a reference stack: each picked surface's offset from z (picks minus surface) is filled
    across the map (linear inside the picks, nearest outside), then interpolated between the picked levels
    (constant above the top pick and below the bottom one), and added to z. Gives every surface the fault throws
    and corrections that the picks reveal, not just the picked ones.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    K, NI, NX = z.shape
    pairs = sorted(assign_guides(guides, ref_ij, NT, K), key=lambda x: x[0])
    if not pairs:
        return z.copy()
    II, XX = np.meshgrid(np.arange(NI), np.arange(NX), indexing="ij")
    ks, offs = [], []
    for k, g in pairs:
        m = np.isfinite(g)
        pts, val = np.c_[II[m], XX[m]], (g - z[k])[m]
        full = LinearNDInterpolator(pts, val)(II, XX)
        hole = np.isnan(full)
        if hole.any():
            full[hole] = NearestNDInterpolator(pts, val)(II[hole], XX[hole])
        ks.append(k); offs.append(full)
    ks, offs = np.array(ks), np.array(offs)
    kk = np.arange(K)
    out = z.copy()
    for k in kk:
        if k <= ks[0]:
            out[k] += offs[0]
        elif k >= ks[-1]:
            out[k] += offs[-1]
        else:
            j = np.searchsorted(ks, k) - 1
            f = (k - ks[j]) / (ks[j + 1] - ks[j])
            out[k] += (1 - f) * offs[j] + f * offs[j + 1]
    return np.maximum.accumulate(out, axis=0)


# ---------------------------------------------------------------------------
# 3. RGT from surfaces, and surfaces from an RGT
# ---------------------------------------------------------------------------
def rgt_from_surfaces(z, NT):
    """RGT (NI, NX, NT) in 0..1: level of each sample in the surface stack (linear between surfaces)."""
    K, NI, NX = z.shape
    u = np.linspace(0, 1, K)
    t = np.arange(NT, dtype=np.float32)
    out = np.empty((NI, NX, NT), np.float32)
    zz = np.maximum.accumulate(z, axis=0) + np.arange(K)[:, None, None] * 1e-4
    for i in range(NI):
        for j in range(NX):
            out[i, j] = np.interp(t, zz[:, i, j], u)
    return out


def surfaces_from_rgt(rgt, ref_ij, n_surfaces=200):
    """Surfaces (K, NI, NX) of an RGT volume, labelled like flatten(): surface k passes through t0_k at ref_ij."""
    NI, NX, NT = rgt.shape
    t0 = np.linspace(0, NT - 1, n_surfaces)
    prof = np.maximum.accumulate(rgt[ref_ij[0], ref_ij[1]]) + np.arange(NT) * 1e-7
    levels = np.interp(t0, np.arange(NT), prof)
    pr = np.maximum.accumulate(rgt, axis=2) + np.arange(NT) * 1e-7
    out = np.empty((n_surfaces, NI, NX), np.float32)
    for k, u in enumerate(levels):
        jj = np.clip((pr < u).sum(axis=2), 1, NT - 1)
        p0 = np.take_along_axis(pr, (jj - 1)[..., None], 2)[..., 0]
        p1 = np.take_along_axis(pr, jj[..., None], 2)[..., 0]
        out[k] = jj - 1 + np.clip((u - p0) / np.maximum(p1 - p0, 1e-12), 0, 1)
    return out
