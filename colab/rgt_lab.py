"""
Experimental horizon tools for RGT-Est results (used by RGT-Est_Horizon_Lab_Colab.ipynb).

All volumes are (inline, crossline, time) numpy arrays; horizons are (inline, crossline) TWT maps in ms,
NaN where undefined; `samples` is the TWT of each time sample (ms).

- read_rokdoc_horizon : RokDoc horizon .txt (Inline, Crossline, TWT) onto the seismic grid
- sparsify            : keep only every Nth inline and crossline of a horizon (mimics sparse manual picks)
- horizon_errors      : vertical miss between interpreted horizons and the best-matching RGT surfaces
- flatten_section     : seismic resampled onto RGT levels (Wheeler / flattened view)
- continuity          : how flat a flattened section is, per RGT level (adjacent-trace correlation)
- snap_horizons       : move horizons to the nearest peak / trough / zero crossing, robustly
- confidence_volume   : seismic dip vs RGT-contour dip (ms per trace), reflector strength, confidence
- proportional_repair : blend towards proportional layering between guidance horizons where confidence is low
"""

import os
import re
import numpy as np


# ---------------------------------------------------------------------------
# Horizons in and out
# ---------------------------------------------------------------------------
def read_rokdoc_horizon(path, ilines, xlines, samples):
    name, ztype, rows = os.path.splitext(os.path.basename(path))[0], None, []
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = re.match(r'\s*Horizon\s+"(.+?)"', line)
            if m:
                name = m.group(1)
            if "Z Type" in line:
                ztype = line.split(":")[-1].strip()
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    rows.append([float(v) for v in parts[:3]])
                except ValueError:
                    pass
    a = np.array(rows, dtype=float).reshape(-1, 3)
    hil, hxl, hz = np.round(a[:, 0]), np.round(a[:, 1]), a[:, 2]
    ok = np.isfinite(hz) & (hz > -999) & np.isin(hil, ilines) & np.isin(hxl, xlines) \
         & (hz >= samples[0]) & (hz <= samples[-1])
    t = np.full((len(ilines), len(xlines)), np.nan)
    t[np.searchsorted(ilines, hil[ok]), np.searchsorted(xlines, hxl[ok])] = hz[ok]
    return dict(name=name, t=t, ztype=ztype, n_file=len(a))


def sparsify(t, every):
    """Keep the horizon only on every `every`-th inline and every `every`-th crossline (a grid of picked lines)."""
    if not every or every <= 1:
        return t.copy()
    keep = np.zeros(t.shape, bool)
    keep[::every, :] = True
    keep[:, ::every] = True
    return np.where(keep, t, np.nan)


def monotonic(profile):
    """Make an RGT trace strictly increasing downwards (needed to invert it)."""
    return np.maximum.accumulate(profile) + np.arange(profile.shape[-1]) * 1e-7


def trace_surfaces(rgt, levels, samples):
    """TWT (ms) of each RGT level on every trace: (n_levels, NI, NX). Linear inversion of each trace."""
    NI, NX, NT = rgt.shape
    prof = np.maximum.accumulate(rgt, axis=2) + np.arange(NT) * 1e-7
    out = np.empty((len(levels), NI, NX), np.float32)
    # vectorised per level: index of the first sample at or above the level
    for k, u in enumerate(levels):
        j = np.clip((prof < u).sum(axis=2), 1, NT - 1)
        p0 = np.take_along_axis(prof, (j - 1)[..., None], 2)[..., 0]
        p1 = np.take_along_axis(prof, j[..., None], 2)[..., 0]
        f = np.clip((u - p0) / np.maximum(p1 - p0, 1e-12), 0, 1)
        out[k] = samples[0] + (j - 1 + f) * (samples[1] - samples[0])
    return out


def horizon_errors(rgt, horizons, samples, mask=None):
    """For each interpreted horizon: (mean |miss| ms, % within 2 samples), using its best-matching RGT level.
    `mask` (bool NI x NX) restricts where the miss is measured (e.g. only traces that were NOT picked)."""
    dt = samples[1] - samples[0]
    res = []
    for h in horizons:
        t = h["t"]
        i, j = np.nonzero(np.isfinite(t))
        ti = np.clip(np.round((t[i, j] - samples[0]) / dt).astype(int), 0, len(samples) - 1)
        level = np.median(rgt[i, j, ti])
        pred = trace_surfaces(rgt, [level], samples)[0]
        m = np.isfinite(t) if mask is None else (np.isfinite(t) & mask)
        err = np.abs(pred[m] - t[m])
        res.append((float(err.mean()), float(100 * (err <= 2 * dt).mean())))
    return res


# ---------------------------------------------------------------------------
# Flattened (Wheeler) view and continuity
# ---------------------------------------------------------------------------
def flatten_section(seis2d, rgt2d, samples, levels):
    """seis2d, rgt2d: (n_traces, NT). Returns flattened amplitudes (n_levels, n_traces) and the TWT of each level."""
    n = seis2d.shape[0]
    flat = np.full((len(levels), n), np.nan, np.float32)
    t_at = np.full((len(levels), n), np.nan, np.float32)
    for j in range(n):
        tt = np.interp(levels, monotonic(rgt2d[j]), samples, left=np.nan, right=np.nan)
        t_at[:, j] = tt
        ok = np.isfinite(tt)
        flat[ok, j] = np.interp(tt[ok], samples, seis2d[j])
    return flat, t_at


def continuity(flat, window=21, lateral=51):
    """How flat a flattened section is, per RGT level: the correlation of each trace with the average of the
    `lateral` traces around it, in a sliding window of `window` levels, averaged over traces.
    1 = the RGT follows the reflections; lower = events drift or break across the section."""
    from scipy.ndimage import uniform_filter1d
    ok = np.isfinite(flat)
    a = np.where(ok, flat, 0.0).astype(float)
    cnt = uniform_filter1d(ok.astype(float), lateral, axis=1, mode="nearest")
    ref = uniform_filter1d(a, lateral, axis=1, mode="nearest") / np.maximum(cnt, 1e-12)
    ref = np.where(ok, ref, 0.0)
    f = lambda x: uniform_filter1d(x, window, axis=0, mode="nearest")
    n = f(ok.astype(float)) + 1e-12
    ma, mr = f(a) / n, f(ref) / n
    cov = f(a * ref) / n - ma * mr
    va, vr = f(a * a) / n - ma ** 2, f(ref * ref) / n - mr ** 2
    r = cov / np.sqrt(np.maximum(va * vr, 1e-20))
    r[n < 0.5] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(r, axis=1)


def dominant_frequency(cube, samples, interval=None, n_traces=400, seed=0):
    """Peak of the average amplitude spectrum (Hz) over random traces, within `interval` (ms) if given."""
    NI, NX, NT = cube.shape
    dt = (samples[1] - samples[0]) / 1000.0
    m = np.ones(NT, bool) if interval is None else (samples >= interval[0]) & (samples <= interval[1])
    rng = np.random.default_rng(seed)
    tr = cube[rng.integers(0, NI, n_traces), rng.integers(0, NX, n_traces)][:, m]
    tr = (tr - tr.mean(1, keepdims=True)) * np.hanning(tr.shape[1])
    spec = np.abs(np.fft.rfft(tr, n=max(256, 2 * tr.shape[1]), axis=1)).mean(0)
    f = np.fft.rfftfreq(max(256, 2 * tr.shape[1]), dt)
    spec[f < 3] = 0
    return float(f[np.argmax(spec)])


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------
def _track_shifts(cand, valid, seed_tol, max_jump):
    """
    Choose one candidate shift per trace so the shift map is continuous: grow from seeds (traces whose nearest
    event is within seed_tol of the RGT horizon) to neighbours, always taking the candidate closest to the
    neighbour's accepted shift, smoothest links first. Where no candidate is within max_jump, the neighbour's
    shift is carried on (trend kept). Returns shift (samples) and status: 0 no event, 1 tracked, 2 trend kept, 3 seed.
    """
    import heapq
    K, NI, NX = cand.shape
    absd = np.where(np.isfinite(cand), np.abs(cand), np.inf)
    k0 = np.argmin(absd, axis=0)
    s0 = np.take_along_axis(cand, k0[None], 0)[0]                          # nearest event per trace
    shift = np.full((NI, NX), np.nan)
    status = np.zeros((NI, NX), np.int8)
    seeds = valid & np.isfinite(s0) & (np.abs(s0) <= seed_tol)
    if not seeds.any():                                                    # nothing unambiguous: best 5% of traces
        m = np.where(valid & np.isfinite(s0), np.abs(s0), np.nan)
        if np.isfinite(m).any():
            seeds = valid & np.isfinite(s0) & (np.abs(s0) <= np.nanpercentile(m, 5) + 1e-9)
            # those seeds can sit on different cycles (just above / just below the horizon): keep the majority
            sv = s0[seeds]
            edges = np.arange(sv.min() - max_jump, sv.max() + 2 * max_jump, max_jump)
            hist, edges = np.histogram(sv, bins=edges)
            mode = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
            seeds &= np.abs(s0 - mode) <= max_jump
    heap = [(0.0, int(i), int(j), float(s0[i, j])) for i, j in zip(*np.nonzero(seeds))]
    heapq.heapify(heap)
    for _, i, j, v in heap:
        shift[i, j], status[i, j] = v, 3
    done = seeds.copy()
    while heap:
        _, i, j, v = heapq.heappop(heap)
        for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= a < NI and 0 <= b < NX and valid[a, b] and not done[a, b]:
                c = cand[:, a, b]
                d = np.where(np.isfinite(c), np.abs(c - v), np.inf)
                k = int(np.argmin(d))
                done[a, b] = True
                if d[k] <= max_jump:
                    shift[a, b], status[a, b] = c[k], 1
                    heapq.heappush(heap, (float(d[k]), a, b, float(c[k])))
                else:
                    shift[a, b], status[a, b] = v, 2                       # no event close: carry the trend
                    heapq.heappush(heap, (float(max_jump) + 1.0, a, b, v))
    left = valid & ~done & np.isfinite(s0)                                 # unreachable islands: nearest event
    shift[left], status[left] = s0[left], 1
    return np.nan_to_num(shift), status


def snap_horizons(hz_ms, cube, samples, event="auto", window_ms=8.0, max_jump_ms=4.0):
    """
    Move each horizon (n_h, NI, NX, TWT ms) onto the nearest event in `cube` within ±window_ms.
    event: "peak", "trough", "zero" or "auto" (peak or trough, whichever the horizon mostly sits closer to).
    Robustness: rather than snapping each trace on its own (which cycle-skips where the horizon sits between two
    events), the snap is TRACKED: it grows from unambiguous traces to their neighbours, always taking the event
    closest to the neighbour's shift (max_jump_ms tolerance), so the result follows one event continuously.
    Horizons are never allowed to cross.
    Returns (snapped ms, list of per-horizon stats dicts).
    """
    NI, NX, NT = cube.shape
    dt = samples[1] - samples[0]
    w = max(1, int(round(window_ms / dt)))
    ks = np.arange(-w - 1, w + 2)                                           # one extra sample each side
    inwin = np.abs(ks) <= w
    II, XX = np.meshgrid(np.arange(NI), np.arange(NX), indexing="ij")
    out, stats = np.array(hz_ms, dtype=float), []
    for h in range(len(hz_ms)):
        idx = (np.asarray(hz_ms[h], float) - samples[0]) / dt
        valid = np.isfinite(idx)
        base = np.clip(np.round(np.where(valid, idx, 0)).astype(int), 0, NT - 1)
        raw_k = base[None] + ks[:, None, None]
        inside = (raw_k >= 1) & (raw_k <= NT - 2)                           # events touching the volume edge don't count
        cand = np.clip(raw_k, 0, NT - 1)
        amp = cube[II[None], XX[None], cand]                               # (2w+1, NI, NX)
        ev = event
        if ev == "auto":
            ev = "peak" if np.nanmedian(np.where(valid, cube[II, XX, base], np.nan)) >= 0 else "trough"
        # candidate event positions (offset from `base`, sub-sample), NaN where there's no event
        pos = np.full(amp.shape, np.nan)
        if ev in ("peak", "trough"):
            s = amp if ev == "peak" else -amp
            y0, y1, y2 = s[:-2], s[1:-1], s[2:]
            is_ext = (y1 >= y0) & (y1 >= y2) & inwin[1:-1, None, None] & inside[1:-1]
            den = y0 - 2 * y1 + y2
            frac = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0), -0.5, 0.5)
            pos[1:-1] = np.where(is_ext, ks[1:-1, None, None] + frac, np.nan)
        else:                                                               # zero crossings
            a0, a1 = amp[:-1], amp[1:]
            is_x = (np.sign(a0) * np.sign(a1) < 0) & inwin[:-1, None, None] & inside[:-1] & inside[1:]
            frac = np.where(np.abs(a0 - a1) > 1e-12, a0 / (a0 - a1), 0.0)
            pos[:-1] = np.where(is_x, ks[:-1, None, None] + np.clip(frac, 0, 1), np.nan)

        cand = base[None] + pos - np.where(valid, idx, 0)[None]              # candidate shifts (samples), NaN = none
        shift, status = _track_shifts(cand, valid, seed_tol=0.35 * w, max_jump=max_jump_ms / dt)
        found = status > 0
        jump = status == 2
        shift = np.where(valid & found, shift, 0.0)
        out[h] = np.where(valid, samples[0] + (idx + shift) * dt, np.nan)
        stats.append(dict(event=ev, mean_shift_ms=float(np.nanmean(np.abs(shift[valid])) * dt),
                          seeds_pct=float(100 * (status == 3).sum() / max(valid.sum(), 1)),
                          no_event_pct=float(100 * (~found & valid).sum() / max(valid.sum(), 1)),
                          kept_trend_pct=float(100 * (jump & valid).sum() / max(valid.sum(), 1))))
    order = np.argsort([np.nanmean(o) for o in out])
    orig = np.asarray(hz_ms, float)
    crossings = 0
    for a, b in zip(order[:-1], order[1:]):                                 # keep the stack in order
        bad = out[b] <= out[a] + 0.5 * dt
        crossings += int(np.nansum(bad))
        out[a][bad], out[b][bad] = orig[a][bad], orig[b][bad]
    return out, stats, crossings


# ---------------------------------------------------------------------------
# Confidence: RGT contours vs seismic reflector orientation
# ---------------------------------------------------------------------------
def confidence_volume(cube, rgt, dt_ms, step=2, sigma_grad=1.0, sigma_tensor=2.5, tol_ms_per_trace=0.25):
    """
    Do the RGT contours run parallel to the seismic reflections? Compares the two local dips.
    Returns a dict of (NI, NX, NT) float32 volumes:
      misfit     : |seismic dip - RGT-contour dip|, in ms of TWT per trace (inline and crossline combined)
      linearity  : how clearly layered the seismic is (structure-tensor planarity, 0..1); low = chaotic/faulted
      confidence : exp(-(misfit / tol)^2) * linearity, 0..1 (1 = clear reflections that the RGT follows)
    A misfit of 0.25 ms/trace adds up to 25 ms over 100 traces. Computed on a grid decimated by `step`
    and interpolated back.
    """
    from scipy.ndimage import gaussian_filter, zoom
    s = np.ascontiguousarray(cube[::step, ::step, ::step], dtype=np.float32)
    r = np.ascontiguousarray(rgt[::step, ::step, ::step], dtype=np.float32)
    s = s / (np.std(s) + 1e-12)
    g = [gaussian_filter(s, sigma_grad, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    T = np.empty(s.shape + (3, 3), np.float32)
    for a in range(3):
        for b in range(a, 3):
            T[..., a, b] = T[..., b, a] = gaussian_filter(g[a] * g[b], sigma_tensor)
    del g
    lam, vec = np.linalg.eigh(T)                                             # ascending eigenvalues
    del T
    n = vec[..., :, 2]                                                       # reflector normal
    lin = np.clip((lam[..., 2] - lam[..., 1]) / (lam[..., 2] + 1e-12), 0, 1)
    del vec, lam
    nt = np.where(np.abs(n[..., 2]) < 1e-3, 1e-3, n[..., 2])
    ds = np.clip(-n[..., :2] / nt[..., None], -5, 5)                          # seismic dip, samples per trace
    gr = np.stack([gaussian_filter(r, sigma_grad, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], -1)
    gt = np.where(np.abs(gr[..., 2]) < 1e-9, 1e-9, gr[..., 2])
    dr = np.clip(-gr[..., :2] / gt[..., None], -5, 5)                         # RGT-contour dip, samples per trace
    misfit = np.linalg.norm(ds - dr, axis=-1) * dt_ms                        # ms per trace (step cancels out)
    del gr, ds, dr, n
    conf = gaussian_filter(np.exp(-(misfit / tol_ms_per_trace) ** 2) * lin, 1.5)
    full = cube.shape
    up = lambda v, lo, hi: np.clip(zoom(v, [f / c for f, c in zip(full, v.shape)], order=1), lo, hi).astype(np.float32)
    return dict(misfit=up(misfit, 0, None), linearity=up(lin, 0, 1), confidence=up(conf, 0, 1))


# ---------------------------------------------------------------------------
# Repair: proportional layering between guidance horizons where confidence is low
# ---------------------------------------------------------------------------
def proportional_repair(rgt, horizons, samples, confidence, strength=1.0, edge_frac=0.1):
    """
    Between each pair of consecutive guidance horizons (A above B), build the proportional RGT
        prop(t) = L_A + (t - t_A) / (t_B - t_A) * (L_B - L_A)
    (L = the horizon's median RGT value), and blend it in with weight strength * (1 - confidence),
    ramped to zero near A and B (edge_frac of the interval) so the horizons themselves are untouched.
    Outside the guidance horizons the RGT is unchanged. Returns (repaired rgt, blend-weight volume).
    """
    NI, NX, NT = rgt.shape
    dt = samples[1] - samples[0]
    hs = sorted(horizons, key=lambda h: np.nanmean(h["t"]))
    out = rgt.astype(np.float32).copy()
    weight = np.zeros_like(out)
    tt = samples[None, None, :]
    for A, B in zip(hs[:-1], hs[1:]):
        def level(h):
            i, j = np.nonzero(np.isfinite(h["t"]))
            ti = np.clip(np.round((h["t"][i, j] - samples[0]) / dt).astype(int), 0, NT - 1)
            return float(np.median(rgt[i, j, ti]))
        LA, LB = level(A), level(B)
        tA, tB = A["t"][..., None], B["t"][..., None]
        ok = np.isfinite(tA) & np.isfinite(tB) & (tB > tA + 2 * dt)
        s = np.where(ok, (tt - tA) / np.where(ok, tB - tA, 1.0), -1.0)
        inside = (s > 0) & (s < 1)
        prop = LA + s * (LB - LA)
        ramp = np.clip(np.minimum(s, 1 - s) / edge_frac, 0, 1)
        w = np.where(inside, np.clip(strength * (1 - confidence), 0, 1) * ramp, 0.0).astype(np.float32)
        out = np.where(inside, (1 - w) * out + w * prop, out)
        weight = np.maximum(weight, w)
    out = np.maximum.accumulate(out, axis=2)
    return out.astype(np.float32), weight
