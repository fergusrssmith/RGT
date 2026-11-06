"""
Helpers for the RGT-Est own-data / low-frequency-model notebooks.

- read_segy_cube:         read a regular 3D SEG-Y into an (inline, crossline, time) cube,
                          auto-detecting the inline/crossline header bytes.
- load_velocity_on_grid:  bring a velocity SEG-Y (different extent, sampling, header
                          bytes, null-padded traces) onto the seismic grid.
"""

import numpy as np
import segyio

# (inline byte, crossline byte) pairs tried when detecting header positions automatically
CANDIDATES = [(189, 193), (9, 21), (9, 13), (5, 21), (17, 13), (13, 17), (17, 21), (5, 9)]


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------
def _attr(f, byte, dtype=float):
    return np.asarray(f.attributes(byte)[:], dtype=dtype)


def _coords(f):
    """CDP X/Y (181/185), falling back to source X/Y (73/77), with the scalar at byte 71."""
    x, y = _attr(f, 181), _attr(f, 185)
    if not x.any():
        x, y = _attr(f, 73), _attr(f, 77)
    s = _attr(f, 71)
    s = np.where(s == 0, 1, np.where(s < 0, -1 / s, s))
    return x * s, y * s


def _sample_times(f):
    """TWT of each sample (ms). Uses the median delay over all traces, so a few odd
    (e.g. null-padding) trace headers don't shift the time axis."""
    dt = float(segyio.tools.dt(f)) / 1000.0
    t0 = float(np.median(_attr(f, segyio.TraceField.DelayRecordingTime)))
    return t0 + dt * np.arange(len(f.samples)), dt


def detect_ilxl_bytes(f, il_byte=None, xl_byte=None, verbose=True):
    """Return (il_byte, xl_byte, il, xl). Picks the candidate pair giving a clean grid."""
    if il_byte and xl_byte:
        return il_byte, xl_byte, _attr(f, il_byte, np.int64), _attr(f, xl_byte, np.int64)
    n_tr, best = f.tracecount, None
    for b1, b2 in CANDIDATES:
        try:
            a, b = _attr(f, b1, np.int64), _attr(f, b2, np.int64)
        except Exception:
            continue
        nu_a, nu_b = len(np.unique(a)), len(np.unique(b))
        if nu_a < 2 or nu_b < 2:
            continue
        n_pairs = len(np.unique(a * 10**7 + b))
        score = (n_pairs == n_tr) * 2 + (nu_a * nu_b == n_tr)
        if best is None or score > best[0]:
            best = (score, b1, b2, a, b)
    if best is None or best[0] < 2:
        return None, None, None, None
    return best[1], best[2], best[3], best[4]


# ---------------------------------------------------------------------------
# Seismic
# ---------------------------------------------------------------------------
def read_segy_cube(path, il_byte=None, xl_byte=None):
    """Read a 3D SEG-Y into a dict holding the (inline, crossline, time) cube and its geometry."""
    with segyio.open(path, ignore_geometry=True) as f:
        n_tr = f.tracecount
        samples, dt = _sample_times(f)
        il_byte, xl_byte, il_hdr, xl_hdr = detect_ilxl_bytes(f, il_byte, xl_byte)
        if il_byte is None:
            raise ValueError("Could not find inline/crossline headers automatically. "
                             "Set IL_BYTE and XL_BYTE in Settings.")
        cx, cy = _coords(f)
        raw = f.trace.raw[:]

    ilines, xlines = np.unique(il_hdr), np.unique(xl_hdr)
    il_idx, xl_idx = np.searchsorted(ilines, il_hdr), np.searchsorted(xlines, xl_hdr)
    NI, NX, NT = len(ilines), len(xlines), len(samples)
    cube = np.zeros((NI, NX, NT), np.float32)
    cube[il_idx, xl_idx] = raw
    trace_index = -np.ones((NI, NX), np.int64)
    trace_index[il_idx, xl_idx] = np.arange(n_tr)
    X = np.full((NI, NX), np.nan); X[il_idx, xl_idx] = cx
    Y = np.full((NI, NX), np.nan); Y[il_idx, xl_idx] = cy
    return dict(cube=cube, ilines=ilines, xlines=xlines, il_idx=il_idx, xl_idx=xl_idx,
                trace_index=trace_index, X=X, Y=Y, samples=samples, dt=dt, n_tr=n_tr,
                il_byte=il_byte, xl_byte=xl_byte)


def _affine(src_a, src_b, dst):
    """Least-squares fit dst ≈ c0 + c1*src_a + c2*src_b; returns the coefficients."""
    A = np.c_[np.ones_like(src_a), src_a, src_b]
    return np.linalg.lstsq(A, dst, rcond=None)[0]


def grid_transforms(grid):
    """Affine maps between (inline, crossline) numbers and (X, Y) for the seismic grid."""
    II, XX = np.meshgrid(grid["ilines"], grid["xlines"], indexing="ij")
    ok = np.isfinite(grid["X"])
    il, xl, X, Y = II[ok].astype(float), XX[ok].astype(float), grid["X"][ok], grid["Y"][ok]
    to_xy = (_affine(il, xl, X), _affine(il, xl, Y))
    to_ilxl = (_affine(X, Y, il), _affine(X, Y, xl))
    return to_xy, to_ilxl


def _apply(c, a, b):
    return c[0] + c[1] * a + c[2] * b


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------
def load_velocity_on_grid(path, grid, il_byte=None, xl_byte=None, match="auto",
                          vmin=300.0, vmax=10000.0, margin_m=500.0, verbose=True):
    """
    Resample a velocity SEG-Y onto the seismic grid.

    Handles: a different (larger or smaller) lateral extent, a coarser trace spacing,
    null-padded traces (zeros, NaN, -999.25 ... anything outside vmin–vmax), partly null
    traces, different header bytes, and a different time sampling / window.

    match: "ilxl" = place traces by their inline/crossline headers, "xy" = by their X/Y
           coordinates, "auto" = headers if they overlap the seismic, otherwise X/Y.

    Returns (vel, info): vel is (inline, crossline, time) float32 on the seismic grid,
    info is a dict of QC numbers and maps.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    say = print if verbose else (lambda *a, **k: None)
    with segyio.open(path, ignore_geometry=True) as f:
        n_tr = f.tracecount
        t_v, dt_v = _sample_times(f)
        b_il, b_xl, v_il, v_xl = detect_ilxl_bytes(f, il_byte, xl_byte)
        vx, vy = _coords(f)
        data = f.trace.raw[:].astype(np.float32)          # (n_traces, n_samples)

    # ---- 1. nulls: per sample, then per trace
    bad = ~np.isfinite(data) | (data < vmin) | (data > vmax)
    good_frac = 1 - bad.mean(axis=1)
    keep = good_frac >= 0.5                                # mostly-null traces are padding
    n_null, n_partial = int((~keep).sum()), int((keep & (good_frac < 1)).sum())
    say(f"velocity file: {n_tr} traces, {len(t_v)} samples, TWT {t_v[0]:.0f} – {t_v[-1]:.0f} ms every {dt_v:g} ms")
    say(f"  null/padding traces dropped: {n_null}   partly-null traces repaired: {n_partial}")
    if not keep.any():
        raise ValueError(f"No valid velocity traces (all values outside {vmin}–{vmax}). Check units or the null value.")
    idx = np.arange(data.shape[1])
    for k in np.nonzero(keep & (good_frac < 1))[0]:       # fill null samples inside a trace
        g = ~bad[k]
        data[k] = np.interp(idx, idx[g], data[k][g])

    # ---- 2. where each velocity trace sits, in seismic X/Y
    to_xy, to_ilxl = grid_transforms(grid)
    use = match
    if match == "auto":
        overlap = (b_il is not None
                   and np.isin(v_il[keep], grid["ilines"]).any() and np.isin(v_xl[keep], grid["xlines"]).any())
        use = "ilxl" if overlap else "xy"
    if use == "ilxl":
        if b_il is None:
            raise ValueError("No usable inline/crossline headers in the velocity file: use match='xy'.")
        px, py = _apply(to_xy[0], v_il, v_xl), _apply(to_xy[1], v_il, v_xl)
        say(f"  placed by inline/crossline headers (bytes {b_il}/{b_xl})")
        if np.isfinite(vx).all() and np.abs(vx).max() > 0:   # cross-check against the file's own X/Y
            miss = np.hypot(px - vx, py - vy)[keep]
            say(f"  X/Y cross-check: median mismatch {np.median(miss):.1f} m"
                + ("   ← WARNING: headers and coordinates disagree" if np.median(miss) > 50 else ""))
    else:
        px, py = vx, vy
        say("  placed by X/Y coordinates")

    # ---- 3. keep valid traces over the seismic area (plus a margin)
    Xg, Yg = grid["X"], grid["Y"]
    pil, pxl = _apply(to_ilxl[0], px, py), _apply(to_ilxl[1], px, py)
    # Snap traces that sit (almost) exactly on an inline/crossline: rounding in stored coordinates
    # otherwise makes thin, distorted interpolation triangles along the edge of the data.
    for p in (pil, pxl):
        r = np.round(p); close = np.abs(p - r) < 0.05; p[close] = r[close]
    px, py = _apply(to_xy[0], pil, pxl), _apply(to_xy[1], pil, pxl)
    ok_x, ok_y = np.isfinite(Xg), np.isfinite(Yg)
    bin_il = np.hypot(to_xy[0][1], to_xy[1][1])            # metres per inline number
    bin_xl = np.hypot(to_xy[0][2], to_xy[1][2])            # metres per crossline number
    spacing = min(bin_il * np.median(np.diff(grid["ilines"])), bin_xl * np.median(np.diff(grid["xlines"])))
    m_il, m_xl = margin_m / bin_il, margin_m / bin_xl      # margin, in inline / crossline numbers
    near = keep & (pil >= grid["ilines"][0] - m_il) & (pil <= grid["ilines"][-1] + m_il) \
                & (pxl >= grid["xlines"][0] - m_xl) & (pxl <= grid["xlines"][-1] + m_xl)
    if not near.any():
        raise ValueError("No valid velocity traces overlap the seismic. Check the header bytes or try match='xy'.")
    say(f"  valid traces over the seismic area: {int(near.sum())}")

    # ---- 4. resample in time to the seismic samples (hold edge values outside the velocity window)
    t_s = grid["samples"]
    if t_s[0] < t_v[0] or t_s[-1] > t_v[-1]:
        say(f"  NOTE: seismic TWT {t_s[0]:.0f}–{t_s[-1]:.0f} ms extends beyond the velocity window: edge values held")
    tq = np.clip(t_s, t_v[0], t_v[-1])
    j = np.clip(np.searchsorted(t_v, tq) - 1, 0, len(t_v) - 2)
    w = (tq - t_v[j]) / (t_v[j + 1] - t_v[j])
    dv = data[near]
    vt = dv[:, j] * (1 - w) + dv[:, j + 1] * w             # (n_near, NT)

    # ---- 5. laterally onto the seismic nodes (linear inside the data, nearest outside)
    NI, NX, NT = len(grid["ilines"]), len(grid["xlines"]), len(t_s)
    II, XX = np.meshgrid(grid["ilines"], grid["xlines"], indexing="ij")
    gx = np.where(ok_x, Xg, _apply(to_xy[0], II, XX)); gy = np.where(ok_y, Yg, _apply(to_xy[1], II, XX))
    pts, targets = np.c_[px[near], py[near]], np.c_[gx.ravel(), gy.ravel()]
    lin = LinearNDInterpolator(pts, vt)
    vel = lin(targets)                                       # (NI*NX, NT), NaN outside the hull
    outside = np.isnan(vel[:, 0])
    if outside.any():
        vel[outside] = NearestNDInterpolator(pts, vt)(targets[outside])
    vel = vel.reshape(NI, NX, NT).astype(np.float32)

    # exact hits: nodes sitting on a valid velocity trace (within 1% of a bin)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(targets)
    exact = (d < 0.01 * spacing).reshape(NI, NX)
    source = np.where(exact, 0, np.where(outside.reshape(NI, NX), 2, 1))   # 0 exact, 1 interpolated, 2 extrapolated
    n = NI * NX
    say(f"  seismic nodes: {100*(source==0).sum()/n:.1f}% exact, {100*(source==1).sum()/n:.1f}% interpolated, "
        f"{100*(source==2).sum()/n:.1f}% extrapolated (nearest)")
    say(f"  velocity on grid: {vel.min():.0f} – {vel.max():.0f} m/s (median {np.median(vel):.0f})")
    if not 1300 < np.median(vel) < 7000:
        say("  WARNING: median velocity looks unusual for m/s. Check units (ft/s? km/s?) or that this is velocity.")
    info = dict(source=source, n_null=n_null, n_partial=n_partial, match=use,
                vel_points_il=pil[near], vel_points_xl=pxl[near], null_points_il=pil[~keep], null_points_xl=pxl[~keep])
    return vel, info
