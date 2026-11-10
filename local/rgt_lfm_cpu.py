# -*- coding: utf-8 -*-
"""
RGT-Est low-frequency-model workflow, for running locally (e.g. in PyCharm), on CPU or GPU.

Steps
  1. Read the seismic SEG-Y (inline/crossline header bytes detected automatically)
  2. Load guidance horizons (RokDoc .txt: Inline, Crossline, TWT)              [optional]
  3. Load an interval-velocity SEG-Y and put it on the seismic grid             [optional]
  4. Run RGT-Est (pretrained model), unguided and/or horizon-guided; cached to disk
  5. Write the RGT as SEG-Y (trace headers copied from the input) and export horizons
  6. Build a low-frequency AI model from LAS wells in TWT:
       background (velocity -> AI, fitted at the wells) + the wells' detail spread along RGT layers,
       with a leave-one-well-out test (velocity only / wells along flat time / wells along RGT)

All results, including PNG plots, go to <DATA_FOLDER>/outputs/<date_time>/.

Model and method: Dou, Wu, Gao & Bi (2026), "Learning Stratigraphically Consistent Relative Geologic
Time from 3D Seismic Data via Sinusoidal Mapping". Model weights: CC BY 4.0 (Zenodo 10.5281/zenodo.20118902).

Install (Python 3.10–3.12 recommended):
    pip install numpy scipy pandas matplotlib segyio lasio
    pip install torch --index-url https://download.pytorch.org/whl/cpu      (CPU-only PyTorch)
"""

import os
import time
import shutil
import contextlib

import numpy as np

# =====================================================================================
# SETTINGS
# =====================================================================================
DATA_FOLDER   = r"C:\Users\fsinclair\Downloads\RGT"     # all inputs live here
OUTPUT_FOLDER = None                                     # None = <DATA_FOLDER>\outputs\<date_time>

MODEL_FILE    = "RGT-Est_CIG-Benchmark.pt"
SEGY_FILE     = "Full_Stack_8_41_CROP_1793889427461.sgy"
IL_BYTE, XL_BYTE = None, None                            # None = detect automatically

INTERVAL_MS   = (1300, 2300)                             # interval of interest (plots)
WELL_IL, WELL_XL = 1123, 2098                            # location for sections / horizon export

# Horizon guidance: RokDoc horizon .txt exports in TWT. [] = unguided only.
HORIZON_FILES = ["Jubilee.txt", "Ranger.txt"]
USE_GUIDED    = True                                     # use the guided RGT for horizons and the LFM
COMPARE_UNGUIDED = True                                  # also run unguided, to compare (one extra model run)
HORIZON_LEAVE_ONE_OUT = False                            # fair horizon test: one extra model run PER horizon

HORIZON_TIMES_AT_WELL = [1300, 1550, 1800, 2050, 2300]   # horizons traced through these TWTs at WELL_IL/XL

# Velocity model: interval velocity SEG-Y in TWT ("" = none). Extent/sampling may differ; nulls are dropped.
VELOCITY_FILE = "Vint_Vint_1793927905561.sgy"
VEL_IL_BYTE, VEL_XL_BYTE = None, None
VEL_MATCH     = "auto"                                   # "auto", "ilxl" or "xy"

# Wells: LAS files in TWT. "auto" = every .las in DATA_FOLDER, or a list of file names.
WELL_FILES    = "auto"
WELL_AI_CURVE = "AI_WTR"                                 # impedance curve (AI_WTR is usually brine-substituted)
WELL_LOCATIONS = {}                                      # optional {"NORTON_1": (inline, crossline)}
IDW_POWER     = 2
LFM_CUTOFF_HZ = 8
LFM_TAPER_MS  = 100
LFM_TRACE_STEP = 8                                       # also write a decimated LFM

# Compute
DEVICE        = "auto"                                   # "auto", "cpu" or "cuda"
CPU_THREADS   = None                                     # None = PyTorch default (all cores)
CPU_PRECISION = "float32"                                # "float32" or "bfloat16" (about half the memory; newer CPUs)
CACHE_RGT     = True                                     # reuse RGT results from an earlier run of the same inputs

SHOW_PLOTS    = False                                    # True = also open plot windows (each blocks until closed)

GRID = (400, 512, 512)                                   # the model's fixed working grid (time, inline, crossline)


# =====================================================================================
# SEG-Y helpers
# =====================================================================================
CANDIDATES = [(189, 193), (9, 21), (9, 13), (5, 21), (17, 13), (13, 17), (17, 21), (5, 9)]


def _attr(f, byte, dtype=float):
    return np.asarray(f.attributes(byte)[:], dtype=dtype)


def _coords(f):
    x, y = _attr(f, 181), _attr(f, 185)
    if not x.any():
        x, y = _attr(f, 73), _attr(f, 77)
    s = _attr(f, 71)
    s = np.where(s == 0, 1, np.where(s < 0, -1 / s, s))
    return x * s, y * s


def _sample_times(f):
    import segyio
    dt = float(segyio.tools.dt(f)) / 1000.0
    t0 = float(np.median(_attr(f, segyio.TraceField.DelayRecordingTime)))
    return t0 + dt * np.arange(len(f.samples)), dt


def detect_ilxl_bytes(f, il_byte=None, xl_byte=None):
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


def read_segy_cube(path, il_byte=None, xl_byte=None):
    import segyio
    with segyio.open(path, ignore_geometry=True) as f:
        n_tr = f.tracecount
        samples, dt = _sample_times(f)
        il_byte, xl_byte, il_hdr, xl_hdr = detect_ilxl_bytes(f, il_byte, xl_byte)
        if il_byte is None:
            raise ValueError("Could not find inline/crossline headers automatically. Set IL_BYTE and XL_BYTE.")
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
    return dict(cube=cube, ilines=ilines, xlines=xlines, il_idx=il_idx, xl_idx=xl_idx, trace_index=trace_index,
                X=X, Y=Y, samples=samples, dt=dt, n_tr=n_tr, il_byte=il_byte, xl_byte=xl_byte)


def _affine(a, b, dst):
    A = np.c_[np.ones_like(a), a, b]
    return np.linalg.lstsq(A, dst, rcond=None)[0]


def _apply(c, a, b):
    return c[0] + c[1] * a + c[2] * b


def grid_transforms(grid):
    II, XX = np.meshgrid(grid["ilines"], grid["xlines"], indexing="ij")
    ok = np.isfinite(grid["X"])
    il, xl, X, Y = II[ok].astype(float), XX[ok].astype(float), grid["X"][ok], grid["Y"][ok]
    return (_affine(il, xl, X), _affine(il, xl, Y)), (_affine(X, Y, il), _affine(X, Y, xl))


def load_velocity_on_grid(path, grid, il_byte=None, xl_byte=None, match="auto", vmin=300.0, vmax=10000.0, margin_m=500.0):
    import segyio
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import cKDTree
    with segyio.open(path, ignore_geometry=True) as f:
        n_tr = f.tracecount
        t_v, dt_v = _sample_times(f)
        b_il, b_xl, v_il, v_xl = detect_ilxl_bytes(f, il_byte, xl_byte)
        vx, vy = _coords(f)
        data = f.trace.raw[:].astype(np.float32)

    bad = ~np.isfinite(data) | (data < vmin) | (data > vmax)
    good_frac = 1 - bad.mean(axis=1)
    keep = good_frac >= 0.5
    n_null, n_partial = int((~keep).sum()), int((keep & (good_frac < 1)).sum())
    print(f"velocity file: {n_tr} traces, {len(t_v)} samples, TWT {t_v[0]:.0f} – {t_v[-1]:.0f} ms every {dt_v:g} ms")
    print(f"  null/padding traces dropped: {n_null}   partly-null traces repaired: {n_partial}")
    if not keep.any():
        raise ValueError(f"No valid velocity traces (all values outside {vmin}–{vmax}).")
    idx = np.arange(data.shape[1])
    for k in np.nonzero(keep & (good_frac < 1))[0]:
        g = ~bad[k]
        data[k] = np.interp(idx, idx[g], data[k][g])

    to_xy, to_ilxl = grid_transforms(grid)
    use = match
    if match == "auto":
        overlap = (b_il is not None and np.isin(v_il[keep], grid["ilines"]).any()
                   and np.isin(v_xl[keep], grid["xlines"]).any())
        use = "ilxl" if overlap else "xy"
    if use == "ilxl":
        if b_il is None:
            raise ValueError("No usable inline/crossline headers in the velocity file: set VEL_MATCH = 'xy'.")
        px, py = _apply(to_xy[0], v_il, v_xl), _apply(to_xy[1], v_il, v_xl)
        print(f"  placed by inline/crossline headers (bytes {b_il}/{b_xl})")
        if np.isfinite(vx).all() and np.abs(vx).max() > 0:
            miss = np.hypot(px - vx, py - vy)[keep]
            print(f"  X/Y cross-check: median mismatch {np.median(miss):.1f} m"
                  + ("   <- WARNING: headers and coordinates disagree" if np.median(miss) > 50 else ""))
    else:
        px, py = vx, vy
        print("  placed by X/Y coordinates")

    pil, pxl = _apply(to_ilxl[0], px, py), _apply(to_ilxl[1], px, py)
    for p in (pil, pxl):                                    # snap traces sitting on a line (coordinate rounding)
        r = np.round(p); close = np.abs(p - r) < 0.05; p[close] = r[close]
    px, py = _apply(to_xy[0], pil, pxl), _apply(to_xy[1], pil, pxl)

    Xg, Yg = grid["X"], grid["Y"]
    bin_il, bin_xl = np.hypot(to_xy[0][1], to_xy[1][1]), np.hypot(to_xy[0][2], to_xy[1][2])
    spacing = min(bin_il * np.median(np.diff(grid["ilines"])), bin_xl * np.median(np.diff(grid["xlines"])))
    m_il, m_xl = margin_m / bin_il, margin_m / bin_xl
    near = keep & (pil >= grid["ilines"][0] - m_il) & (pil <= grid["ilines"][-1] + m_il) \
                & (pxl >= grid["xlines"][0] - m_xl) & (pxl <= grid["xlines"][-1] + m_xl)
    if not near.any():
        raise ValueError("No valid velocity traces overlap the seismic. Check header bytes or set VEL_MATCH = 'xy'.")
    print(f"  valid traces over the seismic area: {int(near.sum())}")

    t_s = grid["samples"]
    if t_s[0] < t_v[0] or t_s[-1] > t_v[-1]:
        print(f"  NOTE: seismic TWT {t_s[0]:.0f}–{t_s[-1]:.0f} ms extends beyond the velocity window: edge values held")
    tq = np.clip(t_s, t_v[0], t_v[-1])
    j = np.clip(np.searchsorted(t_v, tq) - 1, 0, len(t_v) - 2)
    w = (tq - t_v[j]) / (t_v[j + 1] - t_v[j])
    dv = data[near]
    vt = dv[:, j] * (1 - w) + dv[:, j + 1] * w

    NI, NX, NT = len(grid["ilines"]), len(grid["xlines"]), len(t_s)
    II, XX = np.meshgrid(grid["ilines"], grid["xlines"], indexing="ij")
    gx = np.where(np.isfinite(Xg), Xg, _apply(to_xy[0], II, XX))
    gy = np.where(np.isfinite(Yg), Yg, _apply(to_xy[1], II, XX))
    pts, targets = np.c_[px[near], py[near]], np.c_[gx.ravel(), gy.ravel()]
    vel = LinearNDInterpolator(pts, vt)(targets)
    outside = np.isnan(vel[:, 0])
    if outside.any():
        vel[outside] = NearestNDInterpolator(pts, vt)(targets[outside])
    vel = vel.reshape(NI, NX, NT).astype(np.float32)
    d, _ = cKDTree(pts).query(targets)
    exact = (d < 0.01 * spacing).reshape(NI, NX)
    source = np.where(exact, 0, np.where(outside.reshape(NI, NX), 2, 1))
    n = NI * NX
    print(f"  seismic nodes: {100*(source==0).sum()/n:.1f}% exact, {100*(source==1).sum()/n:.1f}% interpolated, "
          f"{100*(source==2).sum()/n:.1f}% extrapolated (nearest)")
    print(f"  velocity on grid: {vel.min():.0f} – {vel.max():.0f} m/s (median {np.median(vel):.0f})")
    if not 1300 < np.median(vel) < 7000:
        print("  WARNING: median velocity looks unusual for m/s. Check units or that this is velocity.")
    return vel, dict(source=source)


# =====================================================================================
# Horizon tracing (from demo/rgt_helper.py, X. Wu, vectorised port; MIT licence)
# =====================================================================================
def horizons_from_rgt(sig, hu, ux):
    from scipy.ndimage import gaussian_filter1d
    """
    Extract a set of horizons (iso-surfaces) from an RGT volume.

    Parameters
    ----------
    sig : float
        Smoothing width applied to each extracted horizon (sig=0 means no smoothing).
    hu : 1D array_like
        RGT values at which horizons (iso-surfaces) are extracted.
    ux : 3D ndarray, shape (n3, n2, n1)
        The RGT volume. n1 is the vertical axis.

    Returns
    -------
    hzs : 3D ndarray, shape (nh, n3, n2)
        For each requested RGT value hu[ih], a horizon surface giving the
        vertical (i1) position as a function of (i3, i2).
    """
    ux = np.asarray(ux, dtype=np.float32)
    hu = np.asarray(hu, dtype=np.float32)

    n3, n2, n1 = ux.shape
    nh = hu.shape[0]

    # ---------------------------------------------------------------
    # Step 1: build a "monotonic mask" for every trace, vectorized.
    #
    # Java logic per trace (n3, n2):
    #     keep ux[0]; then keep ux[i1] iff ux[i1] > max(kept so far).
    # That is exactly: keep the running cumulative-maximum strict-increase points.
    #
    # A sample at (i3,i2,i1) is kept  <=>  ux[i3,i2,i1] > cummax(ux[i3,i2,:i1]).
    # The first sample (i1=0) is always kept.
    # ---------------------------------------------------------------
    cmax = np.maximum.accumulate(ux, axis=-1)        # shape (n3, n2, n1)
    keep = np.empty_like(ux, dtype=bool)
    keep[..., 0] = True
    keep[..., 1:] = ux[..., 1:] > cmax[..., :-1]      # strict > vs running max so far

    # ---------------------------------------------------------------
    # Step 2: mirror the Java `copy(k1-1, ...)` step (drops the LAST kept sample).
    # We do that by clearing the last True in each trace.
    # ---------------------------------------------------------------
    # Index of the last True along axis=-1 for each (i3,i2).
    # Trick: argmax on the reversed boolean array gives the offset from the end.
    rev = keep[..., ::-1]
    last_offset = rev.argmax(axis=-1)                 # 0 means the very last sample is True
    last_idx = (n1 - 1) - last_offset                 # absolute index of last True in each trace
    i3_idx, i2_idx = np.indices((n3, n2))
    keep[i3_idx, i2_idx, last_idx] = False            # drop it, matching `copy(k1-1, ...)`

    # ---------------------------------------------------------------
    # Step 3: per-trace linear interpolation hu -> x1, vectorized.
    #
    # Trick: we sort kept u-values to the front of each trace by replacing
    # not-kept positions with +inf (so they get pushed to the tail by argsort).
    # Then for each query hu[ih], we find the segment via vectorized searchsorted
    # WITHOUT building a (nh, n3, n2, n1) array -- we iterate only over nh,
    # which is small (typically a handful). Inside the nh loop we still operate
    # on the entire (n3, n2) plane in one shot, so it's still vectorized.
    # ---------------------------------------------------------------
    INF = np.float32(np.inf)
    u_kept = np.where(keep, ux, INF)                              # (n3, n2, n1)
    # Sort each trace ascendingly along axis=-1; sentinels go to the tail.
    order = np.argsort(u_kept, axis=-1, kind="stable")            # (n3, n2, n1)
    u_sorted = np.take_along_axis(u_kept, order, axis=-1)         # (n3, n2, n1)
    # Apply the same permutation to the original sample indices (0..n1-1).
    idx_orig = np.broadcast_to(np.arange(n1, dtype=np.float32),
                               (n3, n2, n1))
    x_sorted = np.take_along_axis(idx_orig, order, axis=-1)       # (n3, n2, n1)

    n_keep = keep.sum(axis=-1)                                    # (n3, n2)

    hzs = np.empty((nh, n3, n2), dtype=np.float32)
    # Loop only over the (small) nh axis. Each iteration is fully vectorized
    # over the (n3, n2) plane and avoids any (nh, n3, n2, n1) intermediates.
    for ih in range(nh):
        q = hu[ih]
        # Number of kept u-values strictly less than q, per trace.
        # Equivalent to per-row np.searchsorted(u_sorted, q, side='left'),
        # but works on the full 3D array via broadcasting + sum.
        # We mask sentinels by counting only "u < q AND kept".
        k = (u_sorted < q).sum(axis=-1).astype(np.int32)          # (n3, n2)

        # Form segment endpoints [lo, hi] = [k-1, k], clamped into [0, n_keep-1].
        hi = np.clip(k, 1, n_keep - 1)                            # ensure hi >= 1
        lo = hi - 1

        u_lo = np.take_along_axis(u_sorted, lo[..., None], axis=-1)[..., 0]
        u_hi = np.take_along_axis(u_sorted, hi[..., None], axis=-1)[..., 0]
        x_lo = np.take_along_axis(x_sorted, lo[..., None], axis=-1)[..., 0]
        x_hi = np.take_along_axis(x_sorted, hi[..., None], axis=-1)[..., 0]

        du = u_hi - u_lo
        du_safe = np.where(du == 0, np.float32(1.0), du)
        t = (q - u_lo) / du_safe
        result = x_lo + t * (x_hi - x_lo)

        # Degenerate traces (length < 2) fall back to x_lo (= 0 on those traces),
        # matching the loop version.
        degenerate = n_keep < 2
        result = np.where(degenerate, x_lo, result)
        hzs[ih] = result.astype(np.float32)

    # ---------------------------------------------------------------
    # Step 4: optional lateral smoothing (matches ref.apply1 then ref.apply2).
    # ---------------------------------------------------------------
    if sig > 0:
        hzs = gaussian_filter1d(hzs, sigma=sig, axis=1, mode="nearest")
        hzs = gaussian_filter1d(hzs, sigma=sig, axis=2, mode="nearest")

    return hzs



# =====================================================================================
# Small utilities
# =====================================================================================
def normalization(data):
    rng = np.max(data) - np.min(data)
    return (data - np.min(data)) / (rng + 1e-6)


def z_score_clip(data, clp_s=2.0):
    z = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return normalization(np.clip(z, -clp_s, clp_s))


class Plots:
    def __init__(self, out_dir):
        self.out_dir, self.n = out_dir, 0

    def save(self, fig, name):
        import matplotlib.pyplot as plt
        self.n += 1
        path = os.path.join(self.out_dir, f"{self.n:02d}_{name}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"  [plot] {os.path.basename(path)}")
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)


def find_input(name):
    if not name:
        return None
    p = os.path.join(DATA_FOLDER, name)
    return p if os.path.exists(p) else None


def list_folder(*exts):
    return sorted(f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(exts))


# =====================================================================================
# Main workflow
# =====================================================================================
def main():
    import pandas as pd
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.set_option("display.width", 200)

    run_start = time.time()
    out_dir = OUTPUT_FOLDER or os.path.join(DATA_FOLDER, "outputs", time.strftime("%Y-%m-%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    plots = Plots(out_dir)
    print(f"Outputs will be written to {out_dir}\n")

    # ---------------------------------------------------------------- 1. inputs
    if not os.path.isdir(DATA_FOLDER):
        raise FileNotFoundError(f"DATA_FOLDER not found: {DATA_FOLDER}")
    MODEL_PATH, SEGY_PATH = find_input(MODEL_FILE), find_input(SEGY_FILE)
    VEL_PATH = find_input(VELOCITY_FILE) if VELOCITY_FILE else None
    HORIZON_PATHS = [find_input(h) for h in HORIZON_FILES]
    if WELL_FILES == "auto":
        WELL_PATHS = [os.path.join(DATA_FOLDER, f) for f in list_folder(".las")]
    else:
        WELL_PATHS = [find_input(f) for f in WELL_FILES]
    missing = [n for n, p in [(MODEL_FILE, MODEL_PATH), (SEGY_FILE, SEGY_PATH)]
               + ([(VELOCITY_FILE, VEL_PATH)] if VELOCITY_FILE else [])
               + list(zip(HORIZON_FILES, HORIZON_PATHS))
               + ([] if WELL_FILES == "auto" else list(zip(WELL_FILES, WELL_PATHS))) if not p]
    for label, p in [("model", MODEL_PATH), ("SEG-Y", SEGY_PATH), ("velocity", VEL_PATH)] + \
                    [("horizon", p) for p in HORIZON_PATHS] + [("well", p) for p in WELL_PATHS]:
        if p:
            print(f"found    {label:9s} {os.path.basename(p)}  ({os.path.getsize(p)/1e6:.1f} MB)")
    if missing:
        print("\nFiles in DATA_FOLDER:")
        for f in sorted(os.listdir(DATA_FOLDER)):
            print("   ", f)
        raise FileNotFoundError(f"Missing in {DATA_FOLDER}: {missing}. Fix the names in SETTINGS.")
    for label, p in (("SEG-Y", SEGY_PATH), ("velocity", VEL_PATH)):
        if p and any(tag in os.path.basename(p) for tag in ("_RGT", "_LFM", "_Vint_on_grid")):
            raise ValueError(f"The {label} file {os.path.basename(p)} looks like an output of an earlier run.")

    # ---------------------------------------------------------------- 2. seismic
    print("\n=== Seismic")
    G = read_segy_cube(SEGY_PATH, IL_BYTE, XL_BYTE)
    cube, ilines, xlines, samples, dt = G["cube"], G["ilines"], G["xlines"], G["samples"], G["dt"]
    il_idx, xl_idx, trace_index, X, Y = G["il_idx"], G["xl_idx"], G["trace_index"], G["X"], G["Y"]
    n_tr, il_byte, xl_byte = G["n_tr"], G["il_byte"], G["xl_byte"]
    NI, NX, NT = cube.shape
    print(f"Header bytes used:  inline {il_byte}, crossline {xl_byte}")
    print(f"Inlines:    {ilines[0]} – {ilines[-1]}  ({NI})")
    print(f"Crosslines: {xlines[0]} – {xlines[-1]}  ({NX})")
    print(f"TWT:        {samples[0]:.0f} – {samples[-1]:.0f} ms, every {dt:g} ms  ({NT} samples)")
    print(f"Traces:     {n_tr}  (missing grid nodes: {(trace_index < 0).sum()})")
    print("Corner coordinates (inline, crossline -> X, Y):")
    for i, j in ((0, 0), (0, NX - 1), (NI - 1, 0), (NI - 1, NX - 1)):
        print(f"   {ilines[i]:>6}, {xlines[j]:>6} -> {X[i, j]:,.1f}, {Y[i, j]:,.1f}")
    if WELL_IL in ilines and WELL_XL in xlines:
        ci, cx_ = int(np.searchsorted(ilines, WELL_IL)), int(np.searchsorted(xlines, WELL_XL))
    else:
        ci, cx_ = NI // 2, NX // 2
        print(f"NOTE: WELL_IL/WELL_XL ({WELL_IL}, {WELL_XL}) is outside the volume: using the centre for sections.")
    ext_il = (xlines[0], xlines[-1], samples[-1], samples[0])          # extent of an inline section
    ext_xl = (ilines[0], ilines[-1], samples[-1], samples[0])          # extent of a crossline section

    def tidx(ms):
        return int(np.clip(round((ms - samples[0]) / dt), 0, NT - 1))

    v = np.percentile(np.abs(cube), 99)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(cube[ci].T, cmap="gray", aspect="auto", vmin=-v, vmax=v, extent=ext_il)
    axes[0].set_title(f"inline {ilines[ci]}"); axes[0].set_xlabel("crossline")
    axes[1].imshow(cube[:, cx_].T, cmap="gray", aspect="auto", vmin=-v, vmax=v, extent=ext_xl)
    axes[1].set_title(f"crossline {xlines[cx_]}"); axes[1].set_xlabel("inline")
    for ax in axes[:2]:
        ax.set_ylabel("TWT (ms)")
        for t_ in INTERVAL_MS: ax.axhline(t_, color="orange", lw=1, ls="--")
    tm = sum(INTERVAL_MS) / 2
    axes[2].imshow(cube[:, :, tidx(tm)], cmap="gray", aspect="auto", vmin=-v, vmax=v, origin="lower",
                   extent=(xlines[0], xlines[-1], ilines[0], ilines[-1]))
    axes[2].set_title(f"time slice {tm:.0f} ms"); axes[2].set_xlabel("crossline"); axes[2].set_ylabel("inline")
    fig.suptitle("QC: input seismic"); plots.save(fig, "seismic_qc")

    # ---------------------------------------------------------------- 3. horizons
    import re
    HZ = []
    if HORIZON_PATHS:
        print("\n=== Horizons")
    for path in HORIZON_PATHS:
        name, ztype, rows = os.path.splitext(os.path.basename(path))[0], None, []
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = re.match(r'\s*Horizon\s+"(.+?)"', line)
                if m: name = m.group(1)
                if "Z Type" in line: ztype = line.split(":")[-1].strip()
                parts = line.replace(",", " ").split()
                if len(parts) >= 3:
                    try: rows.append([float(x) for x in parts[:3]])
                    except ValueError: pass
        a = np.array(rows, dtype=float)
        if ztype and ztype.upper() != "TWT":
            print(f"WARNING: {name} has Z type {ztype!r}, but the seismic is in TWT.")
        hil, hxl, hz = np.round(a[:, 0]), np.round(a[:, 1]), a[:, 2]
        ok = np.isfinite(hz) & (hz > -999) & np.isin(hil, ilines) & np.isin(hxl, xlines) \
             & (hz >= samples[0]) & (hz <= samples[-1])
        t = np.full((NI, NX), np.nan)
        t[np.searchsorted(ilines, hil[ok]), np.searchsorted(xlines, hxl[ok])] = hz[ok]
        if not np.isfinite(t).any():
            print(f"{name}: no points fall on the crop: skipped."); continue
        HZ.append(dict(name=name, t=t))
        print(f"{name}: {len(a):,} points in file, {np.isfinite(t).sum():,} on the crop "
              f"({100*np.isfinite(t).mean():.0f}% of traces), TWT {np.nanmin(t):.0f} – {np.nanmax(t):.0f} ms")
    HZ.sort(key=lambda h: np.nanmean(h["t"]))
    for up, down in zip(HZ, HZ[1:]):
        both = np.isfinite(up["t"]) & np.isfinite(down["t"])
        bad = (down["t"][both] <= up["t"][both]).mean() * 100
        print(f"{up['name']} -> {down['name']}: median gap {np.median(down['t'][both] - up['t'][both]):.0f} ms, "
              f"crossing/touching on {bad:.1f}% of traces" + ("   <- WARNING: horizons should not cross" if bad > 0.5 else ""))

    # ---------------------------------------------------------------- 4. velocity
    VEL = None
    if VEL_PATH:
        print("\n=== Velocity")
        from matplotlib.colors import ListedColormap
        VEL, vel_info = load_velocity_on_grid(VEL_PATH, G, VEL_IL_BYTE, VEL_XL_BYTE, match=VEL_MATCH)
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), gridspec_kw={"width_ratios": [1.2, 2, 2]})
        axes[0].imshow(vel_info["source"], origin="lower", aspect="auto", vmin=-0.5, vmax=2.5,
                       cmap=ListedColormap(["#2a9d8f", "#e9c46a", "#e76f51"]),
                       extent=(xlines[0] - .5, xlines[-1] + .5, ilines[0] - .5, ilines[-1] + .5))
        axes[0].set_title("velocity source per trace\ngreen = exact, yellow = interpolated, red = nearest")
        for ax, vsec, ssec, ext, title in ((axes[1], VEL[ci].T, cube[ci].T, ext_il, f"Vint inline {ilines[ci]}"),
                                           (axes[2], VEL[:, cx_].T, cube[:, cx_].T, ext_xl, f"Vint crossline {xlines[cx_]}")):
            im = ax.imshow(vsec, cmap="turbo", aspect="auto", extent=ext)
            vv = np.percentile(np.abs(ssec), 99)
            ax.imshow(ssec, cmap="gray", aspect="auto", extent=ext, vmin=-vv, vmax=vv, alpha=0.3)
            ax.set_title(title); ax.set_ylabel("TWT (ms)"); plt.colorbar(im, ax=ax, label="Vint (m/s)")
        plots.save(fig, "velocity_qc")

    # ---------------------------------------------------------------- 5. RGT
    print("\n=== RGT-Est")
    seis_norm = z_score_clip(cube) * 2 - 1

    def guidance_volume(use):
        from scipy.ndimage import grey_dilation
        ids = np.zeros((NI, NX, NT), np.int16)
        for k, h in enumerate(use, 1):
            i, j = np.nonzero(np.isfinite(h["t"]))
            ids[i, j, np.clip(np.round((h["t"][i, j] - samples[0]) / dt).astype(int), 0, NT - 1)] = k
        ids = grey_dilation(ids, size=2)
        vol = np.zeros((NI, NX, NT), np.float32)
        for k, h in enumerate(use, 1):
            vol[ids == k] = (np.nanmean(h["t"]) - samples[0]) / dt / NT * 2 - 1
        return vol

    cache_dir = os.path.join(DATA_FOLDER, "outputs", "_rgt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(SEGY_PATH)
    seis_key = f"{os.path.splitext(SEGY_FILE)[0]}_{st.st_size}_{int(st.st_mtime)}"
    runner = {"model": None}

    def rgt_for(label, use):
        # cache key: seismic file + which horizons, with a fingerprint of their values (re-exports get a new key)
        key = seis_key + "_" + ("unguided" if not use else
                                "guided_" + "+".join(f"{h['name']}-{int(np.nansum(h['t']) * 10) % 10**8}" for h in use))
        cpath = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_.+-]", "_", key) + ".npy")
        if CACHE_RGT and os.path.exists(cpath):
            print(f"{label}: loaded from cache ({os.path.basename(cpath)})")
            return np.load(cpath)
        if runner["model"] is None:
            runner["model"] = load_model(MODEL_PATH)
        print(f"{label}: running the model ...")
        out = run_model(runner["model"], seis_norm, None if not use else guidance_volume(use), (NT, NI, NX))
        out = z_score_clip(out).transpose(1, 2, 0).astype(np.float32)          # (inline, crossline, time)
        if CACHE_RGT:
            np.save(cpath, out)
        return out

    rgt_unguided = rgt_guided = None
    rgt_loo = {}
    if HZ and USE_GUIDED:
        rgt_guided = rgt_for(f"Guided with {len(HZ)} horizon(s)", HZ)
    if not HZ or not USE_GUIDED or COMPARE_UNGUIDED:
        rgt_unguided = rgt_for("Unguided", [])
    if HZ and HORIZON_LEAVE_ONE_OUT and len(HZ) >= 2:
        for h in HZ:
            rgt_loo[h["name"]] = rgt_for(f"Guided without {h['name']}", [o for o in HZ if o is not h])
    rgt = rgt_guided if rgt_guided is not None else rgt_unguided
    print("Using the", "GUIDED" if rgt is rgt_guided else "UNGUIDED", "RGT for the outputs.")

    # ---------------------------------------------------------------- 5b. horizon scores
    if HZ:
        def horizon_errors(vol, hz_list):
            levels = []
            for h in hz_list:
                i, j = np.nonzero(np.isfinite(h["t"]))
                ti = np.clip(np.round((h["t"][i, j] - samples[0]) / dt).astype(int), 0, NT - 1)
                levels.append(np.median(vol[i, j, ti]))
            pred = samples[0] + horizons_from_rgt(sig=0.0, hu=np.array(levels), ux=vol) * dt
            out = []
            for h, p in zip(hz_list, pred):
                m = np.isfinite(h["t"]); err = np.abs(p[m] - h["t"][m])
                out.append((err.mean(), 100 * (err <= 2 * dt).mean()))
            return out

        res = {}
        if rgt_unguided is not None: res["unguided"] = horizon_errors(rgt_unguided, HZ)
        if rgt_loo: res["held out"] = [horizon_errors(rgt_loo[h["name"]], [h])[0] for h in HZ]
        if rgt_guided is not None: res["all guided"] = horizon_errors(rgt_guided, HZ)
        table = pd.DataFrame({"horizon": [h["name"] for h in HZ]})
        for c in res: table[f"{c}: error (ms)"] = [e for e, _ in res[c]]
        for c in res: table[f"{c}: hit %"] = [r for _, r in res[c]]
        print("\nHorizon fit (vertical miss between each interpreted horizon and the best-matching RGT surface):")
        print(table.round(1).to_string(index=False))
        if "held out" not in res and rgt_guided is not None:
            print("  ('all guided' only shows how well the model honours its inputs. For a fair test set "
                  "HORIZON_LEAVE_ONE_OUT = True.)")

    # ---------------------------------------------------------------- 5c. sections
    levels = np.linspace(0.02, 0.98, 45)
    runs = [(n_, r_) for n_, r_ in (("unguided", rgt_unguided), ("guided", rgt_guided)) if r_ is not None]
    for sec_name, sl, ext, xs in (("inline", lambda a: a[ci].T, ext_il, xlines), ("crossline", lambda a: a[:, cx_].T, ext_xl, ilines)):
        fig, axes = plt.subplots(1, len(runs) + 1, figsize=(7 * (len(runs) + 1), 5.5), squeeze=False)
        s_ = sl(cube); v = np.percentile(np.abs(s_), 99)
        axes[0][0].imshow(s_, cmap="gray", aspect="auto", vmin=-v, vmax=v, extent=ext); axes[0][0].set_title("seismic")
        for ax, (n_, r_) in zip(axes[0][1:], runs):
            ax.imshow(s_, cmap="gray", aspect="auto", vmin=-v, vmax=v, extent=ext)
            ax.contour(sl(r_), levels=levels, cmap="jet", linewidths=0.9, extent=(ext[0], ext[1], ext[3], ext[2]))
            for h in HZ:
                ax.plot(xs, h["t"][ci] if sec_name == "inline" else h["t"][:, cx_], color="magenta", lw=2, ls="--")
            ax.set_title(f"{n_} RGT contours (magenta = interpreted horizons)")
        for ax in axes[0]:
            ax.set_ylabel("TWT (ms)"); ax.set_ylim(samples[-1], samples[0])
            for t_ in INTERVAL_MS: ax.axhline(t_, color="orange", lw=1, ls="--")
        title = f"{sec_name} {ilines[ci] if sec_name == 'inline' else xlines[cx_]}"
        fig.suptitle(title); plots.save(fig, f"rgt_{sec_name}")

    # ---------------------------------------------------------------- 6. write RGT / velocity
    print("\n=== Writing SEG-Y")
    import segyio
    stem = os.path.splitext(os.path.basename(SEGY_PATH))[0]
    OUT = {}

    def write_like_input(volume, name):
        out_path = os.path.join(out_dir, f"{stem}_{name}.sgy")
        shutil.copyfile(SEGY_PATH, out_path)
        with segyio.open(out_path, "r+", ignore_geometry=True) as f:
            for k in range(n_tr):
                f.trace[k] = np.ascontiguousarray(volume[il_idx[k], xl_idx[k]], dtype=np.float32)
        print("wrote", os.path.basename(out_path), f"({os.path.getsize(out_path)/1e6:.0f} MB)")
        return out_path

    if rgt_unguided is not None: OUT["rgt"] = write_like_input(rgt_unguided, "RGT")
    if rgt_guided is not None: OUT["rgt_guided"] = write_like_input(rgt_guided, "RGT_guided")
    if VEL is not None: OUT["vint"] = write_like_input(VEL, "Vint_on_grid")

    # ---------------------------------------------------------------- 7. horizons from the RGT
    print("\n=== Horizons traced through the RGT")
    well_rgt = np.maximum.accumulate(rgt[ci, cx_])
    levels_h = np.interp(HORIZON_TIMES_AT_WELL, samples, well_rgt)
    hz_ms = samples[0] + horizons_from_rgt(sig=1.0, hu=levels_h, ux=rgt) * dt
    II, XX = np.meshgrid(ilines, xlines, indexing="ij")
    rows = []
    for t_well, hz in zip(HORIZON_TIMES_AT_WELL, hz_ms):
        print(f"horizon {t_well} ms: at inline {ilines[ci]}/crossline {xlines[cx_]} {hz[ci, cx_]:.0f} ms, "
              f"range {hz.min():.0f} – {hz.max():.0f} ms")
        rows.append(pd.DataFrame({"Horizon": f"RGT_{t_well}ms", "Inline": II.ravel(), "Crossline": XX.ravel(),
                                  "X": X.ravel(), "Y": Y.ravel(), "TWT_ms": hz.ravel()}))
    hz_table = pd.concat(rows)
    hz_table.to_csv(os.path.join(out_dir, f"{stem}_horizons.txt"), sep="\t", index=False, float_format="%.2f")
    for t_well, g_ in hz_table.groupby("Horizon", sort=False):
        g_.drop(columns="Horizon").to_csv(os.path.join(out_dir, f"{stem}_{t_well}.txt"), sep="\t", index=False, float_format="%.2f")
    print(f"wrote {stem}_horizons.txt and one file per horizon")

    # ---------------------------------------------------------------- 8. LFM from the wells
    if not WELL_PATHS:
        print("\nNo LAS wells found: skipping the LFM.")
    else:
        print("\n=== Low-frequency model")
        build_lfm(locals(), plots, out_dir, stem, write_like_input, OUT)

    print(f"\nDone in {(time.time() - run_start)/60:.1f} min. Outputs in {out_dir}")


# =====================================================================================
# Model
# =====================================================================================
def load_model(path):
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    if device == "cpu" and CPU_THREADS:
        torch.set_num_threads(int(CPU_THREADS))
    print(f"PyTorch {torch.__version__} on {device.upper()}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else f" ({torch.get_num_threads()} threads, {CPU_PRECISION})"))
    try:
        import psutil
        print(f"  free memory: {psutil.virtual_memory().available/1e9:.1f} GB "
              "(the model needs a lot: if Windows runs out, try CPU_PRECISION = 'bfloat16')")
    except ImportError:
        pass
    model = torch.jit.load(path, map_location=device).eval()
    return model, device


def run_model(model_dev, seis_norm, guide, out_shape):
    # seis_norm, guide: (inline, crossline, time). Returns the raw RGT as (time, inline, crossline).
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    model, device = model_dev
    with torch.no_grad():
        s = F.interpolate(torch.from_numpy(seis_norm.transpose(2, 0, 1).copy())[None, None], GRID, mode="trilinear")
        if guide is None:
            g = torch.zeros_like(s)
        else:
            g = F.interpolate(torch.from_numpy(guide.transpose(2, 0, 1).copy())[None, None], GRID, mode="nearest")
        x = nn.ReplicationPad3d(8)(torch.cat([s, g, (g != 0).float()], dim=1)).to(device)
        del s, g
        if device == "cuda":
            ctx = torch.autocast(device_type="cuda")
        elif CPU_PRECISION == "bfloat16":
            ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()
        t0 = time.time()
        with ctx:
            rgt = model(x)[:, :, 8:-8, 8:-8, 8:-8]
        print(f"  inference took {time.time() - t0:.1f} s")
        out = F.interpolate(rgt.float().cpu(), out_shape, mode="trilinear").numpy()[0, 0]
    del x, rgt
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# =====================================================================================
# Low-frequency model
# =====================================================================================
def read_well(path):
    import pandas as pd
    base = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".las"):
        import lasio
        las = lasio.read(path)

        def hdr(*keys):
            for k in keys:
                try:
                    v = las.well[k].value
                    if v not in (None, ""):
                        return v
                except (KeyError, AttributeError):
                    pass
            return None
        name = str(hdr("WELL") or base).strip()
        unit, zstk = (las.curves[0].unit or "").lower(), str(hdr("ZSTK") or "").upper()
        if "ms" not in unit and zstk != "TWT":
            raise ValueError(f"{base}: the LAS index is in {unit!r} (Z-stick {zstk!r}), not TWT. Export it in TWT (ms).")
        if WELL_AI_CURVE not in las.keys():
            raise ValueError(f"{name}: no curve {WELL_AI_CURVE!r}. Curves in the file: {list(las.keys())}")
        t, ai = np.asarray(las.index, float), np.asarray(las[WELL_AI_CURVE], float)
        xy = None
        for kx, ky in (("XWELL", "YWELL"), ("X", "Y"), ("XCOORD", "YCOORD")):
            try:
                xy = (float(hdr(kx)), float(hdr(ky))); break
            except (TypeError, ValueError):
                pass
    else:
        raw = pd.read_excel(path, header=None) if path.lower().endswith((".xlsx", ".xls")) \
              else pd.read_csv(path, header=None, sep=None, engine="python")
        num = raw.apply(pd.to_numeric, errors="coerce").iloc[:, :2].dropna()
        t, ai, name, xy = num.iloc[:, 0].to_numpy(float), num.iloc[:, 1].to_numpy(float), base, None
    ok = np.isfinite(t) & np.isfinite(ai) & (ai > 0)
    o = np.argsort(t[ok])
    return name, t[ok][o], ai[ok][o], xy


def build_lfm(ctx, plots, out_dir, stem, write_like_input, OUT):
    import pandas as pd
    import segyio
    import matplotlib.pyplot as plt
    from scipy.signal import butter, sosfiltfilt
    G, rgt, VEL, WELL_PATHS = ctx["G"], ctx["rgt"], ctx["VEL"], ctx["WELL_PATHS"]
    ilines, xlines, samples, dt, X, Y = G["ilines"], G["xlines"], G["samples"], G["dt"], G["X"], G["Y"]
    NI, NX, NT = G["cube"].shape
    tidx = ctx["tidx"]

    to_xy, to_ilxl = grid_transforms(G)
    II, XX = np.meshgrid(ilines, xlines, indexing="ij")
    Xg = np.where(np.isfinite(X), X, _apply(to_xy[0], II, XX))
    Yg = np.where(np.isfinite(Y), Y, _apply(to_xy[1], II, XX))
    t_norm = (samples - samples[0]) / (samples[-1] - samples[0])

    wells = []
    for path in WELL_PATHS:
        name, t, a_raw, xy = read_well(path)
        if name in WELL_LOCATIONS:
            il_w, xl_w = map(float, WELL_LOCATIONS[name]); src = "Settings"
        elif xy is not None:
            il_w, xl_w = _apply(to_ilxl[0], *xy), _apply(to_ilxl[1], *xy); src = "LAS X/Y"
        else:
            print(f"{name}: no X/Y in the file. Add it to WELL_LOCATIONS. Skipped."); continue
        if not (ilines[0] - 0.5 <= il_w <= ilines[-1] + 0.5 and xlines[0] - 0.5 <= xl_w <= xlines[-1] + 0.5):
            print(f"{name}: inline {il_w:.1f}, crossline {xl_w:.1f} is outside this volume. Skipped."); continue
        if len(t) < 10 or t.max() < samples[0] or t.min() > samples[-1]:
            print(f"{name}: the log (TWT {t.min():.0f}–{t.max():.0f} ms) doesn't overlap the seismic. Skipped."); continue
        dtl = float(np.median(np.diff(t)))
        tt = np.arange(t[0], t[-1] + dtl / 2, dtl)
        aa = np.interp(tt, t, a_raw)
        low = sosfiltfilt(butter(4, LFM_CUTOFF_HZ, fs=1000.0 / dtl, output="sos"), aa) if len(tt) > 30 else aa
        ai_s = np.interp(samples, tt, aa, left=np.nan, right=np.nan)
        ai_low = np.interp(samples, tt, low, left=np.nan, right=np.nan)
        valid = np.isfinite(ai_low)
        if valid.sum() < 10:
            print(f"{name}: fewer than 10 samples inside the seismic window. Skipped."); continue
        i_w, j_w = int(np.argmin(np.abs(ilines - il_w))), int(np.argmin(np.abs(xlines - xl_w)))
        x_w, y_w = xy if xy is not None else (Xg[i_w, j_w], Yg[i_w, j_w])
        wells.append(dict(name=name, il=il_w, xl=xl_w, i=i_w, j=j_w, x=x_w, y=y_w, ai=ai_s, ai_low=ai_low, valid=valid,
                          first=int(np.argmax(valid)), last=int(NT - 1 - np.argmax(valid[::-1]))))
        print(f"{name:12s} inline {il_w:7.1f}  crossline {xl_w:7.1f}  ({src})   log TWT {t.min():.0f}–{t.max():.0f} ms "
              f"every {dtl:g} ms   AI {np.nanmin(ai_s):.0f}–{np.nanmax(ai_s):.0f}")
    if not wells:
        print("No usable wells: skipping the LFM."); return

    sos = butter(4, LFM_CUTOFF_HZ, fs=1000.0 / dt, output="sos")
    lowpass = lambda v: sosfiltfilt(sos, v, axis=-1).astype(np.float32)
    K = 1001; U = np.linspace(0, 1, K)

    def fit_background(ws):
        y = np.concatenate([w["ai_low"][w["valid"]] for w in ws])
        if VEL is not None:
            v = np.concatenate([VEL[w["i"], w["j"]][w["valid"]] for w in ws]).astype(float)
            b, ln_a = np.polyfit(np.log(v), np.log(y), 1)
            if not 0.3 < b < 3:
                b, ln_a = 1.0, np.mean(np.log(y) - np.log(v))
            return ("velocity", np.exp(ln_a), b)
        tw = np.concatenate([samples[w["valid"]] for w in ws])
        c1, c0 = np.polyfit(tw, y, 1)
        return ("trend", c0, c1)

    def bg_trace(m, i, j):
        return m[1] * VEL[i, j].astype(float) ** m[2] if m[0] == "velocity" else m[1] + m[2] * samples

    def bg_volume(m):
        if m[0] == "velocity":
            return m[1] * VEL.astype(np.float32) ** np.float32(m[2])
        return np.broadcast_to((m[1] + m[2] * samples).astype(np.float32), (NI, NX, NT)).copy()

    def well_table(w, m, coord):
        r = w["ai_low"] - bg_trace(m, w["i"], w["j"])
        f, l = w["first"], w["last"]
        r[:f], r[l + 1:] = r[f], r[l]
        k = np.arange(NT)
        dist = np.where(w["valid"], 0, np.minimum(np.abs(k - f), np.abs(k - l))) * dt
        conf = np.where(dist < LFM_TAPER_MS, 0.5 * (1 + np.cos(np.pi * dist / LFM_TAPER_MS)), 0.0)
        prof = t_norm if coord == "time" else np.maximum.accumulate(rgt[w["i"], w["j"]]) + np.arange(NT) * 1e-7
        return np.interp(U, prof, r), np.interp(U, prof, conf, left=0.0, right=0.0)

    def idw(x, y, w):
        return 1.0 / ((x - w["x"]) ** 2 + (y - w["y"]) ** 2 + 1.0) ** (IDW_POWER / 2)

    def residual_field(u, weights, tables):
        k = np.rint(np.clip(u, 0, 1) * (K - 1)).astype(np.int32)
        num = np.zeros(u.shape, np.float32); den = np.zeros(u.shape, np.float32); cmax = np.zeros(u.shape, np.float32)
        for W, (Rw, Cw) in zip(weights, tables):
            Wb = W[..., None] if np.ndim(W) else W
            c = Cw[k].astype(np.float32)
            num += Wb * c * Rw[k]; den += Wb * c; np.maximum(cmax, c, out=cmax)
        return num / (den + 1e-30) * cmax

    model = fit_background(wells)
    if model[0] == "velocity":
        print(f"\nBackground: AI ~ {model[1]:.4g} x Vint^{model[2]:.3f}, fitted on all {len(wells)} well(s)")
        for w in wells:
            m_ = w["valid"]; r_ = np.corrcoef(w["ai_low"][m_], bg_trace(model, w["i"], w["j"])[m_])[0, 1]
            print(f"   {w['name']:12s} correlation of log and velocity background: {r_:.2f}")
    else:
        print(f"\nNo velocity model: background = AI trend with TWT ({model[2]:.3f} per ms)")
    bg_label = "velocity only" if model[0] == "velocity" else "trend only"
    tables = [well_table(w, model, "rgt") for w in wells]
    bg = bg_volume(model)
    lfm_bg = lowpass(bg)
    lfm = lowpass(bg + residual_field(rgt, [idw(Xg, Yg, w) for w in wells], tables))
    del bg

    loo = {}
    if len(wells) >= 2:
        rows = []
        for h in wells:
            others = [w for w in wells if w is not h]
            m = fit_background(others)
            bg_h = bg_trace(m, h["i"], h["j"])
            preds = {bg_label: bg_h}
            for coord, label in (("time", "wells along flat time"), ("rgt", "wells along RGT")):
                tabs = [well_table(w, m, coord) for w in others]
                u = t_norm if coord == "time" else rgt[h["i"], h["j"]]
                preds[label] = bg_h + residual_field(u, [idw(Xg[h["i"], h["j"]], Yg[h["i"], h["j"]], w) for w in others], tabs)
            loo[h["name"]] = {}
            for label, p in preds.items():
                p = lowpass(p); loo[h["name"]][label] = p
                ok, y = h["valid"], h["ai_low"]
                e = (p[ok] - y[ok]) / y[ok] * 100
                rows.append(dict(well=h["name"], method=label, rms_pct=np.sqrt(np.mean(e ** 2)), bias_pct=e.mean(),
                                 correlation=np.corrcoef(p[ok], y[ok])[0, 1]))
        res = pd.DataFrame(rows)
        order = [bg_label, "wells along flat time", "wells along RGT"]
        piv = res.pivot(index="well", columns="method", values="rms_pct")[order]
        piv.loc["MEAN"] = piv.mean()
        print("\nLEAVE-ONE-WELL-OUT: each well predicted from the others. RMS error % (lower is better):")
        print(piv.round(2).to_string())
        print("\nCorrelation with the well's low-passed log (higher is better):")
        print(res.pivot(index="well", columns="method", values="correlation")[order].round(2).to_string())
        res.to_csv(os.path.join(out_dir, f"{stem}_LFM_blind_well_test.csv"), index=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        piv.drop("MEAN").plot.bar(ax=ax, color=["#999999", "#e9c46a", "#7b3fbf"], rot=0)
        ax.set_ylabel("blind-well RMS error (%)"); ax.set_title("Leave-one-well-out: lower is better")
        plots.save(fig, "lfm_blind_test")
    else:
        print("\n(Only one well: the leave-one-well-out test needs at least 2.)")

    n = len(wells)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 6), squeeze=False)
    for ax, w in zip(axes[0], wells):
        ax.plot(w["ai"], samples, color="0.75", lw=0.6, label="log")
        ax.plot(w["ai_low"], samples, color="C3", lw=2, label=f"log <={LFM_CUTOFF_HZ} Hz")
        ax.plot(lfm_bg[w["i"], w["j"]], samples, color="C0", lw=1.3, ls="--", label=bg_label)
        if w["name"] in loo:
            ax.plot(loo[w["name"]]["wells along RGT"], samples, color="#7b3fbf", lw=1.5, label="blind: wells along RGT")
        ax.plot(lfm[w["i"], w["j"]], samples, color="k", lw=1, ls=":", label="final LFM")
        ax.set_ylim(samples[-1], samples[0]); ax.set_title(w["name"]); ax.set_xlabel("AI")
        for t_ in INTERVAL_MS: ax.axhline(t_, color="orange", lw=0.8, ls="--")
    axes[0][0].set_ylabel("TWT (ms)"); axes[0][0].legend(fontsize=7)
    plots.save(fig, "lfm_wells")

    w0 = wells[0]
    vmin, vmax = np.nanpercentile(lfm[w0["i"]], [1, 99])
    tm = tidx(sum(INTERVAL_MS) / 2)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), gridspec_kw={"width_ratios": [1.3, 2, 2]})
    im = axes[0].imshow(lfm[:, :, tm], cmap="viridis", origin="lower", aspect="auto",
                        extent=(xlines[0], xlines[-1], ilines[0], ilines[-1]))
    for w in wells:
        axes[0].plot(w["xl"], w["il"], "r*", ms=12)
        axes[0].annotate(w["name"], (w["xl"], w["il"]), color="w", fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set_title(f"final LFM at {samples[tm]:.0f} ms"); axes[0].set_xlabel("crossline"); axes[0].set_ylabel("inline")
    plt.colorbar(im, ax=axes[0], label="AI")
    for ax, (title, vol) in zip(axes[1:], (("final LFM", lfm), (bg_label, lfm_bg))):
        im = ax.imshow(vol[w0["i"]].T, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax,
                       extent=(xlines[0], xlines[-1], samples[-1], samples[0]))
        ax.axvline(w0["xl"], color="red", lw=1, ls=":")
        ax.set_title(f"{title}: inline {ilines[w0['i']]} (through {w0['name']})")
        ax.set_xlabel("crossline"); ax.set_ylabel("TWT (ms)")
        for t_ in INTERVAL_MS: ax.axhline(t_, color="orange", lw=1, ls="--")
    plt.colorbar(im, ax=axes[2], label="AI")
    plots.save(fig, "lfm_sections")

    OUT["lfm"] = write_like_input(lfm, "LFM")
    OUT["lfm_background"] = write_like_input(lfm_bg, f"LFM_{bg_label.replace(' ', '_')}")

    sel_i = np.arange(w0["i"] % LFM_TRACE_STEP, NI, LFM_TRACE_STEP)
    sel_x = np.arange(w0["j"] % LFM_TRACE_STEP, NX, LFM_TRACE_STEP)
    trace_index = G["trace_index"]
    if (trace_index[np.ix_(sel_i, sel_x)] < 0).any():
        print("Some decimated grid nodes have no input trace: skipping the decimated file.")
        return
    spec = segyio.spec()
    spec.iline, spec.xline = G["il_byte"], G["xl_byte"]
    spec.ilines, spec.xlines = ilines[sel_i], xlines[sel_x]
    spec.samples, spec.sorting, spec.format = samples, 2, 5
    out_path = os.path.join(out_dir, f"{stem}_LFM_step{LFM_TRACE_STEP}.sgy")
    with segyio.open(ctx["SEGY_PATH"], ignore_geometry=True) as src, segyio.create(out_path, spec) as dst:
        dst.text[0] = src.text[0]
        n = 0
        for i in sel_i:
            for j in sel_x:
                dst.header[n] = src.header[int(trace_index[i, j])]
                dst.trace[n] = np.ascontiguousarray(lfm[i, j], dtype=np.float32)
                n += 1
    OUT["lfm_decimated"] = out_path
    print("wrote", os.path.basename(out_path), f"({len(sel_i)} inlines x {len(sel_x)} crosslines)")


if __name__ == "__main__":
    main()
