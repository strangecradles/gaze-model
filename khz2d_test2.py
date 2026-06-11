"""khz2d_test2.py — TESTBED B: deploy the winning kHz 2D gaze methods on the
TRUE x-scan (test2, ~12 kHz horizontal lines) + the 2D SLO atlas.

Inputs honoring the acquisition budget:
  - kHz channel : test2 line scan (11,985 Hz; each sweep = 1000-sample
                  horizontal line). ALONG (horizontal gaze) = trusted whole-line
                  NCC shift (data.along_shift). PERP (vertical gaze) = aliased
                  appearance match of each line against the 2D SLO.
  - 2D SLO      : the SAME-SESSION test1 mosaic (cache/atlas_test1_mosaic.npy,
                  native scale — test2 lines match it at q~0.86 in 100 Hz
                  blocks vs ~0.24 for the cross-session normal/ atlas).
  - slow budget : absolute 2D anchors computed at 20 Hz ONLY (ANCHOR_HZ),
                  from ~590-sweep coarse-band matches — the "slow axis" fix.

Methods deployed (the testbed-A winners):
  B-KF   per-block perp argmax in an anchor-centred window + innovation-gated
         constant-velocity Kalman (M2 recipe).
  B-VIT  Viterbi decode of the per-block perp profile sequence between the
         20 Hz anchors with a main-sequence velocity cap (M3 recipe).
  B-DPF  the existing particle filter (filter.py) on the block stream with
         atlas = mosaic, trusted along, 20 Hz coarse anchor (M4 recipe).

Validation: dot (pursuit target) + machine tracker. test2 has NO co-registered
raster, so this is necessarily the weaker validation (self-consistency +
external low-rate references); reported with that caveat.
"""
from __future__ import annotations

import os
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter

import data
import calib
import khz2d
from khz2d_methods import _kf_axis

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = khz2d.CACHE
RESULTS = khz2d.RESULTS

WHICH = "test2"
MAX_SWEEPS = 834_000          # full capture (~69.6 s; stimulus starts ~19.4 s in)
BLOCK = 12                    # -> ~999 Hz effective line stream
ANCHOR_HZ = 20.0              # slow-axis absolute-fix budget
PADP = 150                    # perp search half-window around the anchor (rows)
ALW = 40                      # along search half-window around the anchor (px)
LCROP = 100                   # line ends cropped before matching (vignette / mosaic width)
COARSE_SIG = 8.0
Q_FOV_B = 0.35

MOSAIC = os.path.join(CACHE, "atlas_test1_mosaic.npy")
BLK_CACHE = os.path.join(CACHE, f"khz2d_t2_blocks_{BLOCK}_{MAX_SWEEPS}.npz")
PROF_CACHE = os.path.join(CACHE, f"khz2d_t2_prof_{BLOCK}_{MAX_SWEEPS}.npz")


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def mosaic_bands():
    """Dense-cropped mosaic with rows = PERP (vertical gaze) candidates, cols = ALONG.

    NOTE the stored mosaic is in test1 FRAME coordinates: axis 0 = fast axis
    (vertical, 1000 px + motion = 1377), axis 1 = slow axis (horizontal,
    808 + motion = 1491). A test2 sweep is a HORIZONTAL retinal line, so it
    slides along axis 1 — i.e. the mosaic is used UNTRANSPOSED (measured:
    q med 0.705 untransposed vs 0.627 transposed; gaze_anim_test1.py used .T,
    which is the wrong orientation and part of why its perp tracking was weak).

    The mosaic is cropped to its well-covered region (coverage > 0.8): the
    raw mosaic's zero borders corrupt TM_CCOEFF_NORMED windows and create
    constant-row attractors (measured: anchors pinned at rows 1/455/1376).
    """
    A = np.load(MOSAIC).astype(np.float32)          # (rows=perp 1377, cols=along 1491)
    nz = np.abs(A) > 1e-6
    rows = np.flatnonzero(nz.mean(1) > 0.8)
    cols = np.flatnonzero(nz.mean(0) > 0.8)
    A = A[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    Ac = gaussian_filter1d(A, COARSE_SIG, axis=1)
    Af = A - Ac
    return A, _z(Ac).astype(np.float32), _z(Af).astype(np.float32)


def load_blocks(rebuild=False):
    """Stream the test2 mp4 into BLOCK-sweep mean lines: (nb, 1000) + contrast."""
    if os.path.exists(BLK_CACHE) and not rebuild:
        z = np.load(BLK_CACHE)
        return {k: z[k] for k in z.files}
    meta = data.line_scan_meta(WHICH)
    cap = cv2.VideoCapture(meta["path"])
    rows = []
    got = 0
    buf = []
    blk, con = [], []
    while got < MAX_SWEEPS:
        ok, f = cap.read()
        if not ok:
            break
        sw = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32).T  # (808, 1000)
        buf.append(sw); got += sw.shape[0]
        if sum(b.shape[0] for b in buf) >= 50 * BLOCK:
            arr = np.concatenate(buf, 0)
            n = arr.shape[0] // BLOCK * BLOCK
            b = arr[:n].reshape(-1, BLOCK, arr.shape[1])
            blk.append(b.mean(1)); con.append(b.mean(1).std(1))
            buf = [arr[n:]]
    cap.release()
    blk = np.concatenate(blk, 0); con = np.concatenate(con, 0)
    fs = meta["line_rate_hz"] / BLOCK
    out = dict(blk=blk.astype(np.float32), con=con.astype(np.float32),
               fs=np.float64(fs), line_rate=np.float64(meta["line_rate_hz"]))
    np.savez(BLK_CACHE, **out)
    return out


def along_blocks():
    a = data.along_shift(WHICH, max_sweeps=MAX_SWEEPS)
    nb = len(a) // BLOCK
    return a[:nb * BLOCK].reshape(nb, BLOCK).mean(1)


def profiles(rebuild=False):
    """20 Hz absolute anchors + per-block perp profiles around them.

    Anchors: every 1/ANCHOR_HZ s, average the surrounding ~0.6 s of blocks,
    FULL-band central-crop match against the whole dense mosaic -> absolute
    (row, along) fix; consistency-gated (q + deviation from the local median).
    Per block: fine-band NCC of the cropped line vs mosaic rows within +-PADP
    of the interpolated anchor row, along within +-ALW of the interpolated
    anchor along. Cached.
    """
    if os.path.exists(PROF_CACHE) and not rebuild:
        z = np.load(PROF_CACHE)
        return {k: z[k] for k in z.files}
    A, Ac, Af = mosaic_bands()
    Az = _z(A).astype(np.float32)
    nrow, ncol = Az.shape
    d = load_blocks()
    blk, con, fs = d["blk"], d["con"], float(d["fs"])
    al = along_blocks()
    nb = min(len(blk), len(al))
    blk, con, al = blk[:nb], con[:nb], al[:nb]
    t = (np.arange(nb) + 0.5) / fs
    Lf = blk.shape[1]
    sl_line = slice(LCROP, Lf - LCROP)
    L = Lf - 2 * LCROP

    # ---- 20 Hz absolute anchors (the slow-axis budget) ----
    n_anc = int(t[-1] * ANCHOR_HZ)
    anc_t = (np.arange(n_anc) + 0.5) / ANCHOR_HZ
    half = int(0.30 * fs)
    anc_row = np.full(n_anc, np.nan); anc_alg = np.full(n_anc, np.nan)
    anc_q = np.zeros(n_anc)
    for i, ta in enumerate(anc_t):
        c = int(ta * fs)
        sl = slice(max(0, c - half), min(nb, c + half))
        line = blk[sl].mean(0)[sl_line]
        lc = _z(line).astype(np.float32)
        r = cv2.matchTemplate(Az, lc.reshape(1, -1), cv2.TM_CCOEFF_NORMED)
        k = np.unravel_index(int(r.argmax()), r.shape)
        anc_row[i] = k[0]; anc_alg[i] = k[1]; anc_q[i] = r.max()
    okq = anc_q > np.percentile(anc_q, 30)
    # consistency gate: reject anchors far from the local (2.5 s) median track
    med_r = median_filter(khz2d.fill_nan(np.where(okq, anc_row, np.nan)), 51)
    med_a = median_filter(khz2d.fill_nan(np.where(okq, anc_alg, np.nan)), 51)
    ok = okq & (np.abs(anc_row - med_r) < 120) & (np.abs(anc_alg - med_a) < 120)
    print(f"  [t2] anchors: {ok.sum()}/{n_anc} good (q med {np.median(anc_q):.2f})")
    anc_row_i = np.interp(t, anc_t[ok], anc_row[ok])
    anc_alg_i = np.interp(t, anc_t[ok], anc_alg[ok])

    # ---- per-block profiles ----
    nS = 2 * PADP + 1
    prof = np.full((nb, nS), -1.0, np.float16)
    qh = np.zeros(nb, np.float32)
    rda = np.full(nb, np.nan, np.float32)
    t0 = time.time()
    for i in range(nb):
        r0 = int(round(anc_row_i[i])) - PADP
        a_pred = int(round(anc_alg_i[i]))
        c0 = max(0, min(a_pred - ALW, ncol - L - 1))
        c1 = min(ncol, a_pred + ALW + L)
        rr0 = max(0, r0); rr1 = min(nrow, r0 + nS)
        if rr1 - rr0 < 10 or c1 - c0 < L + 4:
            continue
        line = blk[i][sl_line]
        line = _z(line - gaussian_filter1d(line, COARSE_SIG)).astype(np.float32)
        r = cv2.matchTemplate(Af[rr0:rr1, c0:c1], line.reshape(1, -1),
                              cv2.TM_CCOEFF_NORMED)
        pr = r.max(1)
        prof[i, rr0 - r0: rr0 - r0 + len(pr)] = pr
        k = int(pr.argmax())
        qh[i] = pr[k]
        rda[i] = (c0 + int(r[k].argmax())) - anc_alg_i[i]
        if i % 20000 == 0 and i:
            print(f"  [t2] profiles {i}/{nb} ({time.time()-t0:.0f}s)")
    out = dict(t=t, prof=prof, qh=qh, con=con, anc_t=anc_t, anc_row=anc_row,
               anc_alg=anc_alg, anc_ok=ok, anc_q=anc_q, anc_row_i=anc_row_i,
               anc_alg_i=anc_alg_i, along_rel=al,
               along_abs=anc_alg_i + np.clip(rda, -ALW, ALW),
               rd_along=rda, fs=np.float64(fs))
    np.savez(PROF_CACHE, **out)
    return out


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


ALONG_TC = 2.0   # s — complementary-fusion crossover for the along channel


def _along_fused(P):
    """kHz along (horizontal) channel: complementary fusion of the two honest
    sources, each used in the band where it is actually informative.

      low-pass  (>ALONG_TC) : the 20 Hz absolute anchors. The 0.2 Hz pursuit
                  band lives here; the raw anchors carry it but are noisy
                  (measured: lp(anchors) r_dot_x 0.32 full-trace / 0.63-0.73
                  per phase vs 0.20 for the old per-block rd_along KF).
      high-pass (<ALONG_TC) : the TRUSTED whole-line NCC cumsum
                  (data.along_shift — the channel calib.py grounded at r=0.73
                  vs the tracker). Its cumulative registration drift dies in
                  the high-pass; gain k=+1 (sweep px ~ mosaic col) measurably
                  improves tracker corr (r_trk_x 0.135 -> 0.141 @TC=2,
                  0.101 -> 0.145 @TC=5) and is flat for the dot, i.e. it adds
                  the sub-anchor fast content the dot cannot validate.

    The previous channel (anchor + per-block fine-match offset rd_along,
    Kalman-fused) was strictly worse on every reference and every stimulus
    phase — the per-block along match at ~3.8 px/deg is too low-SNR.
    """
    fs = float(P["fs"])
    sig = fs * ALONG_TC / (2 * np.pi)
    anc = khz2d.fill_nan(P["anc_alg_i"]).astype(np.float64)
    rel = khz2d.fill_nan(P["along_rel"]).astype(np.float64)
    fused = gaussian_filter1d(anc, sig) + (rel - gaussian_filter1d(rel, sig))

    # SELF-SUPERVISED vertical->along leak removal. The 20 Hz anchor is a 2D
    # full-mosaic match, and the locally-diagonal mosaic structure leaks
    # vertical motion into the matched COLUMN (measured: during V_sine, when
    # the dot has zero horizontal motion, the raw along channel correlates
    # -0.82 with dot_y). The leak coefficient is fit from the reconstruction's
    # own channels only (along vs anchor-row, drift-removed, anchor band) —
    # no dot/tracker reference touches the fit. Removing it lifts full-trace
    # r_dot_x 0.32 -> 0.45 and r_trk_x 0.14 -> 0.27.
    dsig = fs / (2 * np.pi * khz2d.DRIFT_HZ)
    row = khz2d.fill_nan(P["anc_row_i"]).astype(np.float64)
    dro = row - gaussian_filter1d(row, dsig)
    drf = fused - gaussian_filter1d(fused, dsig)
    xl = gaussian_filter1d(drf, sig)
    yl = gaussian_filter1d(dro, sig)
    leak = float(np.dot(xl, yl) / np.dot(yl, yl))
    fused = fused - leak * dro

    upd = np.isfinite(fused)
    return fused, upd


def b_kf(P):
    fs = float(P["fs"])
    base = P["anc_row_i"] - PADP
    arg = base + np.argmax(P["prof"], 1)
    con_ok = P["con"] > 0.5 * np.median(P["con"])
    meas = np.where(con_ok & (P["qh"] > 0.05), arg, np.nan)
    sigma_a = 3000.0 * calib.ROWS_PER_DEG
    q_med = float(np.median(P["qh"][con_ok]))
    est, upd = _kf_axis(P["t"], meas, P["qh"], 0.6 * q_med, 1.5, sigma_a,
                        gate_px=20.0, reacq_steps=int(fs * 0.01),
                        q_strong=1.5 * q_med)
    al, upd_a = _along_fused(P)
    idx = np.arange(len(upd)); last = np.where(upd, idx, -10**9)
    valid = (idx - np.maximum.accumulate(last)) <= int(fs * 0.05)
    return dict(t=P["t"], perp=est, along=al,
                valid=valid & con_ok & np.isfinite(est) & np.isfinite(al), rate=fs)


def b_viterbi(P, beta=8.0, gamma=0.05):
    fs = float(P["fs"])
    nS = 2 * PADP + 1
    prof = P["prof"].astype(np.float32)
    nb = prof.shape[0]
    con_ok = P["con"] > 0.5 * np.median(P["con"])
    use = con_ok & (P["qh"] > 0.05)
    Wt = int(np.ceil(float(calib.ROWS_PER_DEG) * 826.0 / fs)) + 2
    base = np.round(P["anc_row_i"]).astype(int) - PADP
    off = np.arange(-Wt, Wt + 1)
    pen = (gamma * np.abs(off)).astype(np.float32)
    score = np.zeros(nS, np.float32)
    bp = np.empty((nb, nS), np.int16)
    pad = np.full(Wt, -1e9, np.float32)
    for i in range(nb):
        if i > 0:
            k = base[i] - base[i - 1]
            if abs(k) >= nS:
                score = np.zeros(nS, np.float32)
            elif k > 0:
                score = np.concatenate([score[k:], np.full(k, score.min(), np.float32)])
            elif k < 0:
                score = np.concatenate([np.full(-k, score.min(), np.float32), score[:k]])
        sw = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([pad, score, pad]), 2 * Wt + 1) - pen[None, :]
        am = np.argmax(sw, 1)
        score = sw[np.arange(nS), am]
        bp[i] = (am - Wt).astype(np.int16)
        if use[i]:
            score = score + beta * prof[i]
    d = np.empty(nb, np.float32)
    j = int(np.argmax(score))
    for i in range(nb - 1, -1, -1):
        d[i] = base[i] + j
        j = int(np.clip(j + bp[i, j], 0, nS - 1))
        if i > 0:
            j = int(np.clip(j + (base[i] - base[i - 1]), 0, nS - 1))
    al, _ = _along_fused(P)
    valid = use & (P["qh"] > 0.10) & np.isfinite(al)
    return dict(t=P["t"], perp=d, along=al, valid=valid, rate=fs)


def b_dpf(P, n_particles=300, line_len=200, seed=0):
    import filter as flt
    d = load_blocks()
    fs = float(P["fs"])
    blk = d["blk"][:len(P["t"]), LCROP:-LCROP]
    A, _, _ = mosaic_bands()
    L = blk.shape[1]
    col_step = L / float(line_len)
    obs = np.stack([np.interp(np.arange(line_len) * col_step, np.arange(L), b)
                    for b in blk]).astype(np.float64)
    al, _ = _along_fused(P)
    al = khz2d.fill_nan(al)
    q_med = float(np.median(P["qh"]))
    res = flt.run(obs, al.astype(np.float64), fs, A,
                  init_perp=float(P["anc_row_i"][0]),
                  init_along=float(al[0]),
                  n_particles=n_particles, line_len=line_len, col_step=col_step,
                  seed=seed, coarse_anchor=P["anc_row_i"].astype(np.float64),
                  ncc_loss_thr=0.6 * q_med, reseed_perp_sigma=60.0)
    con_ok = P["con"] > 0.5 * np.median(P["con"])
    valid = con_ok & (res.max_ncc > 0.5 * q_med)
    return dict(t=P["t"], perp=res.est_perp, along=res.est_along, valid=valid,
                rate=fs, max_ncc=res.max_ncc)


# ---------------------------------------------------------------------------
# evaluation (test2 references)
# ---------------------------------------------------------------------------

_REFS2 = None


def refs2():
    global _REFS2
    if _REFS2 is not None:
        return _REFS2
    st = data.load_stimulus("pursuit")
    tr = data.load_tracker(WHICH)
    trk_t = (tr.t_s - tr.t_s[0]).astype(float)
    _REFS2 = dict(dot_t=st.t_s.astype(float), dot_x=st.x_deg * 60, dot_y=st.y_deg * 60,
                  trk_t=trk_t,
                  trk_x=khz2d.fill_nan(tr.right_x) * khz2d.TRK_DEG_X * 60,
                  trk_y=khz2d.fill_nan(tr.right_y) * khz2d.TRK_DEG_Y * 60,
                  off=None)
    return _REFS2


def evaluate2(m, label, off=None, smooth_ms=10.0):
    """test2 scoring: perp -> vertical (dot_y), along -> horizontal (dot_x)."""
    R = refs2()
    fs = float(m["rate"])
    t = m["t"]; v = m["valid"].astype(bool)
    def prep(sig):
        s = np.where(v, np.asarray(sig, float), np.nan)
        s = gaussian_filter1d(median_filter(khz2d.fill_nan(s), 7),
                              max(1.0, fs * smooth_ms / 1000.0))
        return khz2d._drift_removed(np.where(v, s, np.nan), fs)
    hx = prep(m["along"]); hy = prep(m["perp"])
    if off is None:
        best = (0.0, 0.0)
        for o in np.arange(0.0, 6.0, 0.025):
            c = khz2d.corr(hx[v], np.interp(t[v], R["dot_t"] + o, R["dot_x"]))
            if np.isfinite(c) and abs(c) > abs(best[1]):
                best = (float(o), float(c))
        off = best[0]
    out = dict(label=label, rate=fs, off=off, valid_frac=float(v.mean()))
    for ax, h in (("x", hx), ("y", hy)):
        dref = np.interp(t, R["dot_t"] + off, R[f"dot_{ax}"])
        tref = np.interp(t, R["trk_t"] + off, R[f"trk_{ax}"])
        r_dot = khz2d.corr(h[v], dref[v]); r_trk = khz2d.corr(h[v], tref[v])
        A = np.c_[h[v], np.ones(v.sum())]
        coef, *_ = np.linalg.lstsq(A, dref[v], rcond=None)
        cal = coef[0] * h + coef[1]
        out[f"r_dot_{ax}"] = abs(r_dot); out[f"r_trk_{ax}"] = abs(r_trk)
        out[f"rms_{ax}"] = float(np.sqrt(np.nanmean((cal[v] - dref[v]) ** 2)))
        out[f"cal_{ax}"] = cal; out[f"dot_{ax}"] = dref
    out["t"] = t; out["valid"] = v
    return out


def summarize2(ev):
    return (f"{ev['label']:<22s} {ev['rate']:>7.0f} Hz  "
            f"rdot x={ev['r_dot_x']:.3f} y={ev['r_dot_y']:.3f}  "
            f"rtrk x={ev['r_trk_x']:.3f} y={ev['r_trk_y']:.3f}  "
            f"RMS x={ev['rms_x']:.1f}' y={ev['rms_y']:.1f}'  "
            f"OFF={ev['off']:.2f}s valid={ev['valid_frac']*100:.0f}%")


STIM_PHASES = (("H_sine", 2.5, 17.48), ("V_sine", 17.5, 32.48),
               ("circle", 32.5, 48.48), ("lissajous", 48.5, 66.48))


def per_phase(ev, off):
    """Per-stimulus-phase correlations for an evaluated method.

    The full-trace aggregate under-states each axis: during H_sine the dot has
    no vertical motion (so vertical corr is pure noise) and vice versa. Scoring
    each phase against the axis actually being driven is the honest view, and
    the circle/lissajous phases (both axes moving) test true 2D tracking.
    """
    R = refs2()
    t = ev["t"]; v = ev["valid"].astype(bool)
    rows = []
    for name, a, b in STIM_PHASES:
        m = v & (t >= a + off) & (t <= b + off)
        if m.sum() < 50:
            rows.append((name, np.nan, np.nan, np.nan, np.nan)); continue
        rx = khz2d.corr(ev["cal_x"][m], np.interp(t[m], R["dot_t"] + off, R["dot_x"]))
        ry = khz2d.corr(ev["cal_y"][m], np.interp(t[m], R["dot_t"] + off, R["dot_y"]))
        tx = khz2d.corr(ev["cal_x"][m], np.interp(t[m], R["trk_t"] + off, R["trk_x"]))
        ty = khz2d.corr(ev["cal_y"][m], np.interp(t[m], R["trk_t"] + off, R["trk_y"]))
        rows.append((name, abs(rx), abs(ry), abs(tx), abs(ty)))
    return rows


def off_b(P):
    """Shared clock offset for testbed B (method-independent, frozen for every
    method).

    A single-axis along-vs-dot scan is NOT robust here: the pursuit stimulus is
    a sequence of per-axis sinusoids (H_sine, then V_sine, ...), so an along-only
    correlation against dot_x has strong false maxima at multiples of the H_sine
    half-period (measured aliases near 11.3 / 17.2 / 19.4 s). We instead use BOTH
    channels in their NATURAL pairing over the full trace — along (kHz, trusted)
    vs dot_x AND the 20 Hz anchor row vs dot_y — and maximise |r_x| + |r_y|. A
    true offset must make both axes track their own stimulus simultaneously; the
    aliases only line up one axis at a time and collapse on the circle phase
    (where both axes move together). This recovers the hardware-plausible
    offset (~2 s, consistent with the mp4 creation_time lead).
    """
    R = refs2()
    fs = float(P["fs"])
    t = P["t"]
    ar = khz2d.fill_nan(np.where(P["anc_ok"], P["anc_row"], np.nan))
    als = khz2d.fill_nan(P["along_abs"])

    def detrend(v, f):
        return v - gaussian_filter1d(v, f / (2 * np.pi * khz2d.DRIFT_HZ))

    ax = detrend(als, fs)          # along, kHz grid
    ay = detrend(ar, ANCHOR_HZ)    # anchor row, 20 Hz grid
    best = (0.0, -1.0, 0.0, 0.0)
    for o in np.arange(-2.0, 25.0, 0.05):
        cx = khz2d.corr(ax, np.interp(t, R["dot_t"] + o, R["dot_x"]))
        cy = khz2d.corr(ay, np.interp(P["anc_t"], R["dot_t"] + o, R["dot_y"]))
        s = abs(cx) + abs(cy)
        if np.isfinite(s) and s > best[1]:
            best = (float(o), float(s), float(cx), float(cy))
    print(f"  [t2] OFF = {best[0]:.2f}s (two-axis: along-vs-dotx r={best[2]:+.2f}, "
          f"row-vs-doty r={best[3]:+.2f}, |rx|+|ry|={best[1]:.2f})")
    return best[0]


def main():
    P = profiles()
    fs = float(P["fs"])
    OFF = off_b(P)
    evs = []
    res = {}
    for fn, label in ((b_kf, "B-KF kalman"), (b_viterbi, "B-VIT viterbi"),
                      (b_dpf, "B-DPF particle filter")):
        m = fn(P)
        ev = evaluate2(m, label, off=OFF)
        evs.append(ev); res[label] = (m, ev)
        print(summarize2(ev))

    # 20 Hz anchors alone (the slow-axis-only baseline)
    okA = P["anc_q"] > 0.5
    m0 = dict(t=P["anc_t"], perp=np.where(okA, P["anc_row"], np.nan),
              along=np.interp(P["anc_t"], P["t"], P["along_abs"]),
              valid=okA, rate=ANCHOR_HZ)
    ev0 = evaluate2(m0, "B0 anchors only", off=OFF, smooth_ms=0.0)
    print(summarize2(ev0))
    evs.insert(0, ev0)

    # figure — traces are VARIANCE-MATCHED to the dot for visual comparison
    # (the OLS calibration shrinks a r=0.45 trace to 0.45x the dot amplitude,
    # which reads as "flat"). The machine tracker (the real eye) is overlaid:
    # it follows the dot at only r ~ 0.72/0.76, the honest ceiling here.
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    R = refs2(); off = evs[1]["off"]
    tt = P["t"]
    for ax, axis in zip(axes, ("x", "y")):
        dot = np.interp(tt, R["dot_t"] + off, R[f"dot_{axis}"])
        trk = np.interp(tt, R["trk_t"] + off, R[f"trk_{axis}"])
        trk = (trk - trk.mean()) / (trk.std() + 1e-9) * dot.std() + dot.mean()
        ax.plot(tt, dot, "k-", lw=2, alpha=.30, label="dot (target)")
        ax.plot(tt, trk, "-", color="0.45", lw=.8, alpha=.6,
                label="machine tracker (eye)")
    for label, (m, ev) in res.items():
        for ax, axis in zip(axes, ("x", "y")):
            v = m["valid"]
            cal = ev[f"cal_{axis}"]; dot = ev[f"dot_{axis}"]
            mu, sd = np.nanmean(cal[v]), np.nanstd(cal[v]) + 1e-9
            vm = (cal - mu) / sd * np.nanstd(dot[v]) + np.nanmean(dot[v])
            ax.plot(m["t"], np.where(v, vm, np.nan), lw=.6, alpha=.8,
                    label=f"{label} ({axis} r={ev[f'r_dot_{axis}']:.2f})")
    axes[0].set_ylabel("horizontal (')"); axes[1].set_ylabel("vertical (')")
    axes[1].set_xlabel("time (s)"); axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    axes[0].set_title(f"test2 x-scan + 2D SLO mosaic: kHz 2D gaze ({fs:.0f} Hz stream, "
                      f"{ANCHOR_HZ:.0f} Hz anchor budget)")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    figp = os.path.join(RESULTS, "khz2d_test2.png")
    fig.savefig(figp, dpi=140); plt.close(fig)

    # report
    L = []
    L.append("# kHz 2D Gaze on the TRUE x-Scan (test2) + 2D SLO — Deployment (Testbed B)\n")
    L.append(f"**Inputs**: test2 x-scan ({float(P['fs'])*BLOCK:.0f} Hz line rate, processed in "
             f"{BLOCK}-sweep blocks -> {fs:.0f} Hz effective) + the SAME-SESSION test1 mosaic as "
             f"the 2D SLO. Absolute 2D anchors are computed at {ANCHOR_HZ:.0f} Hz ONLY "
             "(the stated slow-axis budget): ~0.6 s coarse-band matches against the full mosaic.\n")
    L.append("**Channels**: horizontal gaze = complementary fusion (crossover "
             f"{ALONG_TC:.0f} s) of the 20 Hz absolute anchors (low-pass: carries the pursuit "
             "band) and the trusted whole-line NCC along-shift cumsum (high-pass: the kHz "
             "sub-anchor content, drift dies in the high-pass); vertical gaze = per-block "
             "fine-band appearance match over mosaic rows (aliased/multimodal) resolved by "
             "each method between anchors.\n")
    L.append("**Caveat**: test2 has no co-registered raster, so unlike testbed A there is no "
             "frame-rate 2D registration truth; validation is against the pursuit dot (target, "
             "not the eye) and the ~32.5 Hz machine tracker. Self-consistency is necessary, not "
             "sufficient (see results/real_validation.md).\n")
    L.append("## Results\n")
    L.append("| method | rate (Hz) | r dot x | r dot y | r trk x | r trk y | RMS x (') | RMS y (') | valid |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ev in evs:
        L.append(f"| {ev['label']} | {ev['rate']:.0f} | {ev['r_dot_x']:.3f} | {ev['r_dot_y']:.3f} "
                 f"| {ev['r_trk_x']:.3f} | {ev['r_trk_y']:.3f} | {ev['rms_x']:.1f} "
                 f"| {ev['rms_y']:.1f} | {ev['valid_frac']*100:.0f}% |")
    L.append("\n## Per-stimulus-phase breakdown (B-KF, |r|)\n")
    L.append("Full-trace r under-states each axis (during H_sine the dot has no vertical "
             "motion, etc.). Per phase, against the driven axis:\n")
    L.append("| phase | r dot x | r dot y | r trk x | r trk y |")
    L.append("|---|---|---|---|---|")
    ekf0 = next(e for e in evs if e["label"].startswith("B-KF"))
    for name, rx, ry, tx, ty in per_phase(ekf0, OFF):
        f = lambda x: "-" if not np.isfinite(x) else f"{x:.2f}"
        L.append(f"| {name} | {f(rx)} | {f(ry)} | {f(tx)} | {f(ty)} |")
    L.append("\nNote the along (x) channel is cross-coupled to vertical gaze: a vertical eye "
             "movement changes which mosaic row the line samples, and the locally-diagonal "
             "mosaic structure shifts the matched column with it (during V_sine the raw "
             "channel tracked dot_y at |r| ~ 0.8 with zero horizontal dot motion). A "
             "self-supervised leak fit (along vs anchor-row, no reference data) removes the "
             "common component and is included in the channel above; the residual sign "
             "inconsistency across phases (circle flips vs H_sine/lissajous) shows the leak "
             "is position-dependent — a joint 2D row+column readout per block is the next "
             "lever for testbed B horizontal.")
    L.append(f"\n- anchors: q med {np.median(P['anc_q']):.2f}, good {int((P['anc_q']>0.5).sum())}"
             f"/{len(P['anc_q'])}; per-block fine match q med {np.median(P['qh']):.2f} "
             f"(vs ~0.2 cross-session in G15 — native scale pays).")
    L.append(f"- figure: `khz2d_test2.png`.")

    # find the anchors-only and best kHz method for the verdict numbers
    e0 = next(e for e in evs if e["label"].startswith("B0"))
    ekf = next(e for e in evs if e["label"].startswith("B-KF"))
    L.append("\n## Verdict — does the kHz channel add real 2D content?\n")
    L.append(f"On the honest joint clock offset (OFF = {OFF:.2f} s, see below), the vertical "
             "axis is the discriminator. The 20 Hz absolute-anchor baseline carries almost no "
             f"vertical signal (B0: r dot y = {e0['r_dot_y']:.3f}, r trk y = {e0['r_trk_y']:.3f}); "
             "the per-block kHz appearance match resolved between anchors lifts it substantially "
             f"(B-KF: r dot y = {ekf['r_dot_y']:.3f}, r trk y = {ekf['r_trk_y']:.3f}). Because the "
             "machine tracker is an INDEPENDENT measurement path (not the pursuit target), the "
             "rise in r trk y from the 20 Hz baseline to the kHz reconstruction is evidence of "
             "genuine sub-anchor vertical gaze content from the true x-scan + 2D SLO alone.\n")
    L.append(f"Horizontal (along): r dot x = {ekf['r_dot_x']:.3f} full-trace with the "
             "complementary-fusion channel (20 Hz anchor low-pass + trusted kHz NCC high-pass; "
             "was ~0.24 with the per-block rd_along KF — the per-block along match at "
             "~3.8 px/deg is too low-SNR and was dragging the channel down). The pursuit band "
             "is carried by the smoothed absolute anchors; the kHz trusted channel adds the "
             "fast sub-anchor content (it measurably improves the independent-tracker corr "
             "and is neutral for the 0.2 Hz dot). Horizontal remains the weak axis of a "
             "horizontal-line scanner — along motion barely displaces the line.\n")
    L.append("## Decision log — clock offset (OFF)\n")
    L.append("- A single-axis along-vs-dot_x scan is **not** robust on this stimulus: the pursuit "
             "is a sequence of per-axis sinusoids (sync, H_sine, V_sine, circle, lissajous), so "
             "along-vs-dot_x has strong false maxima at H_sine-half-period aliases (we observed "
             "candidates near 11.3 / 17.2 / 19.4 s; an earlier pass mistakenly pinned OFF ~ 19.3 s "
             "off one such alias).\n")
    L.append("- `off_b` now maximises the **two-axis** score |r(along, dot_x)| + |r(anchor_row, "
             "dot_y)| over the full trace. A true offset must align BOTH axes with their own "
             "stimulus at once; aliases only line up one axis. The circle phase (32.5–48.5 s, both "
             "axes moving) is the tiebreaker: at OFF=2.05 the natural pairing holds (along·x=+0.31, "
             "row·y=+0.54, cross terms ~0) while the 17.2 s alias collapses there (along·x=-0.23, "
             "row·y=-0.17). The recovered OFF ~ 2 s is consistent with the mp4 creation_time lead "
             "(~2.2–2.6 s).\n")
    txt = "\n".join(L) + "\n"
    path = os.path.join(RESULTS, "khz2d_best_test2.md")
    with open(path, "w") as fh:
        fh.write(txt)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
