"""people_fov_pf.py — shared particle-filter harness for people_data_fov captures.

Builds/loads per-subject chain + line caches and runs M4 (physics PF) at full
line rate. Used by people_data_fov_run.py and pf_gain_sweep.py.
"""
from __future__ import annotations

import glob
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter

import data
import filter as flt
import khz2d

HERE = os.path.dirname(os.path.abspath(__file__))
PEOPLE_DIR = os.path.join(HERE, "people_data_fov")
CACHE_ROOT = os.path.join(HERE, "cache", "people_fov")
STIMULUS = "pursuit_fov"

CROPV = khz2d.CROPV
PADH = khz2d.PADH
RDY_MAX = khz2d.RDY_MAX
MAXINC = khz2d.MAXINC
Q_FOV = khz2d.Q_FOV
CONTRAST_FRAC = khz2d.CONTRAST_FRAC
TRK_DEG_X = data.UNITS.tracker_deg_per_unit_x
TRK_DEG_Y = data.UNITS.tracker_deg_per_unit_y
ARC_PER_PX = 60.0 / 126.0


@dataclass(frozen=True)
class Subject:
    name: str
    stem: str

    @property
    def slo_path(self) -> str:
        return os.path.join(PEOPLE_DIR, f"{self.stem}_SLO_0.mp4")

    @property
    def csv_path(self) -> str:
        return os.path.join(PEOPLE_DIR, f"{self.stem}.csv")

    @property
    def cache_dir(self) -> str:
        d = os.path.join(CACHE_ROOT, self.name)
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def chain_cache(self) -> str:
        return os.path.join(self.cache_dir, "chain.npz")

    @property
    def line_cache(self) -> str:
        return os.path.join(self.cache_dir, "lines.npz")


def discover_subjects() -> list[Subject]:
    out = []
    for p in sorted(glob.glob(os.path.join(PEOPLE_DIR, "*_SLO_0.mp4"))):
        stem = os.path.basename(p).replace("_SLO_0.mp4", "")
        m = re.match(r"video_playback_(.+)_\d{8}_\d{6}$", stem)
        name = m.group(1) if m else stem
        out.append(Subject(name=name, stem=stem))
    return out


def subject_by_name(name: str) -> Subject:
    for s in discover_subjects():
        if s.name == name:
            return s
    raise KeyError(f"unknown subject {name!r}")


def load_tracker(sub: Subject) -> data.MachineTrack:
    df = pd.read_csv(sub.csv_path)
    t = pd.to_datetime(df["timestamp"])
    t_s = (t - t.iloc[0]).dt.total_seconds().to_numpy().astype(np.float64)
    dur = t_s[-1] - t_s[0] if len(t_s) > 1 else 1.0
    rate = len(df) / dur if dur > 0 else float("nan")
    return data.MachineTrack(
        t_s,
        df["right_x"].astype(np.float64).to_numpy(),
        df["right_y"].astype(np.float64).to_numpy(),
        df["right_area_mm2"].astype(np.float64).to_numpy(),
        df["video_frame"].astype(np.float64).to_numpy(),
        float(rate), sub.name)


def build_chain(sub: Subject, rebuild: bool = False) -> dict:
    if os.path.exists(sub.chain_cache) and not rebuild:
        z = np.load(sub.chain_cache)
        return {k: z[k] for k in z.files}
    S, pad = 20, 80
    incs = [(0.0, 0.0)]
    qs = [1.0]
    prev = None
    cap = cv2.VideoCapture(sub.slo_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f = 0
    while True:
        okr, fr = cap.read()
        if not okr:
            break
        g = data._deband(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        cur = ((g - g.mean()) / (g.std() + 1e-9)).astype(np.float32)
        if prev is not None:
            H, W = cur.shape
            nstrip = W // S
            refp = cv2.copyMakeBorder(prev, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
            fy = np.empty(nstrip); fx = np.empty(nstrip); fq = np.empty(nstrip)
            for s in range(nstrip):
                c = s * S
                r = cv2.matchTemplate(refp[:, c:c + S + 2 * pad], cur[:, c:c + S],
                                      cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                fy[s] = loc[1] - pad; fx[s] = loc[0] - pad; fq[s] = mx
            incs.append((float(np.median(fy)), float(np.median(fx))))
            qs.append(float(np.median(fq)))
        prev = cur
        f += 1
        if f % 200 == 0:
            print(f"  [{sub.name} chain] frame {f}")
    cap.release()
    incs = np.array(incs, np.float64)
    q = np.array(qs)
    ok = (q > 0.30) & (np.abs(incs).max(1) <= MAXINC)
    pos = np.cumsum(np.where(ok[:, None], incs, 0.0), 0)
    out = dict(t=np.arange(len(pos)) / fps, x=pos[:, 1], y=pos[:, 0],
               inc_x=np.where(ok, incs[:, 1], 0.0),
               inc_y=np.where(ok, incs[:, 0], 0.0),
               ok=ok, q=q, fps=np.float64(fps))
    np.savez(sub.chain_cache, **out)
    return out


def _read_frames(sub: Subject) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(sub.slo_path)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        i += 1
    cap.release()


def _atlas_dewarp_cols(prv_db: np.ndarray, vx: float, vy: float,
                       order: int = 3, sign: float = 1.0) -> np.ndarray:
    """Azimipour intra-frame dewarp of the previous-frame atlas (eqs 9-10).

    Identical math to aoslo_image_recon.dewarp_frame, but the slow/time axis in
    this pipeline is COLUMNS (the galvo sweeps horizontally across c=0..W-1 over
    one ~55 ms frame), so the per-line stabilizing shift is a function of column,
    not row. We remove the WITHIN-FRAME eye motion that the bulk chain-warp left
    behind: assuming linear column-time, the eye displacement at column c relative
    to the frame centre is v*(c+0.5)/W - v/2, and the stabilizing shift is its
    negative (paper convention x_hat = -gx_est).

      corrected[r, c] = prv_db( r - disp_y[c],  c - disp_x[c] )

    `vx`, `vy` are the horizontal/vertical eye velocity (px per frame) during the
    atlas frame's acquisition. Only the within-frame *ramp* matters; the constant
    (bulk) term is already removed by the chain warpAffine, so a column-mean offset
    here is a no-op. order=3 cubic resample (== paper griddata cubic)."""
    Hc, W = prv_db.shape
    frac = (np.arange(W) + 0.5) / W - 0.5                    # -0.5..+0.5 over columns
    disp_x = sign * vx * frac                                # within-frame horiz disp (px)
    disp_y = sign * vy * frac                                # within-frame vert disp (px)
    rr, cc = np.meshgrid(np.arange(Hc), np.arange(W), indexing="ij")
    rows = rr - disp_y[None, :]                              # stabilize = subtract disp
    cols = cc - disp_x[None, :]
    out = map_coordinates(prv_db, [rows.ravel(), cols.ravel()],
                          order=order, mode="nearest")
    return out.reshape(Hc, W).astype(prv_db.dtype)


def _dewarp_line_cache(sub: Subject, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    return os.path.join(sub.cache_dir, f"lines_dewarp{suffix}.npz")


def build_line_measurements(sub: Subject, rebuild: bool = False,
                            dewarp_atlas: bool = False,
                            dewarp_order: int = 3, dewarp_sign: float = 1.0,
                            dewarp_tag: str = "", rebuild_chain: bool = False,
                            ref_frames: int = 1) -> dict:
    if ref_frames > 1:
        cache = os.path.join(sub.cache_dir, f"lines_refavg{ref_frames}.npz")
    elif dewarp_atlas:
        cache = _dewarp_line_cache(sub, dewarp_tag)
    else:
        cache = sub.line_cache
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    # chain is deterministic and unchanged by the dewarp; rebuild it only on request
    ch = build_chain(sub, rebuild=rebuild_chain)
    fps = float(ch["fps"])
    T, FR, COL, LAMV, RDX, QV, QH, CON, OK, PROF = [], [], [], [], [], [], [], [], [], []
    prev_raw = None
    raw_hist: deque = deque(maxlen=max(1, ref_frames))   # recent RAW frames f-1..f-K
    nf = 0
    for f, raw in _read_frames(sub):
        nf = f + 1
        if prev_raw is None:
            prev_raw = raw
            raw_hist.append(raw)
            continue
        H, W = raw.shape
        ok_f = bool(ch["ok"][f])
        sx = -float(ch["inc_x"][f]); sy = -float(ch["inc_y"][f])
        if ok_f:
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            prevw = cv2.warpAffine(prev_raw, M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        else:
            prevw = prev_raw
        cur_db = data._deband(raw)[CROPV:H - CROPV]
        prv_db = data._deband(prevw)[CROPV:H - CROPV]
        Hc = cur_db.shape[0]
        curz = khz2d._zcols(cur_db); prvz = khz2d._zcols(prv_db)

        npad = Hc + 2 * RDY_MAX
        A = np.fft.rfft(curz, n=npad, axis=0)
        B = np.fft.rfft(prvz, n=npad, axis=0)
        cc = np.fft.irfft(np.conj(A) * B, n=npad, axis=0)
        ccb = np.concatenate([cc[-RDY_MAX:], cc[:RDY_MAX + 1]], 0)
        pk = np.argmax(ccb, 0)
        lam = np.empty(W, np.float32); qv = np.empty(W, np.float32)
        for c in range(W):
            k = int(pk[c])
            lam[c] = khz2d._parab(ccb[:, c], k) - RDY_MAX
            qv[c] = ccb[k, c] / Hc

        idx = np.arange(Hc, dtype=np.float32)[:, None] - lam[None, :]
        i0 = np.clip(np.floor(idx).astype(np.int32), 0, Hc - 2)
        frac = np.clip(idx - i0, 0.0, 1.0)
        cols = np.arange(W)[None, :].repeat(Hc, 0)
        cural = cur_db[i0, cols] * (1 - frac) + cur_db[i0 + 1, cols] * frac
        curalz = khz2d._zcols(cural)
        # Horizontal localization atlas. Default: the chain-aligned previous frame
        # (prvz, identical to the committed baseline). With --dewarp-atlas, remove
        # the atlas frame's WITHIN-FRAME eye motion (Azimipour eqs 9-10) so the
        # reference is not rolling-shutter-distorted on the slow/horizontal axis.
        # The vertical/along path above (prvz -> lam) is left untouched.
        if ref_frames > 1 and ok_f and len(raw_hist) >= 1:
            # Multi-frame averaged reference (Task 5): co-register the last K frames to
            # frame f via their cumulative chain offset, then average. sqrt(K) lower
            # single-frame reference noise. For K=1 this reduces exactly to prv_db, so
            # ref_frames=1 == the committed baseline. Tests "single-frame reference
            # noise" vs "intra-frame distortion" as the floor source.
            px_f = float(ch["x"][f]); py_f = float(ch["y"][f])
            acc = np.zeros((H, W), np.float32); cnt = 0
            for j in range(1, len(raw_hist) + 1):
                fk = f - j
                if fk < 0 or not bool(ch["ok"][fk]):
                    continue
                sxj = -(px_f - float(ch["x"][fk])); syj = -(py_f - float(ch["y"][fk]))
                Mj = np.float32([[1, 0, sxj], [0, 1, syj]])
                acc += cv2.warpAffine(raw_hist[-j], Mj, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                cnt += 1
            ref_db = data._deband(acc / max(cnt, 1))[CROPV:H - CROPV] if cnt else prv_db
            prvz_h = khz2d._zcols(ref_db)
        elif dewarp_atlas and ok_f:
            vx = 0.5 * (float(ch["inc_x"][f - 1]) + float(ch["inc_x"][f]))
            vy = 0.5 * (float(ch["inc_y"][f - 1]) + float(ch["inc_y"][f]))
            prvz_h = khz2d._zcols(_atlas_dewarp_cols(prv_db, vx, vy, order=dewarp_order,
                                                     sign=dewarp_sign))
        else:
            prvz_h = prvz
        P = (prvz_h.T @ curalz) / Hc
        di = np.arange(-PADH, PADH + 1)
        src = np.arange(W)[None, :] + di[:, None]
        bad = (src < 0) | (src >= W)
        src = np.clip(src, 0, W - 1)
        prof = P[src, np.arange(W)[None, :]]
        prof[bad] = -1.0
        prof = prof.T
        pk2 = np.argmax(prof, 1)
        rdx = np.empty(W, np.float32); qh = np.empty(W, np.float32)
        for c in range(W):
            k = int(pk2[c])
            rdx[c] = khz2d._parab(prof[c], k) - PADH
            qh[c] = prof[c, k]
        con = raw[CROPV:H - CROPV].std(0)

        tline = f / fps + (np.arange(W) + 0.5) / W / fps
        T.append(tline.astype(np.float64))
        FR.append(np.full(W, f, np.int32)); COL.append(np.arange(W, dtype=np.int16))
        LAMV.append(lam); RDX.append(rdx); QV.append(qv); QH.append(qh)
        CON.append(con.astype(np.float32))
        OK.append(np.full(W, ok_f))
        PROF.append(prof.astype(np.float16))
        prev_raw = raw
        raw_hist.append(raw)
        if f % 100 == 0:
            print(f"  [{sub.name} lines] frame {f}  qh_med={np.median(qh):.2f}")
    out = dict(
        t=np.concatenate(T), frame=np.concatenate(FR), col=np.concatenate(COL),
        lam_v=np.concatenate(LAMV), rdx=np.concatenate(RDX),
        qv=np.concatenate(QV), qh=np.concatenate(QH), con=np.concatenate(CON),
        ok=np.concatenate(OK), prof=np.concatenate(PROF, 0),
        chain_t=ch["t"][:nf], chain_x=ch["x"][:nf], chain_y=ch["y"][:nf],
        chain_ok=ch["ok"][:nf],
        fps=np.float64(fps), line_rate=np.float64(fps * 808), padh=np.int64(PADH),
        dewarp_atlas=np.bool_(dewarp_atlas), dewarp_order=np.int64(dewarp_order),
        dewarp_sign=np.float64(dewarp_sign), ref_frames=np.int64(ref_frames),
    )
    np.savez(cache, **out)
    return out


def fov_mask(lm: dict) -> np.ndarray:
    return (lm["ok"].astype(bool)
            & (lm["qh"] > Q_FOV)
            & (lm["con"] > CONTRAST_FRAC * np.median(lm["con"])))


def compute_refs(sub: Subject, lm: dict) -> dict:
    st = data.load_stimulus(STIMULUS)
    dot_t = st.t_s.astype(float)
    dot_x = st.x_deg * 60.0; dot_y = st.y_deg * 60.0
    tr = load_tracker(sub)
    trk_t = (tr.t_s - tr.t_s[0]).astype(float)
    trk_x = khz2d.fill_nan(tr.right_x) * TRK_DEG_X * 60.0
    trk_y = khz2d.fill_nan(tr.right_y) * TRK_DEG_Y * 60.0

    pos = lm["chain_x"][lm["frame"]] + lm["rdx"]
    fs = float(lm["line_rate"])
    m = fov_mask(lm)
    sig = gaussian_filter1d(median_filter(khz2d.fill_nan(np.where(m, pos, np.nan)), 7),
                            fs * 0.01)
    sig = khz2d._drift_removed(np.where(m, sig, np.nan), fs)
    tt = lm["t"]; mm = m
    best = (0.0, 0.0)
    for o in np.arange(-1.0, 6.0, 0.025):
        c = khz2d.corr(sig[mm], np.interp(tt[mm], dot_t + o, dot_x))
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (float(o), float(c))
    return dict(dot_t=dot_t, dot_x=dot_x, dot_y=dot_y,
                trk_t=trk_t, trk_x=trk_x, trk_y=trk_y, off=best[0], off_r=best[1])


def evaluate(t, x_px, y_px, valid, rate, refs: dict, smooth_ms: float = 2.0) -> dict:
    t = np.asarray(t, float)
    vx = np.where(valid, np.asarray(x_px, float), np.nan)
    vy = np.where(valid, np.asarray(y_px, float), np.nan)
    if smooth_ms > 0 and rate > 0:
        k = max(1.0, rate * smooth_ms / 1000.0)
        vx = gaussian_filter1d(khz2d.fill_nan(vx), k); vx[~valid] = np.nan
        vy = gaussian_filter1d(khz2d.fill_nan(vy), k); vy[~valid] = np.nan
    hx = khz2d._drift_removed(vx, rate); hy = khz2d._drift_removed(vy, rate)
    dot_x = np.interp(t, refs["dot_t"] + refs["off"], refs["dot_x"])
    dot_y = np.interp(t, refs["dot_t"] + refs["off"], refs["dot_y"])
    trk_x = np.interp(t, refs["trk_t"] + refs["off"], refs["trk_x"])
    trk_y = np.interp(t, refs["trk_t"] + refs["off"], refs["trk_y"])

    out = dict(rate=float(rate), valid_frac=float(np.mean(valid)))
    for ax, h, d, tk in (("x", hx, dot_x, trk_x), ("y", hy, dot_y, trk_y)):
        m = valid & np.isfinite(h)
        r_dot = khz2d.corr(h[m], d[m]); r_trk = khz2d.corr(h[m], tk[m])
        if m.sum() > 20 and np.nanstd(h[m]) > 0:
            A = np.c_[h[m], np.ones(m.sum())]
            coef, *_ = np.linalg.lstsq(A, d[m], rcond=None)
            cal = coef[0] * h + coef[1]
            xf = khz2d.fill_nan(h)
            hf = xf - gaussian_filter1d(xf, max(1.0, rate * 25 / 1000.0))
            prec = float(np.sqrt(np.nanmean(hf[m] ** 2)))
            rms = float(np.sqrt(np.nanmean((cal[m] - d[m]) ** 2)))
        else:
            cal = h * np.nan; rms = np.nan; prec = np.nan
        out[f"r_dot_{ax}"] = abs(r_dot) if np.isfinite(r_dot) else np.nan
        out[f"r_trk_{ax}"] = abs(r_trk) if np.isfinite(r_trk) else np.nan
        out[f"rms_{ax}"] = rms
        out[f"prec_{ax}"] = prec
        out[f"cal_{ax}"] = cal
        out[f"dot_{ax}"] = d
    return out


def _pfkw_from_kwargs(**kw) -> dict:
    """Map run_m4 keyword args to ParticleFilter kwargs."""
    import khz2d_methods as M
    return M._m4_pfkw(
        hp_sigma=kw.get("hp_sigma"),
        sigma_along=kw.get("sigma_along"),
        ess_frac=kw.get("ess_frac"),
        roughen_perp=kw.get("roughen_perp"),
        roughen_along=kw.get("roughen_along"),
        ncc_loss_thr=kw.get("ncc_loss_thr"),
        ncc_loss_window=kw.get("ncc_loss_window"),
        reseed_perp_sigma=kw.get("reseed_perp_sigma", 30.0),
        couple_sigma=kw.get("couple_sigma"),
        resample_enabled=kw.get("resample_enabled", True),
        lock_gated_gain=kw.get("lock_gated_gain", False),
        beta_cool=kw.get("beta_cool"),
        roughen_perp_cool=kw.get("roughen_perp_cool"),
        roughen_along_cool=kw.get("roughen_along_cool"),
        beta_hot=kw.get("beta_hot"),
        roughen_perp_hot=kw.get("roughen_perp_hot"),
        roughen_along_hot=kw.get("roughen_along_hot"),
        quality_scaled_along=kw.get("quality_scaled_along", False),
        along_sigma_max=kw.get("along_sigma_max"),
        along_quality_model=kw.get("along_quality_model"),
        multi_hypothesis=kw.get("multi_hypothesis", False),
        hypothesis_top_k=kw.get("hypothesis_top_k"),
        hypothesis_cluster_rows=kw.get("hypothesis_cluster_rows"),
    )


def run_m4(
    sub: Subject,
    lm: dict,
    ch: dict,
    *,
    cache_path: str | None = None,
    dur_s: float | None = None,
    rebuild: bool = False,
    n_particles: int = 300,
    padw: int = 100,
    line_len: int = 200,
    seed: int = 0,
    beta: float | None = None,
    likelihood: str = "physics",
    frame_reader: Callable[[], Iterator[tuple[int, np.ndarray]]] | None = None,
    line_keep: Callable[[int], bool] | None = None,
    mosaic_reject: bool = False,
    mosaic_window_rows: float = 3.0,
    mosaic_prior: bool = False,
    mosaic_prior_sigma_rows: float = 3.0,
    mosaic_prior_reseed_hold: int = 8,
    mosaic_prior_ncc_thr: float = flt.LOCK_NCC_THR,
    mosaic_prior_track_tau_s: float = flt.MOSAIC_PRIOR_TRACK_TAU_S,
    **pf_kw,
) -> dict:
    """Run physics M4 PF on a people_fov capture; save to cache_path or default."""
    out_path = cache_path or os.path.join(sub.cache_dir, "m4_dpf_physics.npz")
    if os.path.exists(out_path) and not rebuild:
        z = np.load(out_path)
        return {k: z[k] for k in z.files}

    rate = float(lm["line_rate"])
    block = 1
    eff = rate / block
    con_med = np.median(lm["con"])
    T_, X_, Y_, V_, NCC_, ESS_, PSACC_ = [], [], [], [], [], [], []
    RES_, RSD_ = [], []          # per-emitted-line resampled / reseeded flags
    AQ_, ASE_, HC_, QV_, QH_, CON_ = [], [], [], [], [], []
    XRAW_, YRAW_, RSLV_ = [], [], []
    RHIDX_, RHRANK_, RHGAP_, RHMARGIN_ = [], [], [], []
    XBEST_, YBEST_, XPATH_, YPATH_, RBLEND_ = [], [], [], [], []
    lag_ms = float(pf_kw.get("lag_ms", flt.HYPOTHESIS_LAG_MS))
    hyp_transition_sigma = float(
        pf_kw.get("hypothesis_transition_sigma_rows",
                  flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS))
    hyp_obs_weight = float(pf_kw.get("hypothesis_obs_weight",
                                     flt.HYPOTHESIS_OBS_WEIGHT))
    hyp_velocity_cost = float(pf_kw.get("hypothesis_velocity_cost",
                                        flt.HYPOTHESIS_VEL_COST))
    hyp_velocity_sigma = float(pf_kw.get("hypothesis_velocity_sigma_deg_s",
                                         flt.HYPOTHESIS_VEL_SIGMA_DEG_S))
    hyp_accel_cost = float(pf_kw.get("hypothesis_acceleration_cost",
                                     flt.HYPOTHESIS_ACCEL_COST))
    hyp_accel_sigma = float(pf_kw.get("hypothesis_acceleration_sigma_deg_s2",
                                      flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2))
    slew_gate = bool(pf_kw.get("slew_gate", False))
    slew_max_deg_s = float(pf_kw.get("slew_max_deg_s", flt.SLEW_GATE_MAX_DEG_S))
    slew_gate_cost = float(pf_kw.get("slew_gate_cost", flt.SLEW_GATE_COST))
    slew_gate_saccade_p = float(
        pf_kw.get("slew_gate_saccade_p", flt.SLEW_GATE_SACCADE_P))
    hyp_blend_immediate = bool(
        pf_kw.get("hypothesis_blend_immediate", flt.HYPOTHESIS_BLEND_IMMEDIATE))
    hyp_blend_delta_rows = float(
        pf_kw.get("hypothesis_blend_delta_rows", flt.HYPOTHESIS_BLEND_DELTA_ROWS))
    hyp_blend_alpha = float(
        pf_kw.get("hypothesis_blend_alpha", flt.HYPOTHESIS_BLEND_ALPHA))
    hyp_blend_saccade_p = float(
        pf_kw.get("hypothesis_blend_saccade_p", flt.HYPOTHESIS_BLEND_SACCADE_P))
    dt_pending = 0               # line periods skipped since last PF step (split-half)
    rng = np.random.default_rng(seed)
    pf = None
    resolver: flt.FixedLagHypothesisResolver | None = None
    resolve_out_idx: list[int] = []
    resolve_chain_x: list[float] = []
    resolve_chain_y: list[float] = []
    prev_db = None
    H = None
    t0 = time.time()
    nfr = len(ch["t"])
    fps = float(ch["fps"])
    _mosaic_lock_override = pf_kw.pop("_mosaic_lock_override", False)
    _pfkw = _pfkw_from_kwargs(**pf_kw)
    use_lag_resolver = bool(_pfkw.get("multi_hypothesis", False))

    def _apply_resolved(est: flt.FixedLagEstimate) -> None:
        out_i = resolve_out_idx[est.index]
        X_[out_i] = resolve_chain_x[est.index] + (est.est_perp - padw)
        Y_[out_i] = resolve_chain_y[est.index] + est.est_along
        RSLV_[out_i] = True
        RHIDX_[out_i] = est.hyp_index
        RHRANK_[out_i] = est.hyp_rank
        RHGAP_[out_i] = est.hyp_logp_gap
        RHMARGIN_[out_i] = est.hyp_logp_margin
        XBEST_[out_i] = resolve_chain_x[est.index] + (est.local_best_perp - padw)
        YBEST_[out_i] = resolve_chain_y[est.index] + est.local_best_along
        XPATH_[out_i] = resolve_chain_x[est.index] + (est.path_perp - padw)
        YPATH_[out_i] = resolve_chain_y[est.index] + est.path_along
        RBLEND_[out_i] = est.blended_immediate

    if mosaic_reject:                              # default OFF -> _pfkw unchanged (byte-identical)
        _pfkw["mosaic_reject"] = True
        _pfkw["mosaic_window_rows"] = float(mosaic_window_rows)
        if _mosaic_lock_override:                  # diagnostic: force always-on, ignore lock gate
            _pfkw["lock_ncc_thr"] = 0.0
            _pfkw["lock_p_pursuit_thr"] = 0.0
    if mosaic_prior:                               # default OFF -> _pfkw unchanged (byte-identical)
        _pfkw["mosaic_prior"] = True
        _pfkw["mosaic_prior_sigma_rows"] = float(mosaic_prior_sigma_rows)
        _pfkw["mosaic_prior_reseed_hold"] = int(mosaic_prior_reseed_hold)
        _pfkw["mosaic_prior_ncc_thr"] = float(mosaic_prior_ncc_thr)
        _pfkw["mosaic_prior_track_tau_s"] = float(mosaic_prior_track_tau_s)
    read = frame_reader or (lambda: _read_frames(sub))
    for f, raw in read():
        if f >= nfr:
            break
        if dur_s is not None and f / fps > dur_s:
            break
        g = data._deband(raw)
        if H is None:
            H = g.shape[0]
        gc = g[CROPV:H - CROPV]
        if prev_db is None:
            prev_db = gc; continue
        ok_f = bool(ch["ok"][f])
        sx = -float(ch["inc_x"][f]); sy = -float(ch["inc_y"][f])
        if ok_f:
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            prevw = cv2.warpAffine(prev_db, M, (prev_db.shape[1], prev_db.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        else:
            prevw = prev_db
        Hc, W = gc.shape
        col_step = Hc / float(line_len)
        atlasT = prevw.T
        if pf is not None and ok_f:
            pf.state.pos_perp = pf.state.pos_perp - float(ch["inc_x"][f])
            pf.state.pos_along = pf.state.pos_along - float(ch["inc_y"][f])
        i0 = (f - 1) * 808
        nb = W // block
        for b in range(nb):
            c0 = b * block; c1 = c0 + block
            cmid = (c0 + c1 - 1) / 2.0
            sl = slice(i0 + c0, i0 + c1)
            gi = i0 + c0
            if line_keep is not None and not line_keep(gi):
                dt_pending += block      # line not in this half: advance the clock only
                continue
            qv = lm["qv"][sl]; qh = lm["qh"][sl]; lam = lm["lam_v"][sl]
            qv_med = float(np.median(qv))
            qh_med = float(np.median(qh))
            con = float(lm["con"][sl].mean())
            tline = float(lm["t"][sl][len(lam) // 2])
            good = ok_f and con > CONTRAST_FRAC * con_med and qv_med > 0.1
            if not good or pf is None:
                if pf is None and good:
                    d0 = float(np.median(lm["rdx"][sl]))
                    a0 = float(np.median(lam))
                    st = flt.init_filter(n_particles, padw + d0, a0, 15.0, 4.0, rng=rng)
                    win = atlasT[max(0, int(cmid) - padw): int(cmid) + padw + 1]
                    pf = flt.ParticleFilter(
                        st, win, line_len, col_step=col_step,
                        beta=(flt.BETA if beta is None else float(beta)),
                        likelihood=likelihood, **_pfkw)
                    if use_lag_resolver and resolver is None:
                        resolver = flt.FixedLagHypothesisResolver(
                            eff, lag_ms=lag_ms,
                            transition_sigma_rows=hyp_transition_sigma,
                            obs_weight=hyp_obs_weight,
                            velocity_cost=hyp_velocity_cost,
                            velocity_sigma_deg_s=hyp_velocity_sigma,
                            acceleration_cost=hyp_accel_cost,
                            acceleration_sigma_deg_s2=hyp_accel_sigma,
                            slew_gate=slew_gate,
                            slew_max_deg_s=slew_max_deg_s,
                            slew_gate_cost=slew_gate_cost,
                            slew_gate_saccade_p=slew_gate_saccade_p,
                            blend_immediate=hyp_blend_immediate,
                            blend_delta_rows=hyp_blend_delta_rows,
                            blend_alpha=hyp_blend_alpha,
                            blend_saccade_p=hyp_blend_saccade_p)
                    dt_pending = 0       # init line is the t0 reference
                if line_keep is None:
                    T_.append(tline); X_.append(np.nan); Y_.append(np.nan)
                    XRAW_.append(np.nan); YRAW_.append(np.nan); RSLV_.append(False)
                    RHIDX_.append(-1); RHRANK_.append(-1); RHGAP_.append(np.nan)
                    RHMARGIN_.append(np.nan)
                    XBEST_.append(np.nan); YBEST_.append(np.nan)
                    XPATH_.append(np.nan); YPATH_.append(np.nan); RBLEND_.append(False)
                    V_.append(False); NCC_.append(0.0)
                    ESS_.append(0.0); PSACC_.append(0.0)
                    RES_.append(False); RSD_.append(False)
                    AQ_.append(qv_med); ASE_.append(np.nan); HC_.append(0)
                    QV_.append(qv_med); QH_.append(qh_med); CON_.append(con)
                else:
                    dt_pending += block  # bad-but-kept line: advance the clock only
                continue
            line = gc[:, c0:c1].mean(1)
            obs = np.interp(np.arange(line_len) * col_step, np.arange(Hc), line)
            lo = int(cmid) - padw
            win = atlasT[max(0, lo): lo + 2 * padw + 1]
            if win.shape[0] < 2 * padw + 1:
                padrows = np.zeros((2 * padw + 1 - win.shape[0], Hc), np.float32)
                win = np.vstack([padrows, win] if lo < 0 else [win, padrows])
            pf.atlas = win.astype(np.float64)
            along_meas = float(np.median(lam))
            dt = (dt_pending + block) / rate
            post = pf.step(obs, along_meas, dt, rng, coarse_anchor=float(padw),
                           along_quality=qv_med)
            dt_pending = 0
            d_est = post.est_perp - padw
            raw_x = float(ch["x"][f]) + d_est
            raw_y = float(ch["y"][f]) + post.est_along
            out_i = len(T_)
            T_.append(tline)
            X_.append(raw_x)
            Y_.append(raw_y)
            XRAW_.append(raw_x); YRAW_.append(raw_y); RSLV_.append(False)
            RHIDX_.append(-1); RHRANK_.append(-1); RHGAP_.append(np.nan)
            RHMARGIN_.append(np.nan)
            XBEST_.append(np.nan); YBEST_.append(np.nan)
            XPATH_.append(np.nan); YPATH_.append(np.nan); RBLEND_.append(False)
            V_.append(post.max_ncc > 0.12)
            NCC_.append(post.max_ncc)
            ESS_.append(post.ess)
            PSACC_.append(post.mode_posterior[1])
            RES_.append(bool(post.resampled)); RSD_.append(bool(post.reseeded))
            AQ_.append(qv_med); ASE_.append(post.along_sigma_eff); HC_.append(len(post.hyp_perp))
            QV_.append(qv_med); QH_.append(qh_med); CON_.append(con)
            if resolver is not None:
                resolve_out_idx.append(out_i)
                resolve_chain_x.append(float(ch["x"][f]))
                resolve_chain_y.append(float(ch["y"][f]))
                est = resolver.push(post, dt_s=dt)
                if est is not None:
                    _apply_resolved(est)
        prev_db = gc
        if f % 100 == 0:
            print(f"  [{sub.name} m4] frame {f}/{nfr} ({time.time()-t0:.0f}s)")
    if resolver is not None:
        for est in resolver.flush():
            _apply_resolved(est)
    T_ = np.array(T_); X_ = np.array(X_); Y_ = np.array(Y_)
    V_ = np.array(V_) & np.isfinite(X_) & np.isfinite(Y_)
    out = dict(t=T_, x_px=X_, y_px=Y_, valid=V_, rate=np.float64(eff),
               max_ncc=np.array(NCC_), ess=np.array(ESS_), p_saccade=np.array(PSACC_),
               resampled=np.array(RES_, dtype=bool), reseeded=np.array(RSD_, dtype=bool),
               along_quality=np.array(AQ_, dtype=np.float64),
               along_sigma_eff=np.array(ASE_, dtype=np.float64),
               hyp_count=np.array(HC_, dtype=np.int16),
               qv=np.array(QV_, dtype=np.float64),
               qh=np.array(QH_, dtype=np.float64),
               con=np.array(CON_, dtype=np.float64),
               x_px_immediate=np.array(XRAW_, dtype=np.float64),
               y_px_immediate=np.array(YRAW_, dtype=np.float64),
               fixed_lag_resolved=np.array(RSLV_, dtype=bool),
               fixed_lag_hyp_index=np.array(RHIDX_, dtype=np.int16),
               fixed_lag_hyp_rank=np.array(RHRANK_, dtype=np.int16),
               fixed_lag_hyp_logp_gap=np.array(RHGAP_, dtype=np.float64),
               fixed_lag_hyp_logp_margin=np.array(RHMARGIN_, dtype=np.float64),
               fixed_lag_local_best_x_px=np.array(XBEST_, dtype=np.float64),
               fixed_lag_local_best_y_px=np.array(YBEST_, dtype=np.float64),
               fixed_lag_path_x_px=np.array(XPATH_, dtype=np.float64),
               fixed_lag_path_y_px=np.array(YPATH_, dtype=np.float64),
               fixed_lag_blended_immediate=np.array(RBLEND_, dtype=bool),
               fixed_lag_lines=np.int64(resolver.lag if resolver is not None else 0),
               fixed_lag_ms=np.float64(lag_ms if resolver is not None else 0.0),
               slew_gate=np.bool_(slew_gate if resolver is not None else False),
               slew_max_deg_s=np.float64(slew_max_deg_s if resolver is not None else 0.0),
               hypothesis_blend_immediate=np.bool_(hyp_blend_immediate if resolver is not None else False),
               hypothesis_blend_delta_rows=np.float64(hyp_blend_delta_rows if resolver is not None else 0.0),
               hypothesis_blend_alpha=np.float64(hyp_blend_alpha if resolver is not None else 0.0),
               hypothesis_blend_saccade_p=np.float64(hyp_blend_saccade_p if resolver is not None else 0.0),
               motion_prior=np.bool_(mosaic_prior),
               motion_prior_sigma_rows=np.float64(mosaic_prior_sigma_rows if mosaic_prior else 0.0),
               motion_prior_ncc_thr=np.float64(mosaic_prior_ncc_thr if mosaic_prior else 0.0),
               motion_prior_tau_s=np.float64(mosaic_prior_track_tau_s if mosaic_prior else 0.0),
               hypothesis_velocity_cost=np.float64(hyp_velocity_cost if resolver is not None else 0.0),
               hypothesis_velocity_sigma_deg_s=np.float64(hyp_velocity_sigma if resolver is not None else 0.0),
               hypothesis_acceleration_cost=np.float64(hyp_accel_cost if resolver is not None else 0.0),
               hypothesis_acceleration_sigma_deg_s2=np.float64(hyp_accel_sigma if resolver is not None else 0.0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    return out


def hf_precision_arcmin(x_arcmin: np.ndarray, rate: float, valid: np.ndarray,
                        ms: float = 25.0) -> float:
    x = np.asarray(x_arcmin, float)
    m = valid.astype(bool) & np.isfinite(x)
    if m.sum() < 100:
        return float("nan")
    xf = khz2d.fill_nan(x)
    k = max(1.0, rate * ms / 1000.0)
    hf = xf - gaussian_filter1d(xf, k)
    return float(np.sqrt(np.nanmean(hf[m] ** 2)))


def frame_jitter_30fps(x_arcmin: np.ndarray, rate: float, valid: np.ndarray,
                       smooth_ms: float = 2.0) -> float:
    x = np.asarray(x_arcmin, float)
    m = valid.astype(bool) & np.isfinite(x)
    xf = gaussian_filter1d(khz2d.fill_nan(x), max(1.0, rate * smooth_ms / 1000.0))
    step = max(1, int(round(rate / 30.0)))
    s30 = xf[::step]
    m30 = m[::step]
    s30 = s30[m30]
    if len(s30) < 3:
        return float("nan")
    return float(np.median(np.abs(np.diff(s30, 2))))


def psd_band_ratio(x: np.ndarray, rate: float, valid: np.ndarray,
                   low_hz: float = 0.0, high_hz: float = 2.0,
                   hf_hz: float = 70.0) -> tuple[float, float]:
    """Return (pursuit_power, hf_power) from Welch PSD on valid samples."""
    from scipy import signal as sig
    x = np.asarray(x, float)
    m = valid.astype(bool) & np.isfinite(x)
    if m.sum() < rate * 2:
        return float("nan"), float("nan")
    xf = khz2d.fill_nan(x)[m]
    nperseg = min(len(xf), max(256, int(rate * 2)))
    f, p = sig.welch(xf, fs=rate, nperseg=nperseg)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        from scipy.integrate import trapezoid as _trapz
    m_low = (f >= low_hz) & (f <= high_hz)
    m_hf = f >= hf_hz
    p_low = float(_trapz(p[m_low], f[m_low]))
    p_hf = float(_trapz(p[m_hf], f[m_hf]))
    return p_low, p_hf
