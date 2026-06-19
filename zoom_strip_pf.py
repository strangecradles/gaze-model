"""Synthetic-reference strip PF tracking for zoom SLO TIFF captures.

This is the paper-style strip path for the zoom/ data: build a synthetic
reference from coarsely registered TIFF frames, use each frame's coarse offset
to choose the corresponding reference ROI for every strip, then feed the whole
NCC response surface into the same IMM particle-filter / fixed-lag resolver used
by people_strip_ladder.py.

The zoom TIFFs do not currently carry a trusted acquisition clock in the repo,
so ``--fps`` is explicit. Absolute speed metrics scale with that value; the
cross-width/split-half repeatability diagnostics do not depend on a dot target.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from typing import Callable

import cv2
import numpy as np
import tifffile

import data
import dynamics
import filter as flt
import khz2d
import people_fov_pf as people_pf
import people_strip_ladder as ladder
import strip_pf_diagnostics as diag

DEFAULT_CAPTURE = os.path.join("zoom", "live200ashton")
DEFAULT_WIDTHS = (64, 32, 16, 15, 8, 4, 2, 1)
DEFAULT_OUT_PREFIX = os.path.join("results", "zoom_strip_pf_live200ashton")
ZOOM_CACHE = os.path.join("cache", "zoom_strip")
ARC = people_pf.ARC_PER_PX


def _tag_float(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def _natural_key(path: str) -> list[object]:
    name = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", name)]


def zoom_tiff_paths(capture_dir: str) -> list[str]:
    paths = (
        glob.glob(os.path.join(capture_dir, "*.tif"))
        + glob.glob(os.path.join(capture_dir, "*.tiff"))
    )
    paths = sorted(set(paths), key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"no TIFF files found under {capture_dir!r}")
    return paths


def capture_tag(capture_dir: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(os.path.normpath(capture_dir)))


def capture_cache_dir(capture_dir: str) -> str:
    path = os.path.join(ZOOM_CACHE, capture_tag(capture_dir))
    os.makedirs(path, exist_ok=True)
    return path


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _prep_frame(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float32)
    x = np.log1p(x - float(np.min(x)))
    x = data._deband(x)
    s = float(np.std(x))
    if s <= 0 or not np.isfinite(s):
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(np.mean(x))) / s).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 1000:
        return float("nan")
    aa = a[m].astype(np.float64)
    bb = b[m].astype(np.float64)
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    d = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / d) if d > 0 else float("nan")


def _sharpness(img: np.ndarray) -> float:
    gy, gx = np.gradient(np.asarray(img, dtype=np.float64))
    return float(np.mean(gx * gx + gy * gy))


def load_zoom_frames(
    capture_dir: str,
    max_frames: int | None = None,
    *,
    rebuild: bool = False,
) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    """Return preprocessed zoom TIFF frames as a disk memmap."""
    paths = zoom_tiff_paths(capture_dir)
    if max_frames is not None:
        paths = paths[:int(max_frames)]
    tag = "all" if max_frames is None else f"n{int(max_frames)}"
    cdir = capture_cache_dir(capture_dir)
    dat = os.path.join(cdir, f"frames_{tag}.dat")
    side = os.path.join(cdir, f"frames_{tag}.npz")
    if os.path.exists(dat) and os.path.exists(side) and not rebuild:
        s = np.load(side)
        shape = tuple(int(x) for x in s["shape"])
        mm = np.memmap(dat, dtype=np.float32, mode="r", shape=shape)
        return mm, s["raw_mean"], s["raw_std"], s["paths"]

    first = tifffile.imread(paths[0])
    h, w = first.shape[:2]
    mm = np.memmap(dat, dtype=np.float32, mode="w+", shape=(len(paths), h, w))
    raw_mean = np.zeros(len(paths), dtype=np.float64)
    raw_std = np.zeros(len(paths), dtype=np.float64)
    for i, path in enumerate(paths):
        raw = tifffile.imread(path)
        if raw.shape[:2] != (h, w):
            raise ValueError(f"{path} has shape {raw.shape}, expected {(h, w)}")
        raw_mean[i] = float(np.mean(raw))
        raw_std[i] = float(np.std(raw))
        mm[i] = _prep_frame(raw)
    mm.flush()
    np.savez(
        side,
        shape=np.asarray(mm.shape, dtype=np.int64),
        raw_mean=raw_mean,
        raw_std=raw_std,
        paths=np.asarray(paths),
    )
    mm = np.memmap(dat, dtype=np.float32, mode="r", shape=(len(paths), h, w))
    return mm, raw_mean, raw_std, np.asarray(paths)


def synthetic_ref_path(
    capture_dir: str,
    max_ref_frames: int,
    passes: int,
    max_shift: float,
    ncc_thr: float,
) -> str:
    tag = (
        f"synthetic_ref_n{int(max_ref_frames)}"
        f"_p{int(passes)}_ms{_tag_float(max_shift)}_q{_tag_float(ncc_thr)}.npz"
    )
    return os.path.join(capture_cache_dir(capture_dir), tag)


def build_synthetic_reference(
    capture_dir: str,
    *,
    max_ref_frames: int = 120,
    passes: int = 2,
    max_shift: float = 80.0,
    ncc_thr: float = 0.25,
    rebuild: bool = False,
    rebuild_frames: bool = False,
) -> dict:
    """Build a coarse registered-and-averaged synthetic zoom reference."""
    out_path = synthetic_ref_path(capture_dir, max_ref_frames, passes, max_shift, ncc_thr)
    if os.path.exists(out_path) and not rebuild:
        return _load_npz(out_path)

    frames, raw_mean, _raw_std, paths = load_zoom_frames(
        capture_dir, max_ref_frames, rebuild=rebuild_frames)
    good = raw_mean > 0.45 * float(np.median(raw_mean))
    good_idx = np.flatnonzero(good)
    if good_idx.size == 0:
        good_idx = np.arange(len(frames))
        good[:] = True
    early = good_idx[:min(60, good_idx.size)]
    ref_idx = int(max(early, key=lambda i: _sharpness(frames[int(i)])))
    ref = np.asarray(frames[ref_idx], dtype=np.float32).copy()
    offsets = np.full((len(frames), 2), np.nan, dtype=np.float64)
    ncc = np.full(len(frames), np.nan, dtype=np.float64)
    keep = np.zeros(len(frames), dtype=bool)
    coverage = np.zeros(ref.shape, dtype=np.float64)

    for _pass in range(max(1, int(passes))):
        acc = np.zeros(ref.shape, dtype=np.float64)
        cov = np.zeros(ref.shape, dtype=np.float64)
        keep[:] = False
        for i in good_idx:
            i = int(i)
            dy, dx = data._phase_corr2d(ref, np.asarray(frames[i]))
            offsets[i] = (dy, dx)
            if abs(dy) > max_shift or abs(dx) > max_shift:
                continue
            aligned = data._shift_im(np.asarray(frames[i]), int(dy), int(dx))
            q = _ncc(ref, aligned)
            ncc[i] = q
            if not np.isfinite(q) or q < float(ncc_thr):
                continue
            m = np.isfinite(aligned)
            acc[m] += aligned[m]
            cov[m] += 1.0
            keep[i] = True
        if not keep.any():
            keep[ref_idx] = True
            acc = np.asarray(frames[ref_idx], dtype=np.float64)
            cov = np.ones(ref.shape, dtype=np.float64)
        ref = np.where(cov > 0, acc / np.maximum(cov, 1.0), 0.0).astype(np.float32)
        ref = ladder._nz(ref)
        coverage = cov

    out = dict(
        ref=ref.astype(np.float32),
        coverage=coverage.astype(np.float32),
        keep=keep,
        offsets=offsets,
        ncc=ncc,
        ref_idx=np.int64(ref_idx),
        paths=paths,
        capture=np.asarray(capture_dir),
        max_ref_frames=np.int64(max_ref_frames),
        passes=np.int64(passes),
        max_shift=np.float64(max_shift),
        ncc_thr=np.float64(ncc_thr),
    )
    np.savez(out_path, **out)
    return out


def zoom_strip_cache_path(
    capture_dir: str,
    strip_width: int,
    n_frames: int | None,
    fps: float,
    *,
    method: str = "pfref",
    suffix: str = "",
) -> str:
    frame_tag = "all" if n_frames is None else f"f{int(n_frames)}"
    tag = (
        f"zoom_strip_{method}_s{int(strip_width)}_{frame_tag}"
        f"_fps{_tag_float(fps)}{suffix}.npz"
    )
    return os.path.join(capture_cache_dir(capture_dir), tag)


def zoom_pf_method_name(
    *,
    n_particles: int = ladder.PF_N_PARTICLES,
    beta: float = ladder.PF_BETA,
    ess_frac: float = ladder.PF_ESS_FRAC,
    roughen_perp: float = ladder.PF_ROUGHEN_PERP,
    roughen_along: float = ladder.PF_ROUGHEN_ALONG,
    init_spread_px: float = ladder.PF_INIT_SPREAD_PX,
    top_k: int = 5,
    lag_ms: float = flt.HYPOTHESIS_LAG_MS,
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
) -> str:
    variant = ladder.pf_variant_name(
        n_particles=n_particles,
        beta=beta,
        ess_frac=ess_frac,
        roughen_perp=roughen_perp,
        roughen_along=roughen_along,
        init_spread_px=init_spread_px,
        top_k=top_k,
        lag_ms=lag_ms,
        transition_sigma_rows=transition_sigma_rows,
        obs_weight=obs_weight,
        velocity_cost=velocity_cost,
        acceleration_cost=acceleration_cost,
    )
    return "pfref_coarse" if variant == "pf" else f"pfref_coarse_{variant}"


def _ref_suffix(max_ref_frames: int, passes: int, max_shift: float, ncc_thr: float) -> str:
    return (
        f"_refn{int(max_ref_frames)}_rp{int(passes)}"
        f"_ms{_tag_float(max_shift)}_rq{_tag_float(ncc_thr)}"
    )


def _zoom_valid(q: np.ndarray, con: np.ndarray, quality_thr: float,
                contrast_frac: float) -> np.ndarray:
    return ladder.strip_valid_mask(q, con, quality_thr, contrast_frac)


def _apply_fixed_lag(
    est: flt.FixedLagEstimate,
    x_out: list[float],
    y_out: list[float],
    resolved: list[bool],
    hidx: list[int],
    hrank: list[int],
    hgap: list[float],
    hmargin: list[float],
) -> None:
    x_out[est.index] = est.est_perp
    y_out[est.index] = est.est_along
    resolved[est.index] = True
    hidx[est.index] = est.hyp_index
    hrank[est.index] = est.hyp_rank
    hgap[est.index] = est.hyp_logp_gap
    hmargin[est.index] = est.hyp_logp_margin


def run_zoom_strip_pf(
    capture_dir: str = DEFAULT_CAPTURE,
    strip_width: int = 15,
    *,
    fps: float = 30.0,
    n_frames: int | None = 80,
    pad: int = 80,
    quality_thr: float = 0.35,
    contrast_frac: float = people_pf.CONTRAST_FRAC,
    ref_data: dict | None = None,
    max_ref_frames: int = 120,
    ref_passes: int = 2,
    ref_max_shift: float = 80.0,
    ref_ncc_thr: float = 0.25,
    rebuild: bool = False,
    rebuild_ref: bool = False,
    rebuild_frames: bool = False,
    cache_path: str | None = None,
    seed: int = 0,
    n_particles: int = ladder.PF_N_PARTICLES,
    init_spread_px: float = ladder.PF_INIT_SPREAD_PX,
    beta: float = ladder.PF_BETA,
    ess_frac: float = ladder.PF_ESS_FRAC,
    roughen_perp: float = ladder.PF_ROUGHEN_PERP,
    roughen_along: float = ladder.PF_ROUGHEN_ALONG,
    top_k: int = 5,
    cluster_rows: float = ladder.PF_HYPOTHESIS_CLUSTER_ROWS,
    lag_ms: float = flt.HYPOTHESIS_LAG_MS,
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
    velocity_sigma_deg_s: float = flt.HYPOTHESIS_VEL_SIGMA_DEG_S,
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
    acceleration_sigma_deg_s2: float = flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
    sample_keep: Callable[[int, int, int], bool] | None = None,
    effective_rate: float | None = None,
) -> dict:
    """Run the zoom synthetic-reference strip response through the IMM PF."""
    pf_method = zoom_pf_method_name(
        n_particles=n_particles,
        beta=beta,
        ess_frac=ess_frac,
        roughen_perp=roughen_perp,
        roughen_along=roughen_along,
        init_spread_px=init_spread_px,
        top_k=top_k,
        lag_ms=lag_ms,
        transition_sigma_rows=transition_sigma_rows,
        obs_weight=obs_weight,
        velocity_cost=velocity_cost,
        acceleration_cost=acceleration_cost,
    )
    ref_suffix = _ref_suffix(max_ref_frames, ref_passes, ref_max_shift, ref_ncc_thr)
    out_path = cache_path or zoom_strip_cache_path(
        capture_dir, strip_width, n_frames, fps, method=pf_method, suffix=ref_suffix)
    if os.path.exists(out_path) and not rebuild:
        return _load_npz(out_path)

    ref_data = ref_data or build_synthetic_reference(
        capture_dir,
        max_ref_frames=max_ref_frames,
        passes=ref_passes,
        max_shift=ref_max_shift,
        ncc_thr=ref_ncc_thr,
        rebuild=rebuild_ref,
        rebuild_frames=rebuild_frames,
    )
    frames, _raw_mean, _raw_std, paths = load_zoom_frames(
        capture_dir, n_frames, rebuild=rebuild_frames)
    ref = np.asarray(ref_data["ref"], dtype=np.float32)
    if frames.shape[1:] != ref.shape:
        raise ValueError(f"frame shape {frames.shape[1:]} != reference shape {ref.shape}")
    n, h, w = frames.shape
    if strip_width > w:
        raise ValueError(f"strip width {strip_width} exceeds frame width {w}")
    nstrip = w // int(strip_width)
    full_rate = float(nstrip * float(fps))
    rate_nominal = float(effective_rate) if effective_rate is not None else full_rate
    coarse = np.full((n, 2), np.nan, dtype=np.float64)
    ref_offsets = np.asarray(ref_data.get("offsets", np.empty((0, 2))), dtype=np.float64)
    for i in range(n):
        if i < len(ref_offsets) and np.isfinite(ref_offsets[i]).all():
            coarse[i] = ref_offsets[i]
        else:
            dy, dx = data._phase_corr2d(ref, np.asarray(frames[i]))
            coarse[i] = (dy, dx)
    finite_coarse = coarse[np.isfinite(coarse).all(axis=1)]
    max_coarse = float(np.nanmax(np.abs(finite_coarse))) if finite_coarse.size else 0.0
    border = int(pad + np.ceil(max_coarse) + 2)
    refp = cv2.copyMakeBorder(ref, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=0)
    rng = np.random.default_rng(seed)
    resolver: flt.FixedLagHypothesisResolver | None = None
    st: dynamics.ParticleState | None = None
    prev_t = float("nan")

    T: list[float] = []
    XRAW: list[float] = []
    YRAW: list[float] = []
    XPF: list[float] = []
    YPF: list[float] = []
    Q: list[float] = []
    CON: list[float] = []
    ESS: list[float] = []
    MAXNCC: list[float] = []
    PSACC: list[float] = []
    RESAMP: list[bool] = []
    RSLV: list[bool] = []
    HCOUNT: list[int] = []
    HIDX: list[int] = []
    HRANK: list[int] = []
    HGAP: list[float] = []
    HMARGIN: list[float] = []

    t0 = time.time()
    for frame_idx in range(n):
        cur = np.asarray(frames[frame_idx], dtype=np.float32)
        keep_idx = [
            s for s in range(nstrip)
            if sample_keep is None or bool(sample_keep(frame_idx * nstrip + s, frame_idx, s))
        ]
        if not keep_idx:
            continue
        rate = float(effective_rate) if effective_rate is not None else (
            full_rate * len(keep_idx) / max(nstrip, 1)
        )
        if resolver is None:
            resolver = flt.FixedLagHypothesisResolver(
                rate,
                lag_ms=lag_ms,
                transition_sigma_rows=transition_sigma_rows,
                obs_weight=obs_weight,
                velocity_cost=velocity_cost,
                velocity_sigma_deg_s=velocity_sigma_deg_s,
                acceleration_cost=acceleration_cost,
                acceleration_sigma_deg_s2=acceleration_sigma_deg_s2,
            )
        for s in keep_idx:
            col0 = s * int(strip_width)
            strip = cur[:, col0:col0 + int(strip_width)]
            coarse_y, coarse_x = coarse[frame_idx]
            top = int(round(border + coarse_y - pad))
            left = int(round(border + col0 + coarse_x - pad))
            region = refp[top:top + h + 2 * pad,
                          left:left + int(strip_width) + 2 * pad]
            if region.shape != (h + 2 * pad, int(strip_width) + 2 * pad):
                continue
            r = cv2.matchTemplate(region, strip, cv2.TM_CCOEFF_NORMED).astype(np.float64)
            _, mx, _, loc = cv2.minMaxLoc(r.astype(np.float32))
            yy = khz2d._parab(r[:, loc[0]], int(loc[1]))
            xx = khz2d._parab(r[loc[1], :], int(loc[0]))
            raw_y = float(coarse_y + (yy - pad))
            raw_x = float(coarse_x + (xx - pad))
            t = float((frame_idx + (col0 + strip_width / 2.0) / w) / float(fps))
            dt = (1.0 / rate) if not np.isfinite(prev_t) else max(t - prev_t, 1e-9)
            if st is None:
                st = flt.init_filter(
                    int(n_particles),
                    raw_x,
                    raw_y,
                    float(init_spread_px),
                    float(init_spread_px),
                    rng=rng,
                )
            else:
                st = dynamics.predict(st, dt, rng)
            resp_x = pad + (st.pos_perp - coarse_x)
            resp_y = pad + (st.pos_along - coarse_y)
            ncc = ladder.sample_response_bilinear(r, resp_y, resp_x, fill=-1.0)
            max_ncc = float(np.nanmax(ncc)) if np.isfinite(ncc).any() else -1.0
            w_obs = np.exp(float(beta) * (ncc - max_ncc))
            weights = st.weight * w_obs
            sw = float(np.sum(weights))
            if not np.isfinite(sw) or sw <= 0.0:
                st = flt.init_filter(
                    int(n_particles),
                    raw_x,
                    raw_y,
                    float(init_spread_px),
                    float(init_spread_px),
                    rng=rng,
                )
                weights = st.weight.copy()
            else:
                weights = weights / sw
                st.weight = weights
            ess = float(1.0 / np.sum(weights ** 2))
            post, masks = ladder._strip_pf_posterior(
                st,
                weights,
                ncc,
                ess=ess,
                resampled=False,
                top_k=top_k,
                cluster_rows=cluster_rows,
            )
            out_i = len(T)
            T.append(t)
            XRAW.append(raw_x)
            YRAW.append(raw_y)
            XPF.append(post.est_perp)
            YPF.append(post.est_along)
            Q.append(float(mx))
            CON.append(float(np.std(strip)))
            ESS.append(ess)
            MAXNCC.append(max_ncc)
            PSACC.append(float(post.mode_posterior[1]))
            RESAMP.append(False)
            RSLV.append(False)
            HCOUNT.append(len(post.hyp_perp))
            HIDX.append(-1)
            HRANK.append(-1)
            HGAP.append(np.nan)
            HMARGIN.append(np.nan)
            est = resolver.push(post, dt_s=dt)
            if est is not None:
                _apply_fixed_lag(est, XPF, YPF, RSLV, HIDX, HRANK, HGAP, HMARGIN)
            if ess < float(ess_frac) * st.n:
                idx = flt._hypothesis_resample(weights, masks, rng, 6)
                st = ladder._roughen_particle_state(
                    st, idx, rng, roughen_perp, roughen_along)
                RESAMP[out_i] = True
            prev_t = t
        if (frame_idx + 1) % 25 == 0:
            print(
                f"  [zoom PF {capture_tag(capture_dir)} S={strip_width}] "
                f"frame {frame_idx + 1}/{n} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    if resolver is not None:
        for est in resolver.flush():
            _apply_fixed_lag(est, XPF, YPF, RSLV, HIDX, HRANK, HGAP, HMARGIN)

    t = np.asarray(T, dtype=np.float64)
    q = np.asarray(Q, dtype=np.float64)
    con = np.asarray(CON, dtype=np.float64)
    if effective_rate is not None:
        rate_out = float(effective_rate)
    elif len(t) > 1 and t[-1] > t[0]:
        rate_out = float((len(t) - 1) / (t[-1] - t[0]))
    else:
        rate_out = rate_nominal
    valid = _zoom_valid(q, con, quality_thr, contrast_frac)
    out = dict(
        t=t,
        x_px=np.asarray(XPF, dtype=np.float64),
        y_px=np.asarray(YPF, dtype=np.float64),
        valid=valid,
        rate=np.float64(rate_out),
        x_px_immediate=np.asarray(XRAW, dtype=np.float64),
        y_px_immediate=np.asarray(YRAW, dtype=np.float64),
        q=q,
        con=con,
        strip_width=np.int64(strip_width),
        pad=np.int64(pad),
        nstrip=np.int64(nstrip),
        frame_cols=np.int64(w),
        frame_rows=np.int64(h),
        coarse_y_px=coarse[:n, 0].astype(np.float64),
        coarse_x_px=coarse[:n, 1].astype(np.float64),
        coarse_border=np.int64(border),
        fps=np.float64(fps),
        n_frames=np.int64(n),
        quality_thr=np.float64(quality_thr),
        contrast_frac=np.float64(contrast_frac),
        n_particles=np.int64(n_particles),
        pf_beta=np.float64(beta),
        pf_ess_frac=np.float64(ess_frac),
        pf_roughen_perp=np.float64(roughen_perp),
        pf_roughen_along=np.float64(roughen_along),
        pf_init_spread_px=np.float64(init_spread_px),
        ess=np.asarray(ESS, dtype=np.float64),
        max_ncc=np.asarray(MAXNCC, dtype=np.float64),
        p_saccade=np.asarray(PSACC, dtype=np.float64),
        resampled=np.asarray(RESAMP, dtype=bool),
        top_k=np.int64(top_k),
        fixed_lag_ms=np.float64(lag_ms),
        fixed_lag_lines=np.int64(resolver.lag if resolver is not None else 0),
        fixed_lag_resolved=np.asarray(RSLV, dtype=bool),
        hyp_count=np.asarray(HCOUNT, dtype=np.int16),
        fixed_lag_hyp_index=np.asarray(HIDX, dtype=np.int16),
        fixed_lag_hyp_rank=np.asarray(HRANK, dtype=np.int16),
        fixed_lag_hyp_logp_gap=np.asarray(HGAP, dtype=np.float64),
        fixed_lag_hyp_logp_margin=np.asarray(HMARGIN, dtype=np.float64),
        hypothesis_transition_sigma_rows=np.float64(transition_sigma_rows),
        hypothesis_obs_weight=np.float64(obs_weight),
        hypothesis_velocity_cost=np.float64(velocity_cost),
        hypothesis_acceleration_cost=np.float64(acceleration_cost),
        capture=np.asarray(capture_dir),
        paths=paths,
        ref_idx=np.int64(ref_data.get("ref_idx", -1)),
        synthetic_ref_keep_frac=np.float64(np.mean(np.asarray(ref_data.get("keep", []), dtype=bool))),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    return out


def immediate_run_from_pf(run: dict) -> dict:
    out = dict(run)
    out["x_px"] = np.asarray(run["x_px_immediate"], dtype=np.float64)
    out["y_px"] = np.asarray(run["y_px_immediate"], dtype=np.float64)
    return out


def _split_keep(mode: str, parity: int, frame_cols: int, strip_width: int):
    nstrip = frame_cols // strip_width

    def keep(global_i: int, _frame_idx: int, _strip_idx: int) -> bool:
        if mode == "evenodd":
            return (global_i % 2) == parity
        if mode == "frameblock":
            return ((global_i // max(nstrip, 1)) % 2) == parity
        raise ValueError(mode)

    return keep


def run_zoom_split(
    capture_dir: str,
    width: int,
    fps: float,
    n_frames: int,
    mode: str,
    parity: int,
    *,
    ref_data: dict,
    rebuild: bool = False,
    **kwargs,
) -> dict:
    frame_cols = int(ref_data["ref"].shape[1])
    full_rate = (frame_cols // int(width)) * float(fps)
    eff_rate = full_rate / 2.0
    pf_method = zoom_pf_method_name(
        n_particles=int(kwargs.get("n_particles", ladder.PF_N_PARTICLES)),
        beta=float(kwargs.get("beta", ladder.PF_BETA)),
        ess_frac=float(kwargs.get("ess_frac", ladder.PF_ESS_FRAC)),
        roughen_perp=float(kwargs.get("roughen_perp", ladder.PF_ROUGHEN_PERP)),
        roughen_along=float(kwargs.get("roughen_along", ladder.PF_ROUGHEN_ALONG)),
        init_spread_px=float(kwargs.get("init_spread_px", ladder.PF_INIT_SPREAD_PX)),
        top_k=int(kwargs.get("top_k", 5)),
        lag_ms=float(kwargs.get("lag_ms", flt.HYPOTHESIS_LAG_MS)),
        transition_sigma_rows=float(kwargs.get(
            "transition_sigma_rows", flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS)),
        obs_weight=float(kwargs.get("obs_weight", flt.HYPOTHESIS_OBS_WEIGHT)),
        velocity_cost=float(kwargs.get("velocity_cost", flt.HYPOTHESIS_VEL_COST)),
        acceleration_cost=float(kwargs.get("acceleration_cost", flt.HYPOTHESIS_ACCEL_COST)),
    )
    max_ref_frames = int(kwargs.get("max_ref_frames", 120))
    ref_passes = int(kwargs.get("ref_passes", 2))
    ref_max_shift = float(kwargs.get("ref_max_shift", 80.0))
    ref_ncc_thr = float(kwargs.get("ref_ncc_thr", 0.25))
    suffix = (
        f"_{mode}{int(parity)}"
        + _ref_suffix(max_ref_frames, ref_passes, ref_max_shift, ref_ncc_thr)
    )
    cache_path = zoom_strip_cache_path(
        capture_dir,
        width,
        n_frames,
        fps,
        method=f"{pf_method}_split",
        suffix=suffix,
    )
    return run_zoom_strip_pf(
        capture_dir,
        width,
        fps=fps,
        n_frames=n_frames,
        ref_data=ref_data,
        rebuild=rebuild,
        cache_path=cache_path,
        sample_keep=_split_keep(mode, parity, frame_cols, width),
        effective_rate=eff_rate,
        **kwargs,
    )


def _format_float(value, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{x:.{digits}f}" if np.isfinite(x) else ""


def write_zoom_outputs(rows: list[dict], out_prefix: str, *, capture: str,
                       fps: float, n_frames: int, ref_data: dict) -> tuple[str, str]:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = out_prefix + ".csv"
    md_path = out_prefix + ".md"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# Zoom Synthetic-Reference Strip PF: {capture_tag(capture)}", ""]
    lines.append(
        "Ground-truth-free diagnostics for strip observations matched to a "
        "coarse registered synthetic reference, then inferred with the IMM PF."
    )
    lines.append(
        f"Assumed frame rate: {fps:g} Hz; frames tracked: {n_frames}; "
        f"reference frame index: {int(ref_data.get('ref_idx', -1))}; "
        f"reference keep fraction: {float(np.mean(np.asarray(ref_data.get('keep', []), dtype=bool))):.3f}."
    )
    slopes = [r for r in rows if r.get("kind") == "rate_slope"]
    if slopes:
        lines += ["", "## Rate Scaling"]
        lines.append("| method | log-log p99.9 speed vs rate slope | n |")
        lines.append("|---|---:|---:|")
        for r in slopes:
            lines.append(
                f"| {r.get('method','')} | "
                f"{_format_float(r.get('loglog_speed_rate_slope'))} | {r.get('n','')} |"
            )
        lines.append("")
        lines.append("Slope near 1 means fixed-size jumps are being differentiated by the sampling rate; flatter is better.")
    pairs = [r for r in rows if r.get("kind") == "pair"]
    if pairs:
        lines += ["", "## Repeatability"]
        lines.append("| label | n | all RMS ' | HF25 RMS ' | slow50 RMS ' | slow corr |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in pairs:
            lines.append(
                f"| {r.get('label','')} | {r.get('n','')} | "
                f"{_format_float(r.get('rms_all_arcmin'))} | "
                f"{_format_float(r.get('rms_hf25_arcmin'))} | "
                f"{_format_float(r.get('rms_slow50_arcmin'))} | "
                f"{_format_float(r.get('corr_slow50'))} |"
            )
    evs = [r for r in rows if r.get("kind") == "evidence"]
    if evs:
        lines += ["", "## Evidence"]
        lines.append("| method | S | rate Hz | valid | max NCC med | ESS frac med | RMS vs immediate ' | jump>=3 | p99.9 speed px/s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in evs:
            lines.append(
                f"| {r.get('method','')} | {r.get('strip_width','')} | "
                f"{_format_float(r.get('rate'), 1)} | "
                f"{_format_float(r.get('valid_frac'))} | "
                f"{_format_float(r.get('max_ncc_med'))} | "
                f"{_format_float(r.get('ess_frac_med'))} | "
                f"{_format_float(r.get('rms_vs_immediate_arcmin'))} | "
                f"{_format_float(r.get('jump_ge3_frac'))} | "
                f"{_format_float(r.get('speed_p999_px_s'), 1)} |"
            )
    lines.append("")
    lines.append("No dot-correlation metric is reported here because these zoom TIFFs are not paired to the people-data pursuit target.")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return csv_path, md_path


def run_zoom_diagnostics(
    *,
    capture_dir: str,
    fps: float,
    n_frames: int,
    widths: list[int],
    run_splits: bool,
    split_widths: list[int],
    split_frames: int,
    rebuild: bool,
    rebuild_ref: bool,
    rebuild_frames: bool,
    out_prefix: str,
    **kwargs,
) -> tuple[list[dict], str, str]:
    ref_cache_kwargs = dict(
        max_ref_frames=int(kwargs.pop("max_ref_frames")),
        ref_passes=int(kwargs.pop("ref_passes")),
        ref_max_shift=float(kwargs.pop("ref_max_shift")),
        ref_ncc_thr=float(kwargs.pop("ref_ncc_thr")),
    )
    ref_data = build_synthetic_reference(
        capture_dir,
        max_ref_frames=ref_cache_kwargs["max_ref_frames"],
        passes=ref_cache_kwargs["ref_passes"],
        max_shift=ref_cache_kwargs["ref_max_shift"],
        ncc_thr=ref_cache_kwargs["ref_ncc_thr"],
        rebuild=rebuild_ref,
        rebuild_frames=rebuild_frames,
    )
    rows: list[dict] = []
    runs_by_method: dict[str, dict[int, dict]] = {"raw_ref": {}, "pf": {}}
    for width in widths:
        run = run_zoom_strip_pf(
            capture_dir,
            width,
            fps=fps,
            n_frames=n_frames,
            ref_data=ref_data,
            rebuild=rebuild,
            rebuild_frames=rebuild_frames,
            **ref_cache_kwargs,
            **kwargs,
        )
        raw_run = immediate_run_from_pf(run)
        runs_by_method["raw_ref"][width] = raw_run
        runs_by_method["pf"][width] = run
        rows.append(diag.evidence_metrics(raw_run, "raw_ref", width))
        rows.append(diag.evidence_metrics(run, "pf", width))
        print(
            f"  [zoom diagnostics] S={width:>2} rate={float(run['rate']):.1f} Hz "
            f"valid={float(np.mean(run['valid'])):.3f} "
            f"jump={diag.step_metrics(run)['jump_ge3_frac']:.3f}",
            flush=True,
        )
    rows.extend(diag.rate_scaling_rows(runs_by_method))
    for method, by_width in runs_by_method.items():
        if 15 in by_width and 1 in by_width:
            rows.append(diag.pair_agreement(by_width[1], by_width[15], f"{method}:S1_vs_S15"))
        if 16 in by_width and 1 in by_width:
            rows.append(diag.pair_agreement(by_width[1], by_width[16], f"{method}:S1_vs_S16"))
        if 2 in by_width and 1 in by_width:
            rows.append(diag.pair_agreement(by_width[1], by_width[2], f"{method}:S1_vs_S2"))

    if run_splits:
        split_kwargs = dict(kwargs)
        split_kwargs.update(ref_cache_kwargs)
        for width in split_widths:
            for mode in ("evenodd", "frameblock"):
                a = run_zoom_split(
                    capture_dir,
                    width,
                    fps,
                    split_frames,
                    mode,
                    0,
                    ref_data=ref_data,
                    rebuild=rebuild,
                    rebuild_frames=rebuild_frames,
                    **split_kwargs,
                )
                b = run_zoom_split(
                    capture_dir,
                    width,
                    fps,
                    split_frames,
                    mode,
                    1,
                    ref_data=ref_data,
                    rebuild=rebuild,
                    rebuild_frames=rebuild_frames,
                    **split_kwargs,
                )
                rows.append(diag.pair_agreement(
                    a, b, f"pf_split:{mode}:S{width}:f{split_frames}"))

    final_prefix = out_prefix
    if not final_prefix.endswith(f"_f{n_frames}"):
        final_prefix += f"_f{n_frames}"
    csv_path, md_path = write_zoom_outputs(
        rows, final_prefix, capture=capture_dir, fps=fps,
        n_frames=n_frames, ref_data=ref_data)
    return rows, csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--dur-frames", type=int, default=80)
    ap.add_argument("--widths", default=",".join(str(w) for w in DEFAULT_WIDTHS))
    ap.add_argument("--ref-frames", type=int, default=120)
    ap.add_argument("--ref-passes", type=int, default=2)
    ap.add_argument("--ref-max-shift", type=float, default=80.0)
    ap.add_argument("--ref-ncc-thr", type=float, default=0.25)
    ap.add_argument("--pad", type=int, default=80)
    ap.add_argument("--quality-thr", type=float, default=0.35)
    ap.add_argument("--contrast-frac", type=float, default=people_pf.CONTRAST_FRAC)
    ap.add_argument("--n-particles", type=int, default=ladder.PF_N_PARTICLES)
    ap.add_argument("--pf-beta", type=float, default=ladder.PF_BETA)
    ap.add_argument("--pf-ess-frac", type=float, default=ladder.PF_ESS_FRAC)
    ap.add_argument("--pf-roughen-perp", type=float, default=ladder.PF_ROUGHEN_PERP)
    ap.add_argument("--pf-roughen-along", type=float, default=ladder.PF_ROUGHEN_ALONG)
    ap.add_argument("--pf-init-spread-px", type=float, default=ladder.PF_INIT_SPREAD_PX)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--lag-ms", type=float, default=flt.HYPOTHESIS_LAG_MS)
    ap.add_argument("--hypothesis-transition-sigma-rows", type=float,
                    default=flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS)
    ap.add_argument("--hypothesis-obs-weight", type=float,
                    default=flt.HYPOTHESIS_OBS_WEIGHT)
    ap.add_argument("--hypothesis-velocity-cost", type=float,
                    default=flt.HYPOTHESIS_VEL_COST)
    ap.add_argument("--hypothesis-acceleration-cost", type=float,
                    default=flt.HYPOTHESIS_ACCEL_COST)
    ap.add_argument("--run-splits", action="store_true")
    ap.add_argument("--split-widths", default="15,1")
    ap.add_argument("--split-frames", type=int, default=40)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rebuild-ref", action="store_true")
    ap.add_argument("--rebuild-frames", action="store_true")
    ap.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    args = ap.parse_args(argv)

    widths = ladder.parse_strip_widths(args.widths)
    split_widths = ladder.parse_strip_widths(args.split_widths)
    _rows, csv_path, md_path = run_zoom_diagnostics(
        capture_dir=args.capture,
        fps=args.fps,
        n_frames=args.dur_frames,
        widths=widths,
        run_splits=args.run_splits,
        split_widths=split_widths,
        split_frames=args.split_frames,
        rebuild=args.rebuild,
        rebuild_ref=args.rebuild_ref,
        rebuild_frames=args.rebuild_frames,
        out_prefix=args.out_prefix,
        pad=args.pad,
        quality_thr=args.quality_thr,
        contrast_frac=args.contrast_frac,
        max_ref_frames=args.ref_frames,
        ref_passes=args.ref_passes,
        ref_max_shift=args.ref_max_shift,
        ref_ncc_thr=args.ref_ncc_thr,
        n_particles=args.n_particles,
        init_spread_px=args.pf_init_spread_px,
        beta=args.pf_beta,
        ess_frac=args.pf_ess_frac,
        roughen_perp=args.pf_roughen_perp,
        roughen_along=args.pf_roughen_along,
        top_k=args.top_k,
        lag_ms=args.lag_ms,
        transition_sigma_rows=args.hypothesis_transition_sigma_rows,
        obs_weight=args.hypothesis_obs_weight,
        velocity_cost=args.hypothesis_velocity_cost,
        acceleration_cost=args.hypothesis_acceleration_cost,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
