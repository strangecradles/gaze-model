"""oculo_smooth.py — event-preserving gaze smoothing for display.

Goal: one trajectory that looks smooth during pursuit/fixation but still shows
real saccades and (when above the noise floor) ocular tremor. This is NOT a
slow+HF decomposition for storage — it produces a single best-estimate path.

Method
------
1. Repair short mosaic-alias excursions: a discontinuous jump that returns to
   the prior track within ~1 ms is held on the causal baseline. Smooth ramps are
   left untouched.
2. Detect saccade intervals from velocity (MAD threshold, same spirit as
   khz2d_report.saccade_stats), padded so the ballistic shape is preserved.
3. On non-saccade samples, low-pass heavily (~PURSUIT_LP_HZ) to remove PF
   line-rate estimation noise.
4. On non-saccade samples, optionally add back a tremor band (70–120 Hz) only
   where its amplitude exceeds a fixation-period noise floor (~2× MAD).
5. On saccade samples, keep the lightly smoothed raw trace (minimal filtering).

Raw line-rate PF output should stay in cache untouched; apply this only for
display or a derived ``traj_*`` product.

Usage::
    from oculo_smooth import oculomotor_trajectory
    x_disp = oculomotor_trajectory(x_arcmin, rate, valid)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_dilation

PURSUIT_LP_HZ = 25.0       # LP during fixation/pursuit (kills PF line noise)
VEL_SMOOTH_MS = 5.0         # light event smoothing; avoids one-line alias edges as "saccades"
SACC_K_MAD = 8.0            # velocity threshold = k * MAD (raised for noisy kHz traces)
SACC_MIN_MS = 1.5           # min saccade run length
SACC_PAD_MS = 12.0          # pad saccade mask so onset/offset aren't clipped
SACC_MIN_AMP = 2.0          # arcmin — run must displace at least this much (khz2d_report)
TREMOR_LO_HZ = 70.0
TREMOR_HI_HZ = 120.0
TREMOR_SNR = 2.0            # add tremor only if |amp| > SNR * fixation noise floor
ALIAS_JUMP_ARCMIN = 1.5     # ~3 px in people_fov: too fast for a microsaccade onset
ALIAS_RETURN_ARCMIN = 0.75  # must return close to the causal baseline
ALIAS_MAX_MS = 1.0          # short jump-return flips; longer plateaus are not "repaired"


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    i = np.arange(len(x))
    m = np.isfinite(x)
    if not m.any():
        return np.zeros_like(x)
    return np.interp(i, i[m], x[m])


def _runs_with_amp(
    mask: np.ndarray, x: np.ndarray, min_len: int, min_amp: float,
) -> np.ndarray:
    """True on velocity runs that are long enough AND displace >= min_amp."""
    out = np.zeros_like(mask, dtype=bool)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len and abs(x[min(j, n - 1)] - x[i]) >= min_amp:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _dilate_ms(mask: np.ndarray, pad_ms: float, rate: float) -> np.ndarray:
    pad = max(1, int(round(rate * pad_ms / 1000.0)))
    if pad <= 0:
        return mask
    struct = np.ones(2 * pad + 1, dtype=bool)
    return binary_dilation(mask, structure=struct)


def repair_alias_excursions(
    x: np.ndarray,
    rate: float,
    valid: np.ndarray | None = None,
    *,
    jump_arcmin: float = ALIAS_JUMP_ARCMIN,
    return_arcmin: float = ALIAS_RETURN_ARCMIN,
    max_ms: float = ALIAS_MAX_MS,
    event_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Suppress short discontinuous jump-return mosaic aliases.

    The people-FOV failure mode is a one-mosaic-peak step that enters in one line
    and returns within a handful of lines. Real microsaccades in this data are
    smooth ramps over milliseconds, so they do not trip this discontinuity test.
    ``event_mask`` lets the caller exempt true saccade intervals entirely.
    """
    x = np.asarray(x, float)
    rate = float(rate)
    if valid is None:
        valid = np.isfinite(x)
    else:
        valid = np.asarray(valid, dtype=bool)
    if event_mask is None:
        event_mask = np.zeros_like(valid, dtype=bool)
    else:
        event_mask = np.asarray(event_mask, dtype=bool)

    raw = _fill_nan(x)
    out = raw.copy()
    look = max(2, int(round(rate * max_ms / 1000.0)))
    i, n = 1, len(raw)
    while i < n - 1:
        if not (valid[i] and valid[i - 1]) or event_mask[i] or event_mask[i - 1]:
            i += 1
            continue
        baseline = out[i - 1]
        if abs(raw[i] - baseline) >= jump_arcmin:
            end = min(n - 1, i + look)
            seg = raw[i + 1:end + 1]
            ok = valid[i + 1:end + 1] & ~event_mask[i + 1:end + 1]
            returned = np.where(ok & (np.abs(seg - baseline) <= return_arcmin))[0]
            if returned.size:
                j = i + 1 + int(returned[0])
                # Keep continuity with the returned sample without adding a visible edge.
                out[i:j] = np.linspace(baseline, raw[j], j - i + 1)[:-1]
                i = j
                continue
        i += 1

    out = out.astype(float)
    out[~valid] = np.nan
    return out


def _bandpass(x: np.ndarray, lo_hz: float, hi_hz: float, rate: float) -> np.ndarray:
    k_lo = max(1.0, rate / (2 * np.pi * lo_hz))
    k_hi = max(1.0, rate / (2 * np.pi * hi_hz))
    return gaussian_filter1d(x, k_hi) - gaussian_filter1d(x, k_lo)


def oculomotor_trajectory(
    x: np.ndarray,
    rate: float,
    valid: np.ndarray | None = None,
    *,
    pursuit_lp_hz: float = PURSUIT_LP_HZ,
    vel_smooth_ms: float = VEL_SMOOTH_MS,
    sacc_k_mad: float = SACC_K_MAD,
    sacc_min_ms: float = SACC_MIN_MS,
    sacc_pad_ms: float = SACC_PAD_MS,
    sacc_min_amp: float = SACC_MIN_AMP,
    tremor_lo_hz: float = TREMOR_LO_HZ,
    tremor_hi_hz: float = TREMOR_HI_HZ,
    tremor_snr: float = TREMOR_SNR,
    include_tremor: bool = True,
    repair_alias: bool = True,
    alias_jump_arcmin: float = ALIAS_JUMP_ARCMIN,
    alias_return_arcmin: float = ALIAS_RETURN_ARCMIN,
    alias_max_ms: float = ALIAS_MAX_MS,
) -> np.ndarray:
    """Return a single event-preserving trajectory (same units as ``x``).

    Parameters
    ----------
    x : position trace (e.g. arcmin), may contain NaN outside FOV.
    rate : sample rate (Hz).
    valid : optional boolean mask; invalid samples become NaN in the output.
    """
    x = np.asarray(x, float)
    rate = float(rate)
    if valid is None:
        valid = np.isfinite(x)
    else:
        valid = np.asarray(valid, dtype=bool)

    raw = _fill_nan(x)
    k_vel = max(1.0, rate * vel_smooth_ms / 1000.0)
    x_light = gaussian_filter1d(raw, k_vel)

    v = np.gradient(x_light) * rate
    vm = valid & np.isfinite(v)
    if vm.sum() < 50:
        out = gaussian_filter1d(raw, rate / (2 * np.pi * pursuit_lp_hz))
        out[~valid] = np.nan
        return out

    mad = float(np.median(np.abs(v[vm] - np.median(v[vm])))) * 1.4826 + 1e-9
    thr = sacc_k_mad * mad
    fast = (np.abs(v) > thr) & valid
    min_run = max(2, int(round(rate * sacc_min_ms / 1000.0)))
    sacc = _dilate_ms(
        _runs_with_amp(fast, x_light, min_run, sacc_min_amp), sacc_pad_ms, rate,
    ) & valid

    if repair_alias:
        raw = _fill_nan(repair_alias_excursions(
            x, rate, valid,
            jump_arcmin=alias_jump_arcmin,
            return_arcmin=alias_return_arcmin,
            max_ms=alias_max_ms,
            event_mask=sacc,
        ))
        x_light = gaussian_filter1d(raw, k_vel)

    k_pursuit = max(1.0, rate / (2 * np.pi * pursuit_lp_hz))
    x_pursuit = gaussian_filter1d(raw, k_pursuit)

    out = np.where(sacc, x_light, x_pursuit)

    if include_tremor and rate > 2 * tremor_hi_hz:
        tremor = _bandpass(x_light, tremor_lo_hz, tremor_hi_hz, rate)
        fix = valid & ~sacc
        if fix.sum() > 100:
            floor = float(np.median(np.abs(tremor[fix]))) * 1.4826 + 1e-9
            keep = np.abs(tremor) > tremor_snr * floor
            out = out + np.where(sacc | ~keep, 0.0, tremor)

    out = out.astype(float)
    out[~valid] = np.nan
    return out


def oculomotor_trajectory_2d(
    x: np.ndarray,
    y: np.ndarray,
    rate: float,
    valid: np.ndarray | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent event-preserving smooth on x and y."""
    if valid is None:
        valid = np.isfinite(x) & np.isfinite(y)
    return (
        oculomotor_trajectory(x, rate, valid, **kwargs),
        oculomotor_trajectory(y, rate, valid, **kwargs),
    )
