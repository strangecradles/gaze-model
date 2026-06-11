"""likelihood.py — multimodal physics observation likelihood (GOALS.md G6).

perp_likelihood(line, atlas, along) -> score_over_rows: the physics atlas-match
score of an observed line against every candidate perp row (along fixed). Nothing
is learned — it is the appearance NCC between the band-filtered observed line and
the band-filtered atlas render at each row.

Band split (PLAN.md "perp fine channel" vs "perp coarse channel"):
  - FINE band (high-pass): razor-sharp peak (sub-0.1 deg when locked) but multimodal
    (many secondary alias peaks) -> the precise-but-aliased channel.
  - COARSE band (low-pass): one broad ~0.5 deg lobe -> the blunt reacquisition anchor.

This is the observation model the differentiable particle filter consumes; the
belief must stay MULTIMODAL over its peaks (never collapse to a unimodal Gaussian).

EMPIRICAL NOTE (see progress.txt G6): on the CLEAN normal/ atlas the fine likelihood
is sharp AND largely unique (the true row dominates; PSR ~2-4). The aliasing grows as
the line carries less along context (short lines -> dense secondary peaks). The literal
"126-row (=1 deg) alias comb" from the sibling is a degraded cross-scale x-scan->atlas
property, NOT reproduced on the clean synthetic substrate; the clean-atlas alias
structure is denser/irregular. ALIAS_SPACING_ROWS (calib, =rows/deg) remains the
geometric scale used by G13's ~820 Hz gate.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import decoder
import calib

ALIAS_SPACING_ROWS = calib.ALIAS_SPACING_ROWS


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _band(x: np.ndarray, sigma: float, band: str) -> np.ndarray:
    if band == "fine":
        return x - gaussian_filter1d(x, sigma)
    if band == "coarse":
        return gaussian_filter1d(x, sigma)
    if band == "full":
        return x
    raise ValueError(f"band must be fine|coarse|full, got {band}")


def perp_likelihood(line, atlas, along: float, band: str = "fine",
                    sigma: float = 6.0, col_step: float = 1.0, rows=None):
    """Atlas-match score of ``line`` over candidate perp rows (along fixed).

    Returns (rows, scores) with scores = NCC in [-1, 1]. ``band`` selects the
    fine (sharp/aliased), coarse (broad/anchor), or full appearance channel.
    """
    A = atlas.ref_map if hasattr(atlas, "ref_map") else np.asarray(atlas)
    H = A.shape[0]
    rows = np.arange(H) if rows is None else np.asarray(rows)
    line = line.detach().cpu().numpy() if isinstance(line, torch.Tensor) else np.asarray(line, float)
    L = len(line)

    perp_t = torch.as_tensor(rows, dtype=torch.float64)
    along_t = torch.full((len(rows),), float(along), dtype=torch.float64)
    R = decoder.render(perp_t, along_t, A, L, col_step=col_step).detach().cpu().numpy()

    obs = _z(_band(line, sigma, band))
    Rb = np.empty_like(R)
    for i in range(R.shape[0]):
        Rb[i] = _z(_band(R[i], sigma, band))
    scores = (Rb * obs[None, :]).mean(1)
    return rows, scores


def top_peaks(scores: np.ndarray, k: int = 5, distance: int = 8,
              height: float | None = None):
    """Indices of the k strongest local maxima (descending score)."""
    h = height if height is not None else np.percentile(scores, 80)
    pk, _ = find_peaks(scores, distance=distance, height=h)
    if len(pk) == 0:
        pk = np.array([int(np.argmax(scores))])
    order = pk[np.argsort(scores[pk])[::-1]]
    return order[:k]


def peak_width_rows(scores: np.ndarray, peak: int | None = None) -> float:
    """FWHM (rows) of the peak at ``peak`` (default global max), half above baseline."""
    p = int(np.argmax(scores)) if peak is None else int(peak)
    base = np.median(scores)
    half = base + (scores[p] - base) / 2.0
    lo = p
    while lo > 0 and scores[lo] > half:
        lo -= 1
    hi = p
    while hi < len(scores) - 1 and scores[hi] > half:
        hi += 1
    return float(hi - lo)


def psr(scores: np.ndarray, exclude: int = 10) -> float:
    """Peak-to-secondary ratio: main peak / strongest peak >exclude rows away."""
    p = int(np.argmax(scores))
    mask = np.abs(np.arange(len(scores)) - p) > exclude
    if not mask.any():
        return np.inf
    second = scores[mask].max()
    return float(scores[p] / (second + 1e-9))


def n_modes(scores: np.ndarray, frac: float = 0.5, distance: int = 8) -> int:
    """Number of significant modes: local maxima whose score exceeds ``frac`` of the
    global-max score (measured above the median baseline). These are the alias modes
    the multimodal belief must carry; a unimodal channel returns 1."""
    base = np.median(scores)
    peak = scores.max()
    thr = base + frac * (peak - base)
    pk, _ = find_peaks(scores, distance=distance, height=thr)
    return int(len(pk))


def is_multimodal(scores: np.ndarray) -> bool:
    return n_modes(scores) >= 2
