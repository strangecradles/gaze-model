"""noise.py — measured-SNR noise model + saccade motion-blur model (GOALS.md G4).

Two physics models applied to rendered atlas lines (decoder.render):

  apply_noise(line, rate)  — additive noise calibrated so a single line's
      reliability (corr between two independent noisy realizations) matches the
      measured per-rate SNR curve. The curve is the Spearman-Brown prediction of
      averaging N = line_rate / rate raw sweeps, anchored on the measured
      single-raw-sweep adjacent-sweep reliability RHO1 ~= 0.70 (sibling
      singleline_snr.py: median 0.70). So reliability(line_rate)=0.70 (N=1) and
      rises toward 1 as the effective rate drops (more averaging) — matching the
      observed ~0.97 split-half at ~200 Hz.

  apply_blur(atlas, perp, along, velocity, dt) — integrates the atlas over the
      gaze displacement during a sweep's integration time dt at the given gaze
      velocity, i.e. averages the rendered line along the path the gaze sweeps
      out. Monotonically attenuates high-spatial-frequency content with velocity
      (the saccade motion blur that erodes the fine perp signature).

Velocities/positions are in ATLAS units (perp rows, along cols); use calib /
data.UNITS to convert deg|arcmin <-> rows. RHO1 and the curve are the measured
substrate G8's synthetic stream and G13's rate sweep rely on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

import decoder

RHO1 = 0.70            # measured single-raw-sweep adjacent reliability
LINE_RATE_HZ = 12000.0  # nominal MEMS line rate (per-capture value in data.UNITS)


# ---------------------------------------------------------------------------
# Measured per-rate SNR curve
# ---------------------------------------------------------------------------


def reliability(rate: float, line_rate: float = LINE_RATE_HZ,
                rho1: float = RHO1) -> float:
    """Reliability (rep-rep correlation) of a single line at effective ``rate``.

    Spearman-Brown over N = line_rate/rate averaged raw sweeps:
        rho_N = N*rho1 / (1 + (N-1)*rho1).
    Monotonically DECREASING in rate (more averaging at low rate -> higher SNR).
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    N = max(1.0, line_rate / rate)
    return float(N * rho1 / (1.0 + (N - 1.0) * rho1))


def snr_power(rate: float, line_rate: float = LINE_RATE_HZ,
              rho1: float = RHO1) -> float:
    """Signal/noise POWER ratio implied by the reliability curve (sigma_s^2/sigma_n^2)."""
    r = reliability(rate, line_rate, rho1)
    return float(r / (1.0 - r)) if r < 1.0 else np.inf


def noise_sigma(line, rate: float, line_rate: float = LINE_RATE_HZ,
                rho1: float = RHO1) -> float:
    """Noise std that makes corr(rep1,rep2)=reliability(rate) for this line.

    With rep = s + n (n independent), corr(rep1,rep2) = sigma_s^2/(sigma_s^2+sigma_n^2)
    = reliability, so sigma_n = sigma_s * sqrt(1/reliability - 1).
    """
    arr = line.detach().cpu().numpy() if isinstance(line, torch.Tensor) else np.asarray(line)
    sigma_s = float(arr.std())
    r = reliability(rate, line_rate, rho1)
    if r >= 1.0:
        return 0.0
    return sigma_s * float(np.sqrt(1.0 / r - 1.0))


def apply_noise(line, rate: float, line_rate: float = LINE_RATE_HZ,
                rho1: float = RHO1, rng: Optional[np.random.Generator] = None,
                seed: Optional[int] = None):
    """Add measured-SNR noise to ``line`` (numpy array or torch tensor).

    Returns the same type as the input. The noise std is set by ``noise_sigma``
    so two independent calls reproduce ``reliability(rate)``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    sigma = noise_sigma(line, rate, line_rate, rho1)
    if isinstance(line, torch.Tensor):
        noise = torch.as_tensor(rng.standard_normal(tuple(line.shape)),
                                dtype=line.dtype, device=line.device) * sigma
        return line + noise
    arr = np.asarray(line, dtype=np.float64)
    return arr + rng.standard_normal(arr.shape) * sigma


# ---------------------------------------------------------------------------
# Saccade motion blur
# ---------------------------------------------------------------------------


def apply_blur(atlas, perp: float, along: float, velocity, dt: float,
               length: int, col_step: float = 1.0, n_sub: int = 16,
               dtype=torch.float64) -> torch.Tensor:
    """Motion-blurred line for gaze (perp, along) moving at ``velocity`` for ``dt``.

    velocity = (v_perp, v_along) in atlas units / second (rows/s, cols/s); the
    gaze sweeps from (perp, along) to (perp + v_perp*dt, along + v_along*dt)
    during the integration window. The blurred line is the mean of ``n_sub``
    rendered lines along that path (box integration over the dwell). Returns a
    differentiable (L,) tensor (frozen decoder; no learned params).
    """
    v_perp, v_along = (float(velocity[0]), float(velocity[1])) if np.ndim(velocity) \
        else (0.0, float(velocity))
    dp = v_perp * dt
    da = v_along * dt
    fracs = (np.arange(n_sub) + 0.5) / n_sub          # box-rule sample centers
    perps = perp + dp * fracs
    alongs = along + da * fracs
    perps_t = torch.as_tensor(perps, dtype=dtype)
    alongs_t = torch.as_tensor(alongs, dtype=dtype)
    lines = decoder.render(perps_t, alongs_t, atlas, length, col_step=col_step,
                           dtype=dtype)                # (n_sub, L)
    return lines.mean(dim=0)


def high_freq_energy(line, sigma: float = 2.0) -> float:
    """Variance of the high-pass-filtered line (proxy for fine spatial detail)."""
    from scipy.ndimage import gaussian_filter1d
    arr = line.detach().cpu().numpy() if isinstance(line, torch.Tensor) else np.asarray(line)
    hp = arr - gaussian_filter1d(arr, sigma)
    return float(hp.var())
