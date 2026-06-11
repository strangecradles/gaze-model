"""synth_stream.py — synthetic, rate-sweepable line-scan stream (GOALS.md G8).

Given a known 2D gaze trajectory (traj_gen.Trajectory), sample the per-person
atlas along it with the frozen decoder (decoder.render) + saccade motion blur,
then inject the measured per-rate noise (noise.apply_noise). This is the LABELED
substrate for the decisive synthetic validation (G10-G13): every line has a known
(perp, along, velocity, mode).

The effective rate R sets BOTH the noise (noise.reliability(R): more averaged raw
sweeps at low R -> higher SNR) AND the motion blur integration window dt = 1/R
(the gaze sweeps further during a lower-rate super-sweep). Rate is settable from
~344 Hz up to the line rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import decoder
import noise as noisemod
import traj_gen


@dataclass
class SyntheticStream:
    trajectory: traj_gen.Trajectory  # the KNOWN labels (resampled to `rate`)
    lines: np.ndarray                # (n, line_len) float64 — noisy blurred lines
    clean: np.ndarray                # (n, line_len) float64 — blurred, pre-noise
    rate: float                      # effective stream rate (Hz)
    line_len: int
    col_step: float


def _resample_traj(traj: traj_gen.Trajectory, rate: float) -> traj_gen.Trajectory:
    """Resample a trajectory onto a uniform grid at ``rate`` (linear interp)."""
    if abs(rate - traj.rate) < 1e-6:
        return traj
    dur = traj.t[-1] - traj.t[0]
    n = max(2, int(round(dur * rate)))
    tnew = traj.t[0] + np.arange(n) / rate
    def ip(y):
        return np.interp(tnew, traj.t, y)
    mode = np.round(np.interp(tnew, traj.t, traj.mode.astype(float))).astype(np.int8)
    return traj_gen.Trajectory(
        t=tnew, perp_arcmin=ip(traj.perp_arcmin), along_arcmin=ip(traj.along_arcmin),
        perp_rows=ip(traj.perp_rows), along_cols=ip(traj.along_cols),
        v_perp=ip(traj.v_perp), v_along=ip(traj.v_along), mode=mode, rate=float(rate),
        anchor_row=traj.anchor_row, anchor_col=traj.anchor_col)


def render_stream(traj: traj_gen.Trajectory, atlas, rate: Optional[float] = None,
                  line_len: int = 200, col_step: float = 1.0, n_sub: int = 12,
                  blur: bool = True, add_noise: bool = True,
                  rng: Optional[np.random.Generator] = None, seed: Optional[int] = None,
                  dtype=torch.float64) -> SyntheticStream:
    """Render a synthetic line-scan stream along ``traj`` from ``atlas``.

    Each line is the box-integrated (motion-blurred) atlas render over the gaze
    displacement during dt = 1/rate, then corrupted with noise.apply_noise(rate).
    """
    A = atlas.ref_map if hasattr(atlas, "ref_map") else np.asarray(atlas)
    if rng is None:
        rng = np.random.default_rng(seed)
    rate = float(traj.rate if rate is None else rate)
    traj = _resample_traj(traj, rate)
    n = len(traj.t)
    dt = 1.0 / rate

    perp = traj.perp_rows.astype(np.float64)
    along = traj.along_cols.astype(np.float64)
    if blur:
        fracs = (np.arange(n_sub) + 0.5) / n_sub                 # box-rule centers
        perp_pts = perp[:, None] + traj.v_perp[:, None] * dt * fracs[None, :]
        along_pts = along[:, None] + traj.v_along[:, None] * dt * fracs[None, :]
    else:
        perp_pts = perp[:, None]
        along_pts = along[:, None]
        n_sub = 1

    pp = torch.as_tensor(perp_pts.ravel(), dtype=dtype)
    aa = torch.as_tensor(along_pts.ravel(), dtype=dtype)
    R = decoder.render(pp, aa, A, line_len, col_step=col_step, dtype=dtype)
    clean = R.reshape(n, n_sub, line_len).mean(1).detach().cpu().numpy()

    if add_noise:
        # per-line measured-SNR noise (vectorized): sigma_i from each line's std
        r = noisemod.reliability(rate)
        sig = clean.std(axis=1, keepdims=True) * (np.sqrt(1.0 / r - 1.0) if r < 1 else 0.0)
        lines = clean + rng.standard_normal(clean.shape) * sig
    else:
        lines = clean.copy()

    return SyntheticStream(traj, lines, clean, rate, line_len, col_step)


def make_synthetic(duration: float, rate: float, seed: int, atlas,
                   line_len: int = 200, col_step: float = 1.0, n_sub: int = 12,
                   blur: bool = True, add_noise: bool = True) -> SyntheticStream:
    """Convenience: generate a trajectory at ``rate`` and render its stream."""
    traj = traj_gen.sample_trajectory(duration, rate, seed)
    return render_stream(traj, atlas, rate=rate, line_len=line_len, col_step=col_step,
                         n_sub=n_sub, blur=blur, add_noise=add_noise, seed=seed + 1)


if __name__ == "__main__":
    import argparse
    import data
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=2000.0)
    ap.add_argument("--duration", type=float, default=2.0)
    args = ap.parse_args()
    A = data.load_atlas()
    s = make_synthetic(args.duration, args.rate, 0, A)
    print(f"synthetic stream: {s.lines.shape} @ {s.rate:.0f} Hz; "
          f"reliability {noisemod.reliability(s.rate):.3f}; "
          f"perp range {np.ptp(s.trajectory.perp_rows):.1f} rows, "
          f"saccade frac {np.mean(s.trajectory.mode == 1):.2f}")
