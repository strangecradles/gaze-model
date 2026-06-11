"""decoder.py — frozen physics decoder (GOALS.md G3).

render(perp, along, atlas) -> line: the 1D line-scan profile a 2D gaze at
(perp, along) produces, by sampling the per-person atlas via DIFFERENTIABLE
bilinear interpolation (torch.grid_sample). NO learned parameters; gradients
flow w.r.t. perp and along so the renderer can sit inside the differentiable
particle filter / analysis-by-synthesis losses.

Coordinate convention (matches data.py / calib.py):
  atlas[perp_row, along_col]  — perp = slow/aliased axis (~125 rows/deg),
                                along = fast/scan axis (trusted, sub-pixel).
A rendered line fixes the perp row and sweeps the along axis: it samples the
atlas at row = perp, columns = along + col_step * (0..L-1). ``perp`` and
``along`` may be fractional (sub-row / sub-column); ``col_step`` scales the
along sampling pitch (atlas-cols per output sample), letting G8 map the atlas
column scale onto the x-scan sweep pitch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(atlas, device=None, dtype=torch.float64) -> torch.Tensor:
    if isinstance(atlas, torch.Tensor):
        t = atlas.to(dtype=dtype)
    else:
        t = torch.as_tensor(np.asarray(atlas), dtype=dtype)
    if device is not None:
        t = t.to(device)
    if t.ndim != 2:
        raise ValueError(f"atlas must be 2D (perp_rows, along_cols); got {tuple(t.shape)}")
    return t


def render(perp, along, atlas, length: int, col_step: float = 1.0,
           dtype=torch.float64) -> torch.Tensor:
    """Render the line(s) produced by gaze (perp, along).

    Parameters
    ----------
    perp, along : float | (B,) tensor — gaze row / along-start (atlas units).
                  Pass tensors with requires_grad=True to backprop through gaze.
    atlas       : (H, W) array/tensor, atlas[perp_row, along_col].
    length      : output line length L (number of along samples).
    col_step    : atlas-columns advanced per output sample (along pitch).

    Returns
    -------
    line : (L,) if perp/along are scalar, else (B, L) — bilinearly sampled,
           gradients flowing w.r.t. perp and along. Uses align_corners=True and
           border padding so a plain clamped-bilinear numpy reference matches
           exactly (see render_numpy).
    """
    A = _to_tensor(atlas, dtype=dtype)
    H, W = A.shape

    perp_t = torch.as_tensor(perp, dtype=dtype)
    along_t = torch.as_tensor(along, dtype=dtype)
    scalar = perp_t.ndim == 0 and along_t.ndim == 0
    perp_t = torch.atleast_1d(perp_t)
    along_t = torch.atleast_1d(along_t)
    perp_t, along_t = torch.broadcast_tensors(perp_t, along_t)
    B = perp_t.shape[0]

    u = torch.arange(length, dtype=dtype)                       # (L,)
    cols = along_t[:, None] + col_step * u[None, :]             # (B, L)
    rows = perp_t[:, None].expand(B, length)                    # (B, L)

    # normalize to grid_sample coords in [-1, 1] with align_corners=True
    gx = 2.0 * cols / (W - 1) - 1.0
    gy = 2.0 * rows / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[:, None, :, :]         # (B, 1, L, 2)

    img = A[None, None].expand(B, 1, H, W)                      # (B, 1, H, W)
    out = F.grid_sample(img, grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    line = out[:, 0, 0, :]                                      # (B, L)
    return line[0] if scalar else line


def render_numpy(perp: float, along: float, atlas, length: int,
                 col_step: float = 1.0) -> np.ndarray:
    """Brute-force clamped-bilinear reference (no torch); for validating render."""
    A = np.asarray(atlas, dtype=np.float64)
    H, W = A.shape
    out = np.empty(length, np.float64)
    for u in range(length):
        col = along + col_step * u
        row = perp
        c = min(max(col, 0.0), W - 1.0)
        r = min(max(row, 0.0), H - 1.0)
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = min(r0 + 1, H - 1), min(c0 + 1, W - 1)
        fr, fc = r - r0, c - c0
        out[u] = (A[r0, c0] * (1 - fr) * (1 - fc) + A[r0, c1] * (1 - fr) * fc +
                  A[r1, c0] * fr * (1 - fc) + A[r1, c1] * fr * fc)
    return out
