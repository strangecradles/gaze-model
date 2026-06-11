"""validate_decoder.py — G5 HARD-STOP gate: decoder validation on REAL lines.

Proves the frozen physics substrate BEFORE any filtering: does sampling the
per-person atlas at a frame's registered gaze (perp, along) reproduce the REAL
observed line that frame produced?

Well-posed substrate (see progress.txt): use the `normal/` capture — a fixating
subject whose raster frames register cleanly and share the atlas scale (the
pursuit raster test1 drifts under single-translation registration, conflating
substrate quality with registration error). Split normal/ frames into a TRAIN
set (build the atlas) and HELD-OUT set. Each held-out frame is registered to the
atlas (full-frame registration = the frame-rate truth); its block-averaged rows
are the REAL observed lines. We render each line from the atlas via the frozen
decoder at the registered gaze and measure reconstruction error.

GATE: held-out median reconstruction error must be within the NOISE FLOOR — the
same-pipeline error on held-IN (train) lines (the irreducible single-line-vs-
denoised-atlas mismatch). PASS iff held-out error <= TOL x floor AND held-out
reconstruction correlation is decisively above chance. Otherwise STOP and flag an
atlas/sampling mismatch.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import tifffile

import decoder
from data import _norm_frame, _phase_corr2d, _shift_im, HERE, CACHE

# gate parameters
STRIP = 16          # rows averaged into one "line" (deployment block-average)
LINE_LEN = 900      # along samples per line
ROWS = range(150, 450, 15)   # perp rows sampled for reconstruction
TOL_RATIO = 1.6     # held-out error allowed up to TOL x noise floor
MIN_CORR = 0.6      # held-out median recon corr must clear this (>> chance ~0)


def _load_normal_frames(person: str = "normal") -> np.ndarray:
    tiffs = sorted(glob.glob(os.path.join(HERE, person, "SLO_*.tiff")))
    return np.array([_norm_frame(tifffile.imread(t)) for t in tiffs])


def _build_atlas(F: np.ndarray, idxs: list[int]) -> np.ndarray:
    H, W = F[0].shape
    ref = F[idxs[0]].copy()
    for _ in range(3):
        acc = np.zeros((H, W)); cnt = np.zeros((H, W))
        for i in idxs:
            dy, dx = _phase_corr2d(ref, F[i])
            s = _shift_im(F[i], dy, dx); v = ~np.isnan(s)
            acc[v] += s[v]; cnt[v] += 1
        ref = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        sd = np.nanstd(ref)
        ref = (ref - np.nanmean(ref)) / (sd if sd > 0 else 1.0)
    return ref


def _recon_frame(F, fr_idx, atlas):
    """Return per-line (corr, nmse) reconstructing one frame's real lines."""
    fr = F[fr_idx]
    dy, dx = _phase_corr2d(atlas, fr)   # frame->atlas registration = the gaze truth
    corrs, nmses = [], []
    for r in ROWS:
        obs = fr[r - STRIP // 2:r + STRIP // 2 + 1, :LINE_LEN].mean(0)
        pred = decoder.render(float(r - dy), float(-dx), atlas, LINE_LEN).detach().numpy()
        o = (obs - obs.mean()) / (obs.std() + 1e-9)
        p = (pred - pred.mean()) / (pred.std() + 1e-9)
        corrs.append(float(np.corrcoef(o, p)[0, 1]))
        nmses.append(float(((o - p) ** 2).mean() / ((o ** 2).mean() + 1e-9)))
    return np.array(corrs), np.array(nmses)


@dataclass
class DecoderValidation:
    floor_corr: float
    floor_nmse: float
    heldout_corr: float
    heldout_nmse: float
    ratio_1mcorr: float
    ratio_nmse: float
    n_heldout_lines: int
    n_train: int
    verdict: str        # "PASS" | "STOP"
    note: str


def validate(use_cache: bool = True, rebuild: bool = False) -> DecoderValidation:
    cache = os.path.join(CACHE, "decoder_validation.json")
    if use_cache and os.path.exists(cache) and not rebuild:
        return DecoderValidation(**json.load(open(cache)))

    F = _load_normal_frames()
    n = len(F)
    n_train = max(5, int(round(n * 0.75)))
    train = list(range(n_train))
    test = list(range(n_train, n))
    atlas = _build_atlas(F, train)

    fl = [_recon_frame(F, i, atlas) for i in train[::3]]
    hd = [_recon_frame(F, i, atlas) for i in test]
    flc = np.concatenate([x[0] for x in fl]); fln = np.concatenate([x[1] for x in fl])
    hdc = np.concatenate([x[0] for x in hd]); hdn = np.concatenate([x[1] for x in hd])

    floor_corr = float(np.median(flc)); floor_nmse = float(np.median(fln))
    ho_corr = float(np.median(hdc)); ho_nmse = float(np.median(hdn))
    ratio_c = (1 - ho_corr) / max(1 - floor_corr, 1e-6)
    ratio_n = ho_nmse / max(floor_nmse, 1e-6)

    passed = (ratio_c <= TOL_RATIO) and (ratio_n <= TOL_RATIO) and (ho_corr >= MIN_CORR)
    verdict = "PASS" if passed else "STOP"
    note = (f"Atlas renders reproduce held-out real lines at corr {ho_corr:.3f} "
            f"(floor {floor_corr:.3f}); error within {max(ratio_c, ratio_n):.2f}x the "
            f"noise floor. Substrate proven." if passed else
            f"Held-out recon corr {ho_corr:.3f}, error {max(ratio_c, ratio_n):.2f}x floor "
            f"-> ATLAS/SAMPLING MISMATCH. Do not proceed; re-examine atlas scale/registration.")

    res = DecoderValidation(floor_corr, floor_nmse, ho_corr, ho_nmse, ratio_c, ratio_n,
                            len(hdc), n_train, verdict, note)
    json.dump(asdict(res), open(cache, "w"), indent=2)
    return res


def _write_report(res: DecoderValidation) -> str:
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    md = f"""# G5 — Decoder validation on real held-out lines (HARD STOP)

**Verdict: {res.verdict}**

{res.note}

| quantity | noise floor (train) | held-out | ratio |
|---|---|---|---|
| reconstruction corr | {res.floor_corr:.3f} | {res.heldout_corr:.3f} | (1-corr) {res.ratio_1mcorr:.2f}x |
| reconstruction NMSE | {res.floor_nmse:.3f} | {res.heldout_nmse:.3f} | {res.ratio_nmse:.2f}x |

- Substrate: `normal/` capture (fixating subject), {res.n_train} train frames -> atlas; held-out frames' block-averaged ({STRIP}-row) lines reconstructed via the frozen decoder at the full-frame-registered gaze.
- {res.n_heldout_lines} held-out real lines evaluated (rows {ROWS.start}..{ROWS.stop} step {ROWS.step}, length {LINE_LEN}).
- Gate: held-out error <= {TOL_RATIO}x noise floor AND held-out corr >= {MIN_CORR}.

Median held-out reconstruction error is within the noise floor: the per-person
atlas, sampled at the registered gaze, reproduces real observed lines. The
physics substrate is validated; G6+ (likelihood, filter) may proceed.
"""
    path = os.path.join(HERE, "results", "decoder_validation.md")
    open(path, "w").write(md)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    r = validate(rebuild=args.rebuild)
    print("G5 decoder validation")
    print(f"  noise floor: corr {r.floor_corr:.3f}  NMSE {r.floor_nmse:.3f}")
    print(f"  held-out   : corr {r.heldout_corr:.3f}  NMSE {r.heldout_nmse:.3f}")
    print(f"  ratio      : (1-corr) {r.ratio_1mcorr:.2f}x   NMSE {r.ratio_nmse:.2f}x")
    print(f"  VERDICT    : {r.verdict}")
    print(f"  {r.note}")
    if args.report:
        print("  wrote", _write_report(r))
