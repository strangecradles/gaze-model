"""Three-way jitter ablation: repeated gaze with per-line observation noise.

The real-data jitter comes from varying (noisy) lines, not RNG on a static line.
Each step uses the same true (perp, along) but independent line noise (~speckle).

Run A: default BETA=20, roughen=0.5, resample on.
Run B: A + resample disabled.
Run C: A + roughen_perp=0 (resample on).
Run D: lock-gated gain (BETA cool + low roughen when locked).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import decoder  # noqa: E402
import filter as flt  # noqa: E402
import noise as noisemod  # noqa: E402

ATLAS = data.load_atlas()
RATE = 14594.0
N_STEPS = 3000
LINE_LEN = 200
TRUE_PERP = 300.0
TRUE_ALONG = 600.0
DT = 1.0 / RATE
# Per-line speckle scale (a.u.). Default 0.15 was uncontrolled; calibrate to the
# measured single-line argmax variance via step0_recalib.py and override with
# the OBS_NOISE_SIG env var to make the ablation quantitatively load-bearing.
OBS_NOISE_SIG = float(os.environ.get("OBS_NOISE_SIG", "0.15"))


def _noisy_static_lines(seed=0, obs_sig=None):
    obs_sig = OBS_NOISE_SIG if obs_sig is None else obs_sig
    clean = decoder.render(
        torch.tensor(TRUE_PERP, dtype=torch.float64),
        torch.tensor(TRUE_ALONG, dtype=torch.float64),
        ATLAS.ref_map, LINE_LEN, col_step=1.0,
    ).detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    lines = np.empty((N_STEPS, LINE_LEN), np.float64)
    for t in range(N_STEPS):
        noisy = clean + rng.normal(0, obs_sig, LINE_LEN)
        lines[t] = noisemod.apply_noise(noisy, RATE, rng=rng)
    return lines


def _run(label, seed=0, obs_sig=None, **pf_kw):
    lines = _noisy_static_lines(seed, obs_sig=obs_sig)
    along = np.full(N_STEPS, TRUE_ALONG)
    coarse = np.full(N_STEPS, TRUE_PERP)
    rng = np.random.default_rng(seed + 1)
    st = flt.init_filter(300, TRUE_PERP, TRUE_ALONG, 2.0, 2.0, rng=rng)
    pf = flt.ParticleFilter(st, ATLAS.ref_map, LINE_LEN, col_step=1.0, **pf_kw)
    est = []
    for t in range(N_STEPS):
        post = pf.step(lines[t], float(along[t]), DT, rng, coarse_anchor=float(coarse[t]))
        est.append(post.est_perp)
    est = np.array(est)
    tail = est[N_STEPS // 4:]
    d2 = np.abs(np.diff(tail, 2))
    return dict(
        label=label,
        std=float(tail.std()),
        med_d2=float(np.median(d2)),
        range=float(tail.max() - tail.min()),
    )


def test_noisy_static_line_ablation():
    """Per-line observation noise → BETA/roughen/resample set the HF floor."""
    a = _run("A_default", beta=flt.BETA, roughen_perp=flt.ROUGHEN_PERP)
    b = _run("B_no_resample", beta=flt.BETA, roughen_perp=flt.ROUGHEN_PERP,
             resample_enabled=False)
    c = _run("C_no_roughen", beta=flt.BETA, roughen_perp=0.0)
    d = _run("D_cool_gain", lock_gated_gain=True)
    e = _run("E_beta5", beta=5.0, roughen_perp=0.08)

    print("\n[jitter ablation] noisy static line @ {:.0f} Hz".format(RATE))
    for r in (a, b, c, d, e):
        print(f"  {r['label']:16s}  std={r['std']:.4f} rows  "
              f"med|d2|={r['med_d2']:.4f}  range={r['range']:.3f}")

    assert a["med_d2"] > e["med_d2"], (
        f"lower BETA should reduce step jitter: A={a['med_d2']}, E={e['med_d2']}")
    assert d["med_d2"] <= a["med_d2"], (
        f"lock-gated should not worsen jitter: A={a['med_d2']}, D={d['med_d2']}")
    assert c["med_d2"] < a["med_d2"], (
        f"roughen=0 should reduce jitter: A={a['med_d2']}, C={c['med_d2']}")
