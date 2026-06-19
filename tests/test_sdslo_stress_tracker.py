"""SDSLO single-line stress + ambiguity-aware PF tests."""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calib  # noqa: E402
import data  # noqa: E402
import filter as flt  # noqa: E402
import synth_stream as ss  # noqa: E402
import traj_gen  # noqa: E402

ATLAS = data.load_atlas()
RATE = 2000.0
DUR = 0.45
N_PARTICLES = 180


def _gross_persistence_ms(gross: np.ndarray, rate: float) -> float:
    g = gross.astype(np.int8)
    d = np.diff(np.concatenate([[0], g, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    runs = (ends - starts) * 1000.0 / rate
    return float(runs.max()) if runs.size else 0.0


def _run_stressed(seed: int, upgraded: bool):
    stream = ss.make_synthetic(DUR, RATE, seed, ATLAS, stress="combo_sdslo")
    tr = stream.trajectory
    stress = ss.resolve_stress("combo_sdslo")
    rng = np.random.default_rng(9000 + seed)
    along = tr.along_cols + rng.normal(0.0, stress.along_sigma, len(tr.t))
    coarse = tr.perp_rows + rng.normal(0.0, flt.COARSE_SIGMA_ROWS, len(tr.t))
    kw = {}
    if upgraded:
        along_quality = np.full(len(tr.t), 1.0 / stress.along_sigma)
        kw = dict(
            quality_scaled_along=True,
            along_quality=along_quality,
            along_sigma_max=18.0,
            multi_hypothesis=True,
            lag_ms=1.0,
            hypothesis_top_k=5,
            hypothesis_cluster_rows=6.0,
            hypothesis_transition_sigma_rows=3.0,
        )
    res = flt.run(
        stream.lines, along, RATE, ATLAS,
        init_perp=float(tr.perp_rows[0]),
        init_along=float(tr.along_cols[0]),
        n_particles=N_PARTICLES,
        perp_spread=calib.ALIAS_SPACING_ROWS,
        along_spread=2.0,
        line_len=stream.line_len,
        coarse_anchor=coarse,
        seed=seed,
        **kw,
    )
    err = res.est_perp - tr.perp_rows
    fix = tr.mode == 0
    gross = np.abs(err) >= 0.5 * calib.ROWS_PER_DEG
    return dict(
        fix_rms_arcmin=float(np.sqrt(np.mean(err[fix] ** 2)) * calib.ARCMIN_PER_ROW),
        gross=float(np.mean(gross)),
        persist_ms=_gross_persistence_ms(gross, RATE),
    )


def _longest_fix(mode):
    best = (0, 0)
    i, n = 0, len(mode)
    while i < n:
        if mode[i] == 0:
            j = i
            while j < n and mode[j] == 0:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def test_clean_synthetic_is_byte_compatible_with_clean_preset():
    a = ss.make_synthetic(0.2, RATE, 3, ATLAS, line_len=120)
    b = ss.make_synthetic(0.2, RATE, 3, ATLAS, line_len=120, stress="clean")
    assert np.array_equal(a.lines, b.lines)
    assert np.array_equal(a.clean, b.clean)
    assert b.stress is None


def test_combo_sdslo_preset_reduces_context_and_changes_lines():
    clean = ss.make_synthetic(0.2, RATE, 3, ATLAS, line_len=200, add_noise=False)
    combo = ss.make_synthetic(0.2, RATE, 3, ATLAS, stress="combo_sdslo")
    assert combo.line_len == 80
    assert combo.lines.shape[1] == 80
    assert combo.stress.along_sigma == 6.0
    assert not np.allclose(combo.lines, clean.clean[:, :80])


def test_upgraded_pf_reduces_sdslo_stress_persistence_and_fixation_error():
    seeds = (0, 4)
    base = [_run_stressed(seed, upgraded=False) for seed in seeds]
    up = [_run_stressed(seed, upgraded=True) for seed in seeds]
    med_base = {k: float(np.median([r[k] for r in base])) for k in base[0]}
    med_up = {k: float(np.median([r[k] for r in up])) for k in up[0]}
    print(f"\n[SDSLO] baseline {med_base}")
    print(f"[SDSLO] upgraded {med_up}")
    assert med_base["persist_ms"] >= 50.0
    assert med_up["persist_ms"] <= 0.5 * med_base["persist_ms"]
    assert med_up["fix_rms_arcmin"] <= 0.5 * med_base["fix_rms_arcmin"]
    assert med_up["gross"] < med_base["gross"]


def test_upgraded_pf_preserves_clean_fixation():
    rate = 4000.0
    traj = traj_gen.sample_trajectory(0.4, rate, 7)
    a, b = _longest_fix(traj.mode)
    sub = copy.copy(traj)
    for f in ("t", "perp_arcmin", "along_arcmin", "perp_rows", "along_cols",
              "v_perp", "v_along", "mode"):
        setattr(sub, f, getattr(traj, f)[a:b].copy())
    sub.t = sub.t - sub.t[0]
    stream = ss.render_stream(sub, ATLAS, rate=rate, line_len=200, seed=107)
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    rng = np.random.default_rng(57)
    along = ta + rng.normal(0.0, 1.0, len(ta))
    coarse = tp + rng.normal(0.0, flt.COARSE_SIGMA_ROWS, len(tp))

    def run(upgraded):
        kw = {}
        if upgraded:
            kw = dict(quality_scaled_along=True, along_quality=np.ones(len(tp)),
                      multi_hypothesis=True, lag_ms=1.0)
        res = flt.run(
            stream.lines, along, rate, ATLAS,
            init_perp=float(tp[0]), init_along=float(ta[0]),
            n_particles=400, perp_spread=calib.ALIAS_SPACING_ROWS,
            along_spread=2.0, line_len=stream.line_len, coarse_anchor=coarse,
            seed=7, **kw)
        return float(np.sqrt(np.mean((res.est_perp - tp) ** 2)))

    base_rms = run(False)
    up_rms = run(True)
    assert base_rms < 0.1 * calib.ROWS_PER_DEG
    assert up_rms <= base_rms + 1e-9

