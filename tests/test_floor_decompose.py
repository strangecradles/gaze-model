"""Regression tests for the floor-decomposition + H2 machinery.

- floor_multiref._rdx_against reproduces the committed rdx for the N-1 reference (j=1).
- The per-line re-reference is the identity at delta=0 (translation-equivariance baseline).
- crossref_coherence returns a finite/sane crossover on a coherent synthetic pair.

The cache-dependent test skips cleanly if the committed Igor cache is absent.
"""
import os

import numpy as np
import pytest

import people_fov_pf as pf

IGOR = "Igor"


def test_rdx_against_reproduces_committed_j1():
    sub = pf.subject_by_name(IGOR)
    if not os.path.exists(sub.line_cache):
        pytest.skip("committed Igor line cache not present")
    import cv2, data, floor_multiref as fm
    ch = pf.build_chain(sub)
    lm = pf.build_line_measurements(sub)
    frames = {}
    for i, raw in pf._read_frames(sub):
        frames[i] = raw
        if i >= 4:
            break
    f = 3
    H, W = frames[f].shape
    dx = -(float(ch["x"][f]) - float(ch["x"][f - 1]))
    dy = -(float(ch["y"][f]) - float(ch["y"][f - 1]))
    refw = fm._warp_ref(frames[f - 1], dx, dy, (H, W))
    cur_db = data._deband(frames[f])[pf.CROPV:H - pf.CROPV]
    prv_db = data._deband(refw)[pf.CROPV:H - pf.CROPV]
    rdx1, _ = fm._rdx_against(cur_db, prv_db)
    committed = lm["rdx"][lm["frame"] == f]
    assert np.max(np.abs(rdx1 - committed)) < 1e-4   # faithful replica


def test_perline_reref_identity_at_zero_delta():
    import h2_power
    rng = np.random.default_rng(0)
    N, L = 200, 161
    prof = rng.standard_normal((N, L)).astype(np.float32)
    lm_sub = {"prof": prof}
    base = np.array([np.argmax(prof[i]) - (L - 1) // 2 for i in range(N)], float)
    # _parab-refined baseline via the same path with delta=0
    rr0 = h2_power.reref_rdx(lm_sub, np.zeros(N))
    rr0b = h2_power.reref_rdx(lm_sub, np.zeros(N))
    assert np.array_equal(rr0, rr0b)                 # deterministic
    # delta=0 must not move the integer argmax
    assert np.all(np.abs(np.round(rr0) - base) < 1.0)


def test_c1_composition_dev_is_mean_zero_and_fracs_bounded():
    sub = pf.subject_by_name(IGOR)
    if not os.path.exists(os.path.join(sub.cache_dir, "lines_multiref3.npz")):
        pytest.skip("Igor multiref cache not present")
    import c1_composition as cc
    out = cc.run(IGOR)
    comp = out["components"]
    for kname in ("alias", "distortion", "template", "chain"):
        assert 0.0 <= comp[kname]["frac"] <= 1.2     # attributions, mild overlap allowed
    assert out["dominant"] == "alias"                # measured: alias dominates C1
    assert comp["alias"]["frac"] > 0.5


def test_crossref_coherence_on_coherent_pair():
    import floor_decompose as fd
    fs = 2000.0
    n = 40000
    t = np.arange(n) / fs
    # shared mid-band tone (survives the function's high-pass) + independent per-series noise
    shared = np.sin(2 * np.pi * 20.0 * t)
    a = shared + np.random.default_rng(2).standard_normal(n) * 0.3
    b = shared + np.random.default_rng(3).standard_normal(n) * 0.3
    m = np.ones(n, bool)
    out = fd.crossref_coherence(t, a, b, m, fs=fs)
    # coherent at the shared tone (10-50 Hz band), incoherent at high freq
    assert out["coh_10_50hz"] > out["coh_50_200hz"]
