import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import people_strip_ladder as psl  # noqa: E402


def test_parse_strip_widths_preserves_order_and_dedupes():
    assert psl.parse_strip_widths("32, 16,15,16,1") == [32, 16, 15, 1]


def test_parse_methods_preserves_order_and_dedupes():
    assert psl.parse_methods("raw,pf,resolver,raw") == ["raw", "pf", "resolver"]


def test_strip_rate_uses_complete_non_overlapping_strips():
    fps = 15031.283905967452 / 808.0
    assert np.isclose(psl.strip_rate_hz(808, fps, 15), 53 * fps)
    assert np.isclose(psl.strip_rate_hz(808, fps, 1), 808 * fps)


def test_valid_mask_uses_quality_and_subject_contrast_floor():
    q = np.array([0.4, 0.2, 0.5, 0.6])
    con = np.array([10.0, 10.0, 1.0, 20.0])
    valid = psl.strip_valid_mask(q, con, quality_thr=0.35, contrast_frac=0.5)
    assert valid.tolist() == [True, False, False, True]


def test_strip_cache_path_keeps_duration_runs_separate():
    class Sub:
        cache_dir = "/tmp/cache"

    assert psl.strip_cache_path(Sub(), 15).endswith("strip_ladder_s15.npz")
    assert psl.strip_cache_path(Sub(), 15, 2.5).endswith("strip_ladder_s15_d2p5.npz")
    assert psl.strip_cache_path(
        Sub(), 15, 2.5, method="resolver"
    ).endswith("strip_ladder_resolver_s15_d2p5.npz")
    variant = psl.resolver_variant_name(top_k=7, obs_weight=8.0)
    assert variant == "resolver_k7_ow8"
    assert psl.strip_cache_path(
        Sub(), 15, 2.5, method=variant
    ).endswith("strip_ladder_resolver_k7_ow8_s15_d2p5.npz")
    pf_variant = psl.pf_variant_name(n_particles=80, beta=12.0, obs_weight=8.0)
    assert pf_variant == "pf_n80_b12_ow8"
    assert psl.strip_cache_path(
        Sub(), 15, 2.5, method=pf_variant
    ).endswith("strip_ladder_pf_n80_b12_ow8_s15_d2p5.npz")


def test_sample_response_bilinear_interpolates_and_fills():
    r = np.arange(9, dtype=np.float64).reshape(3, 3)
    y = np.array([0.5, 2.5])
    x = np.array([0.5, 1.0])
    out = psl.sample_response_bilinear(r, y, x, fill=-1.0)
    assert np.allclose(out, [2.0, -1.0])


def test_topk_response_peaks_returns_suppressed_local_maxima():
    r = np.zeros((11, 11), dtype=np.float64)
    r[5, 5] = 1.0
    r[5, 6] = 0.9
    r[2, 8] = 0.8
    dy, dx, q = psl.topk_response_peaks(r, 2, pad=5, suppress_radius=1)
    assert q.tolist() == [1.0, 0.8]
    assert abs(dy[0]) < 0.1
    assert 0.0 < dx[0] < 0.5
    assert np.allclose([dy[1], dx[1]], [-3.0, 3.0])
