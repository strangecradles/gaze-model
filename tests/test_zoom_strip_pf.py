import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zoom_strip_pf as zsp  # noqa: E402


def test_natural_key_sorts_numbered_tiffs():
    names = ["test_10.tif", "test_2.tif", "test_01.tif"]
    assert sorted(names, key=zsp._natural_key) == ["test_01.tif", "test_2.tif", "test_10.tif"]


def test_zoom_cache_path_separates_width_frames_fps_and_split():
    path = zsp.zoom_strip_cache_path(
        "zoom/live200ashton", 15, 40, 30.0, method="pfref_split", suffix="_evenodd0")
    assert path.endswith("zoom_strip_pfref_split_s15_f40_fps30_evenodd0.npz")
    assert os.path.join("zoom_strip", "live200ashton") in path


def test_immediate_run_from_pf_swaps_trace_without_mutating_source():
    run = {
        "x_px": np.array([1.0, 2.0]),
        "y_px": np.array([3.0, 4.0]),
        "x_px_immediate": np.array([5.0, 6.0]),
        "y_px_immediate": np.array([7.0, 8.0]),
        "valid": np.array([True, True]),
        "t": np.array([0.0, 1.0]),
        "rate": np.float64(2.0),
    }
    raw = zsp.immediate_run_from_pf(run)
    assert raw["x_px"].tolist() == [5.0, 6.0]
    assert raw["y_px"].tolist() == [7.0, 8.0]
    assert run["x_px"].tolist() == [1.0, 2.0]


def test_zoom_valid_uses_quality_and_relative_contrast():
    q = np.array([0.4, 0.2, 0.5])
    con = np.array([10.0, 10.0, 1.0])
    assert zsp._zoom_valid(q, con, 0.35, 0.5).tolist() == [True, False, False]
