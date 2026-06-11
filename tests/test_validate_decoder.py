"""G5 HARD-STOP gate test: the decoder substrate validates on real held-out lines."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import validate_decoder as vd  # noqa: E402


def test_decoder_substrate_passes_hard_stop():
    r = vd.validate()
    # the substrate must be PROVEN: held-out lines reconstruct from the atlas
    assert r.verdict == "PASS", r.note
    assert r.heldout_corr >= vd.MIN_CORR, f"held-out corr {r.heldout_corr}"
    assert r.ratio_1mcorr <= vd.TOL_RATIO
    assert r.ratio_nmse <= vd.TOL_RATIO


def test_reconstruction_decisively_above_chance():
    # corr ~0 would mean a fundamental atlas/sampling mismatch (the STOP case)
    r = vd.validate()
    assert r.heldout_corr > 0.5
    # held-out should not be wildly worse than the irreducible floor
    assert r.heldout_corr > r.floor_corr - 0.2
