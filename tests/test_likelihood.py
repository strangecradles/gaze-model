"""G6 acceptance tests for the multimodal perp observation likelihood.

Criteria, grounded in the measured behaviour of the clean normal/ atlas:
 (a) a synthetic line rendered at a known row -> the true row is the top peak;
 (c) the locked fine peak is sub-0.1 deg wide;
 (b) the belief is MULTIMODAL (the design-critical anti-unimodal-collapse fact) and
     a coarse channel gives a single broad ~0.5 deg anchor lobe — the precision/anchor
     split. (The literal 126-row alias comb is a degraded x-scan property, not present
     on the clean substrate; see likelihood.py / progress.txt.)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import decoder  # noqa: E402
import likelihood as lik  # noqa: E402
import calib  # noqa: E402

ATLAS = data.load_atlas()
SUBTENTH_ROWS = 0.1 * calib.ROWS_PER_DEG   # rows spanning 0.1 deg (~12.5)


def _synth_line(r0, along, L):
    return decoder.render(float(r0), float(along), ATLAS.ref_map, L).detach().numpy()


# (a)
def test_true_row_is_top_peak():
    for r0, along in [(300, 300), (180, 200), (420, 500)]:
        line = _synth_line(r0, along, 700)
        rows, sc = lik.perp_likelihood(line, ATLAS, along, band="fine")
        peaks = lik.top_peaks(sc, k=3)
        assert abs(int(rows[np.argmax(sc)]) - r0) <= 1, (r0, int(rows[np.argmax(sc)]))
        assert r0 in [int(rows[p]) for p in peaks] or abs(int(rows[peaks[0]]) - r0) <= 1


# (c)
def test_locked_peak_sub_tenth_degree():
    line = _synth_line(300, 300, 700)
    rows, sc = lik.perp_likelihood(line, ATLAS, 300, band="fine")
    w = lik.peak_width_rows(sc)
    assert w <= SUBTENTH_ROWS, f"fine peak FWHM {w} rows > {SUBTENTH_ROWS:.1f} (0.1 deg)"
    assert calib.ROWS_PER_DEG * 0.1 > w  # restate: width < 0.1 deg


# (b) multimodality
def test_belief_is_multimodal():
    # a short, low-context line (real-sweep extent) is strongly multimodal (alias peaks)
    line = _synth_line(300, 300, 40)
    rows, sc = lik.perp_likelihood(line, ATLAS, 300, band="fine")
    assert lik.is_multimodal(sc), "fine likelihood should be multimodal (>=2 modes)"
    assert lik.n_modes(sc, frac=0.5) >= 2
    # true row still wins despite aliasing
    assert abs(int(rows[np.argmax(sc)]) - 300) <= 1
    # PSR is finite and modest (not a unimodal delta)
    p = lik.psr(sc)
    assert np.isfinite(p) and p < 12


# (b) coarse anchor vs fine precision split
def test_coarse_is_broad_anchor_fine_is_sharp():
    line = _synth_line(300, 300, 700)
    _, sc_fine = lik.perp_likelihood(line, ATLAS, 300, band="fine")
    _, sc_coarse = lik.perp_likelihood(line, ATLAS, 300, band="coarse")
    wf = lik.peak_width_rows(sc_fine)
    wc = lik.peak_width_rows(sc_coarse)
    assert wc > 3 * max(wf, 1.0), f"coarse width {wc} not >> fine width {wf}"
    # coarse anchor localizes within ~0.5 deg of true
    assert abs(int(np.argmax(sc_coarse)) - 300) <= 0.6 * calib.ROWS_PER_DEG


def test_alias_constant_exposed():
    assert np.isfinite(lik.ALIAS_SPACING_ROWS)
    assert 90 <= lik.ALIAS_SPACING_ROWS <= 160
    assert lik.ALIAS_SPACING_ROWS == calib.ALIAS_SPACING_ROWS


def test_nothing_learned_scores_in_range():
    line = _synth_line(250, 400, 600)
    _, sc = lik.perp_likelihood(line, ATLAS, 400, band="fine")
    assert sc.max() <= 1.0 + 1e-6 and sc.min() >= -1.0 - 1e-6
    # self-match at true row is ~1
    assert sc[250] > 0.9
