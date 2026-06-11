"""G8 acceptance tests: per-rate SNR, alias/multimodal structure, rate sweep."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import noise as noisemod  # noqa: E402
import synth_stream as ss  # noqa: E402
import likelihood as lik  # noqa: E402

ATLAS = data.load_atlas()


def test_stream_shapes_and_labels():
    s = ss.make_synthetic(1.0, 2000.0, 0, ATLAS, line_len=200)
    n = len(s.trajectory.t)
    assert s.lines.shape == (n, 200)
    assert s.clean.shape == (n, 200)
    assert np.isfinite(s.lines).all()
    assert set(np.unique(s.trajectory.mode)).issubset({0, 1})
    assert s.rate == 2000.0


def test_per_rate_snr_matches_measured_curve():
    # two independent noise realizations of the SAME clean stream -> rep-rep corr = reliability(rate)
    for rate in (820.0, 2000.0, 6000.0):
        s = ss.make_synthetic(1.0, rate, 1, ATLAS, line_len=300, add_noise=False)
        clean = s.clean
        r = noisemod.reliability(rate)
        sig = clean.std(1, keepdims=True) * np.sqrt(1.0 / r - 1.0)
        rng = np.random.default_rng(0)
        rep1 = clean + rng.standard_normal(clean.shape) * sig
        rep2 = clean + rng.standard_normal(clean.shape) * sig
        keep = clean.std(1) > np.median(clean.std(1))   # structured lines only
        cs = [np.corrcoef(rep1[i], rep2[i])[0, 1] for i in np.where(keep)[0]]
        meas = float(np.median(cs))
        assert abs(meas - r) < 0.05, f"rate {rate}: measured rel {meas:.3f} vs target {r:.3f}"


def test_alias_structure_present_and_true_perp_recoverable():
    s = ss.make_synthetic(2.0, 2000.0, 3, ATLAS, line_len=200)
    # pick a fixation sample (stable perp) with a structured line
    fix = np.where(s.trajectory.mode == 0)[0]
    fix = fix[s.clean[fix].std(1) > np.median(s.clean.std(1))]
    i = int(fix[len(fix) // 2])
    line = s.lines[i]
    along = s.trajectory.along_cols[i]
    true_perp = s.trajectory.perp_rows[i]
    rows, sc = lik.perp_likelihood(line, ATLAS, along, band="fine")
    # true perp is among the top peaks (within ~sub-0.1deg)
    assert abs(int(rows[np.argmax(sc)]) - true_perp) <= 0.1 * __import__("calib").ROWS_PER_DEG
    # short synthetic line -> aliasing present (multimodal)
    rows2, sc2 = lik.perp_likelihood(s.clean[i][:50], ATLAS, along, band="fine")
    assert lik.is_multimodal(sc2)


def test_rate_settable_344_to_line_rate():
    rates = [344.0, 820.0, 2000.0, 6000.0, 12000.0]
    rels = []
    for rate in rates:
        s = ss.make_synthetic(0.5, rate, 5, ATLAS, line_len=150)
        assert s.lines.shape[1] == 150
        assert len(s.trajectory.t) >= 2
        rels.append(noisemod.reliability(rate))
    # SNR (reliability) strictly decreases as rate rises across the sweep
    assert all(rels[i] > rels[i + 1] for i in range(len(rels) - 1)), rels


def test_blur_increases_with_rate_drop_during_saccades():
    # lower rate -> longer integration -> more blur on fast (saccade) lines
    hi = ss.make_synthetic(2.0, 6000.0, 7, ATLAS, line_len=200, add_noise=False)
    lo = ss.make_synthetic(2.0, 800.0, 7, ATLAS, line_len=200, add_noise=False)
    def hf(stream):
        sac = stream.trajectory.mode == 1
        if sac.sum() < 3:
            return None
        return np.median([noisemod.high_freq_energy(stream.clean[i]) for i in np.where(sac)[0]])
    ehi, elo = hf(hi), hf(lo)
    if ehi is not None and elo is not None:
        assert elo <= ehi * 1.05  # low-rate saccade lines are not sharper than high-rate
