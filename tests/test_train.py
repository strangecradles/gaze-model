"""G14 training tests (train.py + filter integration).

- Training runs a few epochs and the loss DECREASES (stable, no NaN/inf).
- On held-out synthetic, the learned likelihood does NOT degrade fixation (its
  fixation gross is <= the physics fine-NCC gross) AND the saccade comparison is
  MEASURED and PRINTED honestly (the PLAN-prescribed physics-vs-learned result).
- The learned likelihood integrated into filter.py preserves fixation sub-0.1 deg
  and the physics default is unchanged (no regression).

Datasets are cached (cache/g14_grid_*.npz) so re-runs are fast; a small seed/epoch
budget keeps runtime modest.
"""
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
import train  # noqa: E402

ATLAS = data.load_atlas()
RATE = 2000.0
DUR = 0.5
LINE_LEN = 200
DEG_THRESH_ROWS = 0.1 * calib.ROWS_PER_DEG

# Small held-out budget for test speed (cached after first build). The val seeds
# are chosen to reliably contain saccade lines (saccades are rare ~0.6% of lines;
# seeds 4,5 carry ~59 saccade lines at 0.5 s, seeds 6,7 happen to carry none) so
# the physics-vs-learned saccade comparison is actually measurable. Train/val
# seed sets are DISJOINT (held-out).
_TR_SEEDS = (0, 1, 2)
_VA_SEEDS = (4, 5)


def _datasets():
    tr = train.build_dataset(_TR_SEEDS, ATLAS, rate=RATE, duration=DUR,
                             line_len=LINE_LEN, seed_offset=0)
    va = train.build_dataset(_VA_SEEDS, ATLAS, rate=RATE, duration=DUR,
                             line_len=LINE_LEN, seed_offset=1)
    return tr, va


def test_training_loss_decreases_stable():
    tr_ds, _ = _datasets()
    res = train.train_head(tr_ds, epochs=40, seed=0)
    c = res.loss_curve
    assert all(np.isfinite(x) for x in c), "loss curve has NaN/inf"
    assert c[-1] < c[0], f"loss did not decrease ({c[0]:.3f} -> {c[-1]:.3f})"
    print(f"\n[G14] train loss {c[0]:.3f} -> {c[-1]:.3f} over {len(c)} epochs (stable)")


def test_learned_not_worse_than_physics_on_fixation_and_report_saccade():
    """ASSERT: learned fixation gross <= physics fixation gross (no degradation).
    MEASURE + PRINT: the saccade gross/RMS delta honestly (no fake positive)."""
    tr_ds, va_ds = _datasets()
    res = train.train_head(tr_ds, epochs=120, seed=0)
    cmp = train.evaluate(res.head, va_ds)

    print(f"\n[G14] held-out (val seeds {list(_VA_SEEDS)}): "
          f"fix n={cmp.n_fix}, sac n={cmp.n_sac}")
    print(f"[G14] FIXATION physics gross {cmp.phys_fix_gross:.3f} RMS "
          f"{cmp.phys_fix_rms:.2f}r | learned gross {cmp.learned_fix_gross:.3f} "
          f"RMS {cmp.learned_fix_rms:.2f}r")
    print(f"[G14] SACCADE  physics gross {cmp.phys_sac_gross:.3f} RMS "
          f"{cmp.phys_sac_rms:.1f}r | learned gross {cmp.learned_sac_gross:.3f} "
          f"RMS {cmp.learned_sac_rms:.1f}r")
    if np.isfinite(cmp.phys_sac_gross) and cmp.n_sac:
        print(f"[G14] SACCADE delta: gross {cmp.phys_sac_gross - cmp.learned_sac_gross:+.3f}, "
              f"RMS {cmp.phys_sac_rms - cmp.learned_sac_rms:+.1f}r  "
              f"(verdict: {train._verdict(cmp)})")

    # DoD: learned must NOT degrade fixation (the sub-0.1 deg regime must survive).
    assert cmp.learned_fix_gross <= cmp.phys_fix_gross + 1e-9, (
        f"learned fixation gross {cmp.learned_fix_gross:.3f} worse than physics "
        f"{cmp.phys_fix_gross:.3f}")
    assert cmp.learned_fix_rms < DEG_THRESH_ROWS, (
        f"learned fixation RMS {cmp.learned_fix_rms:.2f}r not sub-0.1 deg")
    # the saccade comparison must have been measured (finite numbers reported)
    assert cmp.n_sac > 0 and np.isfinite(cmp.learned_sac_gross), (
        "saccade comparison must be measured on held-out lines")


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


def _fixation_stream(seed: int, rate: float = 4000.0, dur: float = 0.3):
    tr = traj_gen.sample_trajectory(dur, rate, seed)
    a, b = _longest_fix(tr.mode)
    sub = copy.copy(tr)
    for f in ("t", "perp_arcmin", "along_arcmin", "perp_rows", "along_cols",
              "v_perp", "v_along", "mode"):
        setattr(sub, f, getattr(tr, f)[a:b].copy())
    sub.t = sub.t - sub.t[0]
    return ss.render_stream(sub, ATLAS, rate=rate, line_len=LINE_LEN, seed=seed + 100)


def test_filter_learned_likelihood_preserves_fixation_sub_0p1deg():
    """The learned observation model integrated into filter.run preserves the G10
    fixation accuracy (sub-0.1 deg) and the physics default is unchanged."""
    head = train.load_head()
    if head is None:
        # train + cache a head so the test is self-contained
        tr_ds, _ = _datasets()
        head = train.train_head(tr_ds, epochs=120, seed=0).head
        train.save_head(head)

    stream = _fixation_stream(seed=7, rate=4000.0, dur=0.3)
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    rng = np.random.default_rng(57)
    am = ta + rng.normal(0.0, 1.0, len(ta))

    def _run(mode, hd):
        res = flt.run(stream.lines, am, 4000.0, ATLAS,
                      init_perp=float(tp[0]), init_along=float(ta[0]),
                      n_particles=400, perp_spread=calib.ALIAS_SPACING_ROWS,
                      along_spread=2.0, line_len=LINE_LEN, seed=7,
                      likelihood=mode, learned_head=hd)
        return float(np.sqrt(np.mean((res.est_perp - tp) ** 2)))

    phys_rms = _run("physics", None)
    learn_rms = _run("learned", head)
    print(f"\n[G14] filter fixation perp RMS: physics {phys_rms:.3f}r "
          f"({phys_rms * calib.ARCMIN_PER_ROW:.3f}') | learned {learn_rms:.3f}r "
          f"({learn_rms * calib.ARCMIN_PER_ROW:.3f}') (DoD {DEG_THRESH_ROWS:.1f}r)")
    assert phys_rms < DEG_THRESH_ROWS, "physics default regressed on fixation"
    assert learn_rms < DEG_THRESH_ROWS, (
        f"learned likelihood degraded fixation: {learn_rms:.3f}r >= 0.1 deg")


def test_learned_likelihood_requires_head():
    """filter.run(likelihood='learned') without a head is a clear error; the
    physics default needs no head (no regression to the existing API)."""
    stream = _fixation_stream(seed=7, rate=4000.0, dur=0.1)
    ta = stream.trajectory.along_cols
    raised = False
    try:
        flt.run(stream.lines, ta, 4000.0, ATLAS,
                init_perp=float(stream.trajectory.perp_rows[0]),
                init_along=float(ta[0]), n_particles=50, line_len=LINE_LEN,
                seed=0, likelihood="learned", learned_head=None)
    except ValueError:
        raised = True
    assert raised, "likelihood='learned' without a head must raise ValueError"
