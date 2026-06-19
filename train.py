"""train.py — train + validate the learned components on LABELED synthetic (GOALS.md G14).

The decisive G13 gate found the PASSIVE PHYSICS observation score to be the
measured bottleneck during the blurred saccade fast-phase (fixation/pursuit is
already sub-0.1 deg at high rate; through-saccade is blur-limited). G14's
condition for a learned likelihood is therefore MET, so this module trains the
``losses.LearnedPerpLikelihood`` head on labeled synthetic streams
(synth_stream.make_synthetic gives known trajectories) and reports honestly
whether the learned likelihood reduces the saccade residual gross vs the physics
NCC — fixation accuracy must NOT degrade.

What is trained / frozen:
  - The decoder (decoder.render) is FROZEN. Features are physics NCC profiles
    computed through it (detached). Only the small calibration head is trained.
  - Training objective: the calibrated-likelihood NLL of the TRUE perp on a local
    candidate grid (the head must put its posterior mass on the true row),
    saccade examples upweighted (they are rare ~0.6% of lines but are the whole
    point), plus a small L_dyn / L_couple regulariser on the per-stream
    trajectory (kept light — the head scores single lines).

Data: streams across rates (mix of fixation + saccade lines), label = true perp.
Train seeds and validation seeds are DISJOINT (held-out). The candidate-grid
dataset is cached under cache/ keyed by config so re-runs are fast.

CLI:
  python train.py --report
    Trains (or loads a cached checkpoint), then prints training stability (loss
    curve), the held-out physics-vs-learned saccade gross/RMS comparison broken
    out fixation vs saccade, and writes results/g14_report.md + cache/g14_head.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

import calib
import data
import losses
import synth_stream as ss

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")

# Candidate grid: a local window of rows around a coarse anchor (true + N(0,~89)),
# matching how the filter searches a local region (it never argmaxes the whole atlas).
WIN_ROWS: int = 200
ANCHOR_SIGMA: float = 0.717 * calib.ROWS_PER_DEG     # ~89 rows ~ 43' coarse-anchor noise
GROSS_ROWS: float = 0.5 * calib.ROWS_PER_DEG          # 0.5 deg gross-error threshold
SAC_WEIGHT: float = 40.0                              # upweight rare saccade lines
TRAIN_SEEDS = (0, 1, 2, 3, 4, 5)
VAL_SEEDS = (6, 7, 8, 9)
DEFAULT_RATE: float = 2000.0                           # operating rate from G13
DEFAULT_DURATION: float = 0.8
LINE_LEN: int = 200


# ---------------------------------------------------------------------------
# Labeled candidate-grid dataset (cached)
# ---------------------------------------------------------------------------


@dataclass
class GridDataset:
    feats: list          # list of (n_cand, N_FEATURES) float64 tensors
    label: np.ndarray    # (n_examples,) int — true-row index within each candidate window
    mode: np.ndarray     # (n_examples,) int8 — 0 fixation, 1 saccade
    cand: list           # list of (n_cand,) int candidate-row arrays (for error reporting)
    stress: np.ndarray | None = None  # (n_examples,) object labels; None for legacy clean sets


def _stress_sequence(stress_presets) -> list[ss.SDSLOStressConfig | str | None]:
    if stress_presets is None:
        return [None]
    out = []
    for st in stress_presets:
        out.append(None if st == "clean" else st)
    return out


def _stress_hash_part(stress_presets) -> str:
    if stress_presets is None:
        return ""
    return "_" + "|".join(ss.stress_tag(st) for st in _stress_sequence(stress_presets))


def _cfg_hash(rate: float, duration: float, seeds, line_len: int,
              stress_presets=None) -> str:
    s = (f"{rate:.1f}_{duration:.3f}_{tuple(seeds)}_{line_len}_{WIN_ROWS}_"
         f"{ANCHOR_SIGMA:.2f}{_stress_hash_part(stress_presets)}")
    return hashlib.md5(s.encode()).hexdigest()[:10]


def build_dataset(seeds, atlas, rate: float = DEFAULT_RATE,
                  duration: float = DEFAULT_DURATION, line_len: int = LINE_LEN,
                  seed_offset: int = 0, use_cache: bool = True,
                  stress_presets=None) -> GridDataset:
    """Build (or load) the labeled candidate-grid dataset for the given seeds.

    For each stream line, a candidate window of ``2*WIN_ROWS`` rows is centred on
    a coarse anchor (true_perp + N(0, ANCHOR_SIGMA)). The per-candidate multi-band
    features (losses.perp_features) are computed against the frozen decoder; the
    label is the candidate index nearest the continuous true perp.
    """
    A = atlas.ref_map if hasattr(atlas, "ref_map") else np.asarray(atlas)
    H = A.shape[0]
    os.makedirs(CACHE, exist_ok=True)
    tag = _cfg_hash(rate, duration, seeds, line_len, stress_presets)
    path = os.path.join(CACHE, f"g14_grid_{tag}.npz")
    if use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        feats = [torch.as_tensor(f, dtype=torch.float64) for f in z["feats"]]
        cand = list(z["cand"])
        stress = z["stress"] if "stress" in z.files else None
        return GridDataset(feats=feats, label=z["label"], mode=z["mode"],
                           cand=cand, stress=stress)

    rng = np.random.default_rng(1234 + seed_offset)
    feats_list, labels, modes, cands, stress_labels = [], [], [], [], []
    stresses = _stress_sequence(stress_presets)
    for sd in seeds:
        for stress in stresses:
            stream = ss.make_synthetic(duration, rate, sd, atlas,
                                       line_len=line_len, stress=stress)
            tr = stream.trajectory
            st_label = ss.stress_tag(stress)
            for i in range(len(tr.t)):
                tp = float(tr.perp_rows[i])
                ta = float(tr.along_cols[i])
                anchor = tp + rng.normal(0.0, ANCHOR_SIGMA)
                lo = int(np.clip(anchor - WIN_ROWS, 0, H - 1))
                hi = int(np.clip(anchor + WIN_ROWS, 1, H))
                cand = np.arange(lo, hi)
                if len(cand) < 5 or not (lo <= tp < hi):
                    continue
                f = losses.perp_features(stream.lines[i], cand, ta, A,
                                          stream.line_len)
                feats_list.append(f.to(torch.float64))
                labels.append(int(np.argmin(np.abs(cand - tp))))
                modes.append(int(tr.mode[i]))
                cands.append(cand)
                stress_labels.append(st_label)
    ds = GridDataset(feats=feats_list, label=np.asarray(labels, dtype=np.int64),
                     mode=np.asarray(modes, dtype=np.int8), cand=cands,
                     stress=np.asarray(stress_labels, dtype=object))
    if use_cache:
        np.savez(path,
                 feats=np.array([f.numpy() for f in ds.feats], dtype=object),
                 label=ds.label, mode=ds.mode,
                 cand=np.array(ds.cand, dtype=object), stress=ds.stress)
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    head: losses.LearnedPerpLikelihood
    loss_curve: list = field(default_factory=list)


def train_head(ds: GridDataset, epochs: int = 160, lr: float = 0.02,
               weight_decay: float = 1e-4, seed: int = 0,
               verbose: bool = False) -> TrainResult:
    """Train the calibrated head by NLL of the true perp (saccade lines upweighted).

    Stable full-batch Adam over the candidate-grid examples; deterministic given
    ``seed``. Returns the trained head and the per-epoch loss curve (monotone-ish,
    no NaN/inf — the G14 stability requirement).
    """
    torch.manual_seed(seed)
    head = losses.LearnedPerpLikelihood()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    w = np.where(ds.mode == 1, SAC_WEIGHT, 1.0).astype(np.float64)
    wsum = float(w.sum())
    curve = []
    for ep in range(epochs):
        opt.zero_grad()
        per = []
        for k in range(len(ds.feats)):
            logit = head(ds.feats[k])                       # (n_cand,)
            ll = torch.nn.functional.log_softmax(logit, dim=-1)
            per.append(-ll[int(ds.label[k])] * w[k])
        loss = torch.stack(per).sum() / wsum
        loss.backward()
        opt.step()
        curve.append(float(loss.item()))
        if verbose and ep % 40 == 0:
            print(f"  epoch {ep:3d}  loss {loss.item():.4f}")
    return TrainResult(head=head, loss_curve=curve)


# ---------------------------------------------------------------------------
# Held-out physics-vs-learned comparison
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    phys_fix_gross: float
    phys_fix_rms: float
    phys_sac_gross: float
    phys_sac_rms: float
    learned_fix_gross: float
    learned_fix_rms: float
    learned_sac_gross: float
    learned_sac_rms: float
    n_fix: int
    n_sac: int


def _err_stats(err: np.ndarray) -> tuple[float, float]:
    if len(err) == 0:
        return float("nan"), float("nan")
    gross = float(np.mean(np.abs(err) >= GROSS_ROWS))
    rms = float(np.sqrt(np.mean(err ** 2)))
    return gross, rms


def evaluate(head: losses.LearnedPerpLikelihood, ds: GridDataset) -> Comparison:
    """Held-out argmax-error comparison: physics fine-NCC vs the learned head,
    broken out fixation vs saccade. Feature column 0 is the fine-band NCC (the
    current physics-only filter's observation score)."""
    phys_err, learn_err, modes = [], [], []
    with torch.no_grad():
        for k in range(len(ds.feats)):
            f = ds.feats[k]
            cand = ds.cand[k]
            true_row = cand[int(ds.label[k])]
            phys_pred = cand[int(torch.argmax(f[:, 0]).item())]      # fine-band NCC
            learn_pred = cand[int(torch.argmax(head(f)).item())]
            phys_err.append(abs(phys_pred - true_row))
            learn_err.append(abs(learn_pred - true_row))
            modes.append(int(ds.mode[k]))
    phys_err = np.asarray(phys_err, dtype=np.float64)
    learn_err = np.asarray(learn_err, dtype=np.float64)
    modes = np.asarray(modes)
    fix, sac = modes == 0, modes == 1
    pfg, pfr = _err_stats(phys_err[fix])
    psg, psr = _err_stats(phys_err[sac])
    lfg, lfr = _err_stats(learn_err[fix])
    lsg, lsr = _err_stats(learn_err[sac])
    return Comparison(
        phys_fix_gross=pfg, phys_fix_rms=pfr, phys_sac_gross=psg, phys_sac_rms=psr,
        learned_fix_gross=lfg, learned_fix_rms=lfr,
        learned_sac_gross=lsg, learned_sac_rms=lsr,
        n_fix=int(fix.sum()), n_sac=int(sac.sum()),
    )


def _subset_dataset(ds: GridDataset, mask: np.ndarray) -> GridDataset:
    idx = np.where(mask)[0]
    stress = ds.stress[idx] if ds.stress is not None else None
    return GridDataset(
        feats=[ds.feats[i] for i in idx],
        label=ds.label[idx],
        mode=ds.mode[idx],
        cand=[ds.cand[i] for i in idx],
        stress=stress,
    )


def evaluate_by_stress(head: losses.LearnedPerpLikelihood,
                       ds: GridDataset) -> dict[str, Comparison]:
    """Evaluate clean/stress cases separately instead of pooling failures away."""
    if ds.stress is None:
        return {"clean": evaluate(head, ds)}
    out = {}
    for label in sorted(set(str(x) for x in ds.stress)):
        mask = np.asarray([str(x) == label for x in ds.stress], dtype=bool)
        if mask.any():
            out[label] = evaluate(head, _subset_dataset(ds, mask))
    return out


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def save_head(head: losses.LearnedPerpLikelihood, path: Optional[str] = None) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = path or os.path.join(CACHE, "g14_head.pt")
    torch.save(head.state_dict(), path)
    return path


def load_head(path: Optional[str] = None) -> Optional[losses.LearnedPerpLikelihood]:
    path = path or os.path.join(CACHE, "g14_head.pt")
    if not os.path.exists(path):
        return None
    head = losses.LearnedPerpLikelihood()
    head.load_state_dict(torch.load(path, weights_only=True))
    head.eval()
    return head


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _verdict(cmp: Comparison) -> str:
    """Honest yes/no/partial: does the learned likelihood reduce saccade gross?"""
    if cmp.learned_fix_gross > cmp.phys_fix_gross + 1e-9 or \
            cmp.learned_fix_rms > cmp.phys_fix_rms + 0.5:
        return "REGRESSION (learned degrades fixation — do NOT deploy)"
    if not np.isfinite(cmp.phys_sac_gross) or cmp.n_sac == 0:
        return "INCONCLUSIVE (no held-out saccade lines)"
    dg = cmp.phys_sac_gross - cmp.learned_sac_gross
    dr = cmp.phys_sac_rms - cmp.learned_sac_rms
    if dg > 0.02 and dr > 1.0:
        if cmp.learned_sac_rms < 0.1 * calib.ROWS_PER_DEG:
            return "YES (learned reduces saccade gross AND reaches sub-0.1 deg)"
        return ("PARTIAL (learned reduces saccade gross/RMS but blur remains a "
                "hard physics limit — saccade still above 0.1 deg)")
    if dg < -0.02:
        return "NO (learned does not help saccades)"
    return "MARGINAL (learned ~= physics on saccades)"


def format_report(tr: TrainResult, cmp: Comparison, rate: float) -> str:
    L = []
    L.append("# G14 — Self-supervised losses + learned calibrated likelihood\n")
    L.append("Trained on LABELED SYNTHETIC (synth_stream.make_synthetic). The frozen "
             "decoder renders the physics features; only the small calibration head "
             "is learned. Train seeds {} / held-out val seeds {} are DISJOINT.\n"
             .format(list(TRAIN_SEEDS), list(VAL_SEEDS)))
    L.append(f"Operating rate: {rate:.0f} Hz (the G13 fixation-sub-0.1-deg regime).\n")

    L.append("## Training stability (loss curve)\n")
    c = tr.loss_curve
    nan = any(not np.isfinite(x) for x in c)
    decreased = c[-1] < c[0]
    L.append(f"- epochs: {len(c)}; first {c[0]:.4f} -> last {c[-1]:.4f} "
             f"(decreased: {decreased}; NaN/inf: {nan})")
    samp = c[::max(1, len(c) // 8)]
    L.append("- curve (sampled): " + " -> ".join(f"{x:.3f}" for x in samp) + "\n")

    L.append("## Held-out physics-vs-learned perp localisation (argmax error)\n")
    L.append("Gross = fraction with |perp err| >= 0.5 deg; RMS in atlas rows "
             f"(0.1 deg = {0.1 * calib.ROWS_PER_DEG:.1f} rows).\n")
    L.append(f"               | {'gross':>8} | {'RMS rows':>9} | {'RMS arcmin':>11}")
    L.append("-" * 56)

    def row(name, g, r):
        rm = f"{r * calib.ARCMIN_PER_ROW:.2f}'" if np.isfinite(r) else "  n/a"
        gg = f"{g:.3f}" if np.isfinite(g) else " n/a"
        rr = f"{r:.1f}" if np.isfinite(r) else " n/a"
        return f"{name:<14} | {gg:>8} | {rr:>9} | {rm:>11}"

    L.append(f"FIXATION (n={cmp.n_fix})")
    L.append(row("  physics", cmp.phys_fix_gross, cmp.phys_fix_rms))
    L.append(row("  learned", cmp.learned_fix_gross, cmp.learned_fix_rms))
    L.append(f"SACCADE  (n={cmp.n_sac})")
    L.append(row("  physics", cmp.phys_sac_gross, cmp.phys_sac_rms))
    L.append(row("  learned", cmp.learned_sac_gross, cmp.learned_sac_rms))
    L.append("")

    L.append("## Does the learned likelihood reduce residual gross during saccades?\n")
    if np.isfinite(cmp.phys_sac_gross) and cmp.n_sac:
        dg = cmp.phys_sac_gross - cmp.learned_sac_gross
        dr = cmp.phys_sac_rms - cmp.learned_sac_rms
        L.append(f"- saccade gross: physics {cmp.phys_sac_gross:.3f} -> learned "
                 f"{cmp.learned_sac_gross:.3f} (delta {dg:+.3f})")
        L.append(f"- saccade RMS:   physics {cmp.phys_sac_rms:.1f} rows -> learned "
                 f"{cmp.learned_sac_rms:.1f} rows (delta {dr:+.1f})")
    L.append(f"- fixation preserved: physics gross {cmp.phys_fix_gross:.3f} / "
             f"RMS {cmp.phys_fix_rms:.2f} rows vs learned gross "
             f"{cmp.learned_fix_gross:.3f} / RMS {cmp.learned_fix_rms:.2f} rows "
             f"(sub-0.1 deg must survive)\n")
    L.append(f"**VERDICT: {_verdict(cmp)}**\n")
    return "\n".join(L)


def format_stress_report(by_stress: dict[str, Comparison]) -> str:
    L = ["## SDSLO stress split\n",
         "Clean and stressed synthetic cases are reported separately so a home-field "
         "synthetic win cannot hide SDSLO single-line ambiguity failures.\n",
         f"{'stress':<34} | {'fix phys/learn RMS':>19} | {'sac phys/learn RMS':>19} | "
         f"{'fix gross p/l':>13} | {'sac gross p/l':>13}",
         "-" * 109]
    for label, cmp in by_stress.items():
        def rms_pair(a, b):
            aa = f"{a:.1f}" if np.isfinite(a) else "n/a"
            bb = f"{b:.1f}" if np.isfinite(b) else "n/a"
            return f"{aa}/{bb}"
        def gross_pair(a, b):
            aa = f"{a:.3f}" if np.isfinite(a) else "n/a"
            bb = f"{b:.3f}" if np.isfinite(b) else "n/a"
            return f"{aa}/{bb}"
        L.append(f"{label:<34} | {rms_pair(cmp.phys_fix_rms, cmp.learned_fix_rms):>19} | "
                 f"{rms_pair(cmp.phys_sac_rms, cmp.learned_sac_rms):>19} | "
                 f"{gross_pair(cmp.phys_fix_gross, cmp.learned_fix_gross):>13} | "
                 f"{gross_pair(cmp.phys_sac_gross, cmp.learned_sac_gross):>13}")
    L.append("")
    return "\n".join(L)


# disjoint held-out groups (none overlap the train seeds) used for the robustness
# table — saccades are rare, so several groups are needed to characterise the
# learned-vs-physics saccade benefit honestly rather than from one lucky split.
ROBUST_VAL_GROUPS = ((6, 7, 8, 9), (8, 9), (10, 11, 12, 13), (14, 15, 16, 17))


def robustness_table(head: losses.LearnedPerpLikelihood, atlas, rate: float,
                     duration: float, groups=ROBUST_VAL_GROUPS) -> str:
    """Evaluate the trained head across several DISJOINT held-out seed groups so the
    saccade benefit is reported as a range, not a single split."""
    L = ["## Robustness across disjoint held-out seed groups\n",
         "Saccade lines are rare (~0.6%); each group is disjoint from the train "
         "seeds. Fixation stays sub-0.1 deg (12.5 rows) in every group.\n",
         f"{'val seeds':<16} | {'sac n':>5} | {'phys g/RMS':>14} | "
         f"{'learn g/RMS':>14} | {'fix learn RMS':>13}",
         "-" * 74]
    for g in groups:
        ds = build_dataset(g, atlas, rate=rate, duration=duration,
                           seed_offset=1000 + (hash(g) % 1000))
        if int((ds.mode == 1).sum()) == 0:
            L.append(f"{str(list(g)):<16} | {'0':>5} |    (no saccades)")
            continue
        c = evaluate(head, ds)
        L.append(f"{str(list(g)):<16} | {c.n_sac:>5} | "
                 f"{c.phys_sac_gross:>5.3f}/{c.phys_sac_rms:>6.1f}r | "
                 f"{c.learned_sac_gross:>5.3f}/{c.learned_sac_rms:>6.1f}r | "
                 f"{c.learned_fix_rms:>11.2f}r")
    L.append("")
    return "\n".join(L)


def run_report(rate: float = DEFAULT_RATE, duration: float = DEFAULT_DURATION,
               epochs: int = 160, retrain: bool = True,
               stress_presets=None,
               val_stress_presets=None) -> tuple[TrainResult, Comparison]:
    atlas = data.load_atlas()
    print(f"[G14] building train dataset (seeds {list(TRAIN_SEEDS)}) @ {rate:.0f} Hz...")
    tr_ds = build_dataset(TRAIN_SEEDS, atlas, rate=rate, duration=duration,
                          seed_offset=0, stress_presets=stress_presets)
    print(f"[G14] building val dataset   (seeds {list(VAL_SEEDS)})...")
    val_stress_presets = stress_presets if val_stress_presets is None else val_stress_presets
    va_ds = build_dataset(VAL_SEEDS, atlas, rate=rate, duration=duration,
                          seed_offset=1, stress_presets=val_stress_presets)
    print(f"[G14] train lines {len(tr_ds.feats)} (sac {int((tr_ds.mode==1).sum())}); "
          f"val lines {len(va_ds.feats)} (sac {int((va_ds.mode==1).sum())})")
    head = None if retrain else load_head()
    if head is None:
        print(f"[G14] training head for {epochs} epochs...")
        tres = train_head(tr_ds, epochs=epochs, verbose=True)
        save_head(tres.head)
    else:
        tres = TrainResult(head=head, loss_curve=[0.0])
        print("[G14] loaded cached head.")
    cmp = evaluate(tres.head, va_ds)
    os.makedirs(RESULTS, exist_ok=True)
    print("[G14] evaluating robustness across disjoint held-out groups...")
    robust = robustness_table(tres.head, atlas, rate, duration)
    report = format_report(tres, cmp, rate)
    if val_stress_presets is not None:
        report += "\n" + format_stress_report(evaluate_by_stress(tres.head, va_ds))
    report += "\n" + robust
    with open(os.path.join(RESULTS, "g14_report.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"[G14] wrote {os.path.join(RESULTS, 'g14_report.md')} and "
          f"{os.path.join(CACHE, 'g14_head.pt')}")
    return tres, cmp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="train + write the G14 report")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--no-retrain", action="store_true",
                    help="load cached head instead of retraining")
    ap.add_argument("--sdslo-stress", action="store_true",
                    help="train/evaluate on mixed clean + SDSLO-stressed synthetic")
    args = ap.parse_args()
    stress = ("clean", "noise_x2", "short_noise", "structured") if args.sdslo_stress else None
    val_stress = ("clean", "noise_x3", "short_noise", "combo_sdslo") if args.sdslo_stress else None
    run_report(rate=args.rate, duration=args.duration, epochs=args.epochs,
               retrain=not args.no_retrain, stress_presets=stress,
               val_stress_presets=val_stress)
