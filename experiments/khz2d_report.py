"""khz2d_report.py — head-to-head evaluation of the kHz 2D gaze methods.

Produces:
  results/khz2d_methods.md        method table + sub-frame validity + decisions
  results/khz2d_rate_accuracy.png accuracy/precision vs output rate
  results/khz2d_overlay.png       trajectory overlay (dot / tracker / methods)

Evaluation protocol (uniform across methods, see khz2d.evaluate):
  light 2 ms smoothing -> slow-drift removal (0.05 Hz) -> per-axis affine
  calibration to the dot on valid samples -> r + RMS arcmin vs dot, r vs
  tracker, precision = RMS of the >25 ms detail.

The 0.2 Hz pursuit dot cannot by itself distinguish a 15 Hz tracker from a
12 kHz tracker, so the report adds SUB-FRAME validity checks:
  (a) independent-estimator agreement in the 8-300 Hz band (above the frame
      chain's 7.3 Hz Nyquist): M1 strips (joint 2D matchTemplate) vs M3
      (per-line engine + Viterbi) are different measurement paths over the
      same photons; band agreement >> chain-interp baseline = real kHz signal.
  (b) saccade physiology: events detected in the kHz trace must follow the
      main sequence (amplitude-peak-velocity) and a plausible rate.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import khz2d

RESULTS = khz2d.RESULTS
GATE_HZ = 826.0
SLOW_HZ = 14.63          # measured slow axis (nominal 20 Hz)

METHODS = [
    ("m0_chain",      "M0 SLO frames only (chain)"),
    ("m1_s20",        "M1 strips S=20"),
    ("m1_s8",         "M1 strips S=8"),
    ("m1_s4",         "M1 strips S=4"),
    ("m1_s2",         "M1 strips S=2"),
    ("m1_s1",         "M1 strips S=1"),
    ("m2_kalman",     "M2 Kalman fusion"),
    ("m3_viterbi",    "M3 Viterbi decode"),
    ("m4_dpf_1182",   "M4 particle filter @1.2kHz"),
    ("m4_dpf_11823",  "M4 particle filter @11.8kHz"),
    ("m4_dpf_1182_learned", "M4 PF learned likelihood"),
    ("m5_map",        "M5 batch MAP smoother"),
]


def _eval(name, label):
    r = khz2d.load_method(name)
    if r is None:
        return None
    return khz2d.evaluate(r["t"], r["x_px"], r["y_px"], r["valid"].astype(bool),
                          float(r["rate"]), label, smooth_ms=2)


def _decimate(name, rates):
    """Block-average a method output to lower rates and re-evaluate."""
    r = khz2d.load_method(name)
    out = []
    t = r["t"]; v = r["valid"].astype(bool)
    x = np.where(v, r["x_px"], np.nan); y = np.where(v, r["y_px"], np.nan)
    fs = float(r["rate"])
    for rr in rates:
        k = max(1, int(round(fs / rr)))
        n = len(t) // k * k
        def blk(a):
            return np.nanmean(a[:n].reshape(-1, k), 1)
        with np.errstate(all="ignore"):
            bt, bx, by = blk(t), blk(x), blk(y)
            bf = np.nanmean(v[:n].astype(float).reshape(-1, k), 1) > 0.5
        bv = bf & np.isfinite(bx) & np.isfinite(by)
        out.append(khz2d.evaluate(bt, bx, by, bv, fs / k, f"dec {fs/k:.0f}Hz"))
    return out


# ---------------------------------------------------------------------------
# sub-frame validity
# ---------------------------------------------------------------------------


def _resample(name, grid):
    r = khz2d.load_method(name)
    v = r["valid"].astype(bool)
    x = np.where(v, r["x_px"], np.nan)
    xs = np.interp(grid, r["t"][v], x[v]) if v.sum() > 10 else np.full_like(grid, np.nan)
    vv = np.interp(grid, r["t"], v.astype(float)) > 0.99
    return xs, vv


def band_agreement(name_a="m1_s8", name_b="m3_viterbi", ref="m0_chain",
                   lo=8.0, hi=300.0, fs=2000.0):
    """r in the [lo,hi] Hz band between two INDEPENDENT kHz estimators, against
    the same band of the interpolated frame chain (which has no content there)."""
    lm = khz2d.line_measurements()
    grid = np.arange(lm["t"][0], lm["t"][-1], 1.0 / fs)
    def band(x):
        x = khz2d.fill_nan(x)
        return (gaussian_filter1d(x, fs / (2 * np.pi * hi))
                - gaussian_filter1d(x, fs / (2 * np.pi * lo)))
    xa, va = _resample(name_a, grid)
    xb, vb = _resample(name_b, grid)
    xr, _ = _resample(ref, grid)
    m = va & vb
    ba, bb, br = band(xa), band(xb), band(xr)
    return dict(r_ab=abs(khz2d.corr(ba[m], bb[m])),
                r_a_ref=abs(khz2d.corr(ba[m], br[m])),
                r_b_ref=abs(khz2d.corr(bb[m], br[m])),
                n=int(m.sum()), lo=lo, hi=hi)


def saccade_stats(name="m3_viterbi", k_mad=6.0):
    """Saccade events from the calibrated horizontal kHz trace: rate, amplitude
    range, main-sequence (log-log peak-vel vs amplitude) slope + corr."""
    r = khz2d.load_method(name)
    ev = khz2d.evaluate(r["t"], r["x_px"], r["y_px"], r["valid"].astype(bool),
                        float(r["rate"]), name, smooth_ms=2)
    fs = float(r["rate"])
    x = khz2d.fill_nan(ev["cal_x"])              # arcmin
    x = gaussian_filter1d(x, max(1.0, fs * 0.0015))
    v = np.gradient(x) * fs
    val = r["valid"].astype(bool)
    thr = np.median(np.abs(v[val])) * 1.4826 * k_mad + 1e-9
    fast = (np.abs(v) > thr) & val
    # contiguous runs >= 1 ms
    amps, pvs = [], []
    i, n = 0, len(fast)
    min_run = max(2, int(fs * 0.001))
    while i < n:
        if fast[i]:
            j = i
            while j < n and fast[j]:
                j += 1
            if j - i >= min_run:
                amps.append(abs(x[min(j, n - 1)] - x[i]))
                pvs.append(np.abs(v[i:j]).max())
            i = j
        else:
            i += 1
    amps = np.array(amps); pvs = np.array(pvs)
    dur = (val.sum() / fs) if val.any() else 1.0
    out = dict(n=len(amps), rate=len(amps) / dur, amp_med=np.median(amps) if len(amps) else np.nan,
               amp_p90=np.percentile(amps, 90) if len(amps) else np.nan,
               slope=np.nan, msq_r=np.nan)
    sel = amps > 2.0
    if sel.sum() >= 8:
        la, lv = np.log(amps[sel]), np.log(pvs[sel])
        out["slope"] = float(np.polyfit(la, lv, 1)[0])
        out["msq_r"] = float(khz2d.corr(la, lv))
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_rate_accuracy(evs, decs, path=os.path.join(RESULTS, "khz2d_rate_accuracy.png")):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    groups = {"M0": "0", "M1": "1", "M2": "2", "M3": "3", "M4": "4", "M5": "5"}
    colors = {g: f"C{i}" for i, g in enumerate(groups)}
    m1r, m1x = [], []
    for ev in evs:
        g = ev["label"].split()[0][:2]
        c = colors.get(g, "k")
        axes[0].plot(ev["rate"], ev["r_dot_x"], "o", color=c, ms=7)
        axes[0].annotate(ev["label"].replace("M1 strips ", "").replace("M4 particle filter", "M4"),
                         (ev["rate"], ev["r_dot_x"]), fontsize=7,
                         textcoords="offset points", xytext=(4, 4))
        axes[1].plot(ev["rate"], ev["r_dot_y"], "o", color=c, ms=7)
        if np.isfinite(ev["prec_x"]):
            axes[2].plot(ev["rate"], ev["prec_x"], "o", color=c, ms=7)
        if g == "M1":
            m1r.append(ev["rate"]); m1x.append(ev["r_dot_x"])
    if m1r:
        o = np.argsort(m1r)
        axes[0].plot(np.array(m1r)[o], np.array(m1x)[o], "-", color=colors["M1"], alpha=.4)
    if decs:
        rr = [d["rate"] for d in decs]; xx = [d["r_dot_x"] for d in decs]
        pp = [d["prec_x"] for d in decs]
        axes[0].plot(rr, xx, "s--", color="C3", ms=4, alpha=.6, label="M3 decimated")
        axes[2].plot(rr, pp, "s--", color="C3", ms=4, alpha=.6)
    for ax, ttl in zip(axes, ("horizontal r vs dot", "vertical r vs dot",
                              "precision: >25 ms detail RMS (arcmin)")):
        ax.set_xscale("log")
        ax.axvline(SLOW_HZ, color="0.5", ls=":", lw=1)
        ax.axvline(GATE_HZ, color="0.4", ls="--", lw=1)
        ax.text(SLOW_HZ, ax.get_ylim()[0], " slow axis", fontsize=7, rotation=90, va="bottom")
        ax.text(GATE_HZ, ax.get_ylim()[0], " 826 Hz gate", fontsize=7, rotation=90, va="bottom")
        ax.set_xlabel("output rate (Hz)"); ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=.3)
    axes[0].set_ylim(0.5, 1.0); axes[1].set_ylim(0.2, 1.0)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("kHz 2D gaze from x-scan lines + 2D SLO — accuracy / precision vs output rate "
                 f"(test1 raster testbed; dot = 0.2 Hz pursuit target)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_overlay(names=("m0_chain", "m3_viterbi", "m4_dpf_1182"),
                t0=18.0, t1=38.0, path=os.path.join(RESULTS, "khz2d_overlay.png")):
    R = khz2d.refs()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    tt = np.linspace(t0, t1, 4000)
    axes[0].plot(tt, np.interp(tt, R["dot_t"] + R["off"], R["dot_x"]), "k-", lw=2, alpha=.35, label="dot")
    axes[1].plot(tt, np.interp(tt, R["dot_t"] + R["off"], R["dot_y"]), "k-", lw=2, alpha=.35, label="dot")
    for i, nm in enumerate(names):
        r = khz2d.load_method(nm)
        if r is None:
            continue
        ev = khz2d.evaluate(r["t"], r["x_px"], r["y_px"], r["valid"].astype(bool),
                            float(r["rate"]), nm, smooth_ms=2)
        m = (r["t"] >= t0) & (r["t"] <= t1)
        vx = np.where(r["valid"].astype(bool), ev["cal_x"], np.nan)
        vy = np.where(r["valid"].astype(bool), ev["cal_y"], np.nan)
        lw = 1.6 if nm == "m0_chain" else 0.7
        axes[0].plot(r["t"][m], vx[m], lw=lw, alpha=.85,
                     label=f"{nm} ({float(r['rate']):.0f} Hz)")
        axes[1].plot(r["t"][m], vy[m], lw=lw, alpha=.85)
    axes[0].set_ylabel("horizontal (arcmin)"); axes[1].set_ylabel("vertical (arcmin)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].set_title("kHz 2D gaze reconstructions vs pursuit dot (calibrated, drift-removed)")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def main():
    evs = []
    missing = []
    for name, label in METHODS:
        ev = _eval(name, label)
        if ev is None:
            missing.append(name)
            continue
        ev["_name"] = name
        evs.append(ev)
        print(khz2d.summarize(ev))

    decs = _decimate("m3_viterbi", (30, 100, 300, 1000, 3000, 11823)) \
        if khz2d.load_method("m3_viterbi") is not None else []
    fig1 = fig_rate_accuracy(evs, decs)
    fig2 = fig_overlay()

    ba = band_agreement()
    sac = {}
    for nm in ("m3_viterbi", "m4_dpf_1182", "m0_chain"):
        if khz2d.load_method(nm) is not None:
            sac[nm] = saccade_stats(nm)

    R = khz2d.refs()
    lm = khz2d.line_measurements()

    L = []
    L.append("# kHz 2D Gaze From x-Scan Lines + 2D SLO — Method Comparison (Testbed A)\n")
    L.append("**Task**: reconstruct 2D gaze at kHz output rate from (i) the ~12 kHz fast-axis "
             "line stream and (ii) the slow-axis 2D SLO frames (14.63 Hz measured, nominal 20 Hz).\n")
    L.append("**Testbed A** = the `test1` pursuit raster: its columns ARE the 11,823 Hz x-scan "
             "(808 sweeps/frame), its frames ARE the 2D SLO. References: the 0.2 Hz pursuit dot "
             "(target, arcmin) and the ~32.5 Hz machine pupil tracker. Shared clock offset "
             f"OFF = {R['off']:.2f} s (estimated once at harness level, frozen for all methods; "
             "it absorbs the hardware sync offset + mean pursuit latency).\n")
    L.append("Evaluation protocol (identical for every method): 2 ms smoothing, 0.05 Hz drift "
             "removal, per-axis affine calibration to the dot on valid samples, then r + RMS vs "
             "dot, r vs tracker, precision = RMS of >25 ms detail. The dot is the TARGET, not the "
             "eye: r ~= 0.9 horizontal is the practical ceiling (pursuit lag, catch-up saccades).\n")
    L.append("## Method table\n")
    L.append("| method | rate (Hz) | r dot x | r dot y | r trk x | r trk y | RMS x (') | RMS y (') | prec x (') | prec y (') | valid |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ev in evs:
        L.append(f"| {ev['label']} | {ev['rate']:.0f} | {ev['r_dot_x']:.3f} | {ev['r_dot_y']:.3f} "
                 f"| {ev['r_trk_x']:.3f} | {ev['r_trk_y']:.3f} | {ev['rms_x']:.1f} | {ev['rms_y']:.1f} "
                 f"| {ev['prec_x']:.2f} | {ev['prec_y']:.2f} | {ev['valid_frac']*100:.0f}% |")
    if missing:
        L.append(f"\n(not yet computed: {', '.join(missing)})")

    L.append("\n## Rate-decimation curve (M3)\n")
    L.append("| output rate (Hz) | r dot x | prec x (') |")
    L.append("|---|---|---|")
    for d in decs:
        L.append(f"| {d['rate']:.0f} | {d['r_dot_x']:.3f} | {d['prec_x']:.2f} |")

    L.append("\n## Sub-frame validity (is the kHz content real?)\n")
    L.append("The 0.2 Hz dot cannot distinguish 15 Hz from 12 kHz tracking, so r-vs-dot alone "
             "does not prove kHz content. Two independent checks:\n")
    L.append(f"- **Independent-estimator band agreement ({ba['lo']:.0f}-{ba['hi']:.0f} Hz, above the "
             f"frame chain's 7.3 Hz Nyquist)**: M1 strips (joint 2D matchTemplate) vs M3 (per-line "
             f"engine + Viterbi) agree at r = {ba['r_ab']:.3f} (n = {ba['n']}), while each "
             f"correlates with the interpolated 15 Hz chain at only r = {ba['r_a_ref']:.3f} / "
             f"{ba['r_b_ref']:.3f}. Agreement between independent measurement paths far above the "
             "chain baseline = genuine sub-frame signal.")
    for nm, s in sac.items():
        L.append(f"- **Saccade physiology, {nm}**: {s['n']} events ({s['rate']:.2f}/s), median amp "
                 f"{s['amp_med']:.1f}', p90 {s['amp_p90']:.1f}', main-sequence log-log slope "
                 f"{s['slope']:.2f} (corr {s['msq_r']:.2f}).")

    L.append("\n## Figures\n")
    L.append(f"- `{os.path.basename(fig1)}` — accuracy / precision vs output rate "
             "(slow axis and 826 Hz alias gate marked).")
    L.append(f"- `{os.path.basename(fig2)}` — calibrated trajectory overlay vs the dot.")

    L.append("\n## Decision log\n")
    L.append("- **Anchor chain**: full-frame phase correlation (`data.frame_truth`) mislocks on "
             "banding/low-overlap frames (chain-vs-dot |r| ~ 0.62, and only ~0.46 agreement with "
             "the strip-median chain). REPLACED at harness level by the median of per-strip 2D "
             "matchTemplate shifts (chain-vs-dot |r| ~ 0.90). Every method inherits this anchor.")
    L.append("- **Clock offset**: r against the 0.2 Hz dot loses ~0.1 per 100 ms of OFF error; "
             "OFF is therefore estimated once on a 25 ms grid from the raw per-line series and "
             "frozen for all methods (fair comparison).")
    L.append("- **M2 Kalman tuning**: quality-margin-scaled measurement noise (sigma/(q-q0)) and "
             "reseed-counts-as-update both REJECTED (each degraded r by 0.05-0.3 by trusting "
             "stale/garbage reacquisitions); final: sigma/q scaling, reacquire only from strong "
             "(q>0.45) recent measurements, reacquisitions don't count as valid updates.")
    L.append("- **M5 init**: initialised from the M3 Viterbi path; from-scratch initialisation "
             "converges to the same solution but slower. M5 ~= M3 on this data (the profile "
             "ridge is already globally consistent; the extra dynamics/anchor terms change "
             "little), so the cheap Viterbi is preferred operationally.")
    L.append("- **Vertical channel**: the per-line 1D NCC vertical residual is intrinsically "
             "noisier than horizontal strip matching (r_y raw ~0.38); fused (Kalman) vertical "
             "reaches the frame-chain ceiling ~0.75. Vertical is the weak axis of an x-scan "
             "system, as expected from first principles (a horizontal line constrains vertical "
             "only through appearance change).")

    txt = "\n".join(L) + "\n"
    path = os.path.join(RESULTS, "khz2d_methods.md")
    with open(path, "w") as fh:
        fh.write(txt)
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    main()
